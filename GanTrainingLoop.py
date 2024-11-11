import json
import os
import pickle

import keras
import tensorflow as tf


@tf.function
@keras.saving.register_keras_serializable()
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
def gradient_penalty(critic, dirty_images, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated_clean_images = epsilon * real_images + (1 - epsilon) * fake_images
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated_clean_images)
        pred = critic([dirty_images, interpolated_clean_images], training=True)
    grads = gp_tape.gradient(pred, [interpolated_clean_images])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((grad_l2 - 1.0) ** 2)
    return gp


def save_training_state(generator, discriminator, gen_optimizer, disc_optimizer, epoch, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save generator and discriminator models
    generator.save(os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.keras'))
    discriminator.save(os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.keras'))

    # Save optimizer states
    gen_optimizer_config = gen_optimizer.get_config()
    disc_optimizer_config = disc_optimizer.get_config()

    # save with pickle
    with open(os.path.join(checkpoint_dir, f'optimizer_gen_epoch_{epoch}.pkl'), 'wb') as f:
        pickle.dump(gen_optimizer_config, f)

    with open(os.path.join(checkpoint_dir, f'optimizer_disc_epoch_{epoch}.pkl'), 'wb') as f:
        pickle.dump(disc_optimizer_config, f)

    # Save the last epoch number
    with open(os.path.join(checkpoint_dir, 'last_epoch.txt'), 'w') as f:
        f.write(str(epoch))


# Define function to load the training state
def load_training_state(generator, discriminator, gen_optimizer, disc_optimizer, checkpoint_dir='checkpoints'):
    # Check if a saved epoch exists
    if os.path.exists(os.path.join(checkpoint_dir, 'last_epoch.txt')):
        with open(os.path.join(checkpoint_dir, 'last_epoch.txt'), 'r') as f:
            last_epoch = int(f.read())
        print(f"Resuming training from epoch {last_epoch + 1}")

        # Load generator and discriminator models
        generator = tf.keras.models.load_model(os.path.join(checkpoint_dir, f'generator_epoch_{last_epoch}.keras'))
        discriminator = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, f'discriminator_epoch_{last_epoch}.keras'))

        # Load optimizer states
        with open(os.path.join(checkpoint_dir, f'optimizer_gen_epoch_{last_epoch}.pkl'), 'rb') as f:
            gen_optimizer_config = pickle.load(f)

        with open(os.path.join(checkpoint_dir, f'optimizer_disc_epoch_{last_epoch}.pkl'), 'rb') as f:
            disc_optimizer_config = pickle.load(f)

        # Load optimizer states
        gen_optimizer = gen_optimizer.from_config(gen_optimizer_config)

        disc_optimizer = disc_optimizer.from_config(disc_optimizer_config)

        return generator, discriminator, gen_optimizer, disc_optimizer, last_epoch
    else:
        print("No saved state found, starting from scratch.")
        return generator, discriminator, gen_optimizer, disc_optimizer, 0


def train_gan(generator, discriminator, gen_optimizer, disc_optimizer, dataset, val_dataset, epochs, callbacks,
              log_interval=10, steps_per_epoch=250, steps_per_log=10, disc_steps=3, lambda_l1=0.1, lambda_gp=10.0,
              accumulation_steps=4, resume=False, checkpoint_dir='checkpoints'):
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    if resume:
        # Load the training state if resuming
        generator, discriminator, gen_optimizer, disc_optimizer, start_epoch = load_training_state(
            generator, discriminator, gen_optimizer, disc_optimizer, checkpoint_dir
        )
    else:
        start_epoch = 0  # Start from scratch if not resuming

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks, add_history=True, model=generator)

    logs = {}
    callback_list.on_train_begin(logs=logs)

    generator.compile(optimizer=gen_optimizer)
    discriminator.compile(optimizer=disc_optimizer)

    generator.summary()
    discriminator.summary()

    accumulated_grads_g = [tf.zeros_like(var) for var in generator.trainable_variables]
    accumulated_grads_d = [tf.zeros_like(var) for var in discriminator.trainable_variables]

    for epoch in range(epochs):
        epoch += start_epoch

        callback_list.on_epoch_begin(epoch, logs=logs)
        print(f"Epoch {epoch + 1}/{ start_epoch + epochs}")

        steps_in_epoch = steps_per_epoch or len(dataset)
        steps_in_val = steps_per_log or len(val_dataset)

        disc_step_count = 0
        g_loss = tf.constant(-1.0)

        for step, (clean_images, dirty_images) in enumerate(dataset.take(steps_in_epoch)):
            callback_list.on_batch_begin(step, logs=logs)
            callback_list.on_train_batch_begin(step, logs=logs)

            transformed_images = generator(dirty_images, training=True)

            # Accumulate Discriminator gradients
            with tf.GradientTape() as tape_d:
                real_pred = discriminator([dirty_images, clean_images], training=True)
                fake_pred = discriminator([dirty_images, transformed_images], training=True)

                identity_loss = sum(discriminator.losses)
                gp = gradient_penalty(discriminator, dirty_images, clean_images, transformed_images)

                d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + identity_loss + lambda_gp * gp

            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            accumulated_grads_d = [
                accum_grad + (grad if grad is not None else tf.zeros_like(accum_grad))
                for accum_grad, grad in zip(accumulated_grads_d, grads_d)
            ]

            disc_step_count += 1
            if disc_step_count >= disc_steps:
                disc_step_count = 0

                # Accumulate Generator gradients
                with tf.GradientTape() as tape_g:
                    generated_images = generator(dirty_images, training=True)
                    gan_output = discriminator([dirty_images, generated_images], training=True)
                    identity_loss = sum(generator.losses)
                    g_loss = -tf.reduce_mean(gan_output) + lambda_l1 * tf.reduce_mean(
                        tf.abs(clean_images - generated_images)) + identity_loss

                grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
                accumulated_grads_g = [
                    accum_grad + (grad if grad is not None else tf.zeros_like(accum_grad))
                    for accum_grad, grad in zip(accumulated_grads_g, grads_g)
                ]

            # Apply gradients after accumulation steps
            if (step + 1) % accumulation_steps == 0:
                disc_optimizer.apply_gradients(
                    [(grad / accumulation_steps, var) for grad, var in
                     zip(accumulated_grads_d, discriminator.trainable_variables)]
                )
                accumulated_grads_d = [tf.zeros_like(var) for var in discriminator.trainable_variables]

                gen_optimizer.apply_gradients(
                    [(grad / accumulation_steps, var) for grad, var in
                     zip(accumulated_grads_g, generator.trainable_variables)]
                )
                accumulated_grads_g = [tf.zeros_like(var) for var in generator.trainable_variables]

            logs = {'d_loss': d_loss, 'g_loss': g_loss}
            callback_list.on_train_batch_end(step, logs=logs)

            if step % log_interval == 0:
                print(f"Step: {step + log_interval}/{steps_in_epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        val_mse = 0
        val_step = 0
        for val_step, (val_real_images, val_dirty_images) in enumerate(val_dataset.take(steps_in_val)):
            callback_list.on_test_batch_begin(val_step, logs=logs)
            val_fake_images = generator(val_dirty_images, training=False)
            val_mse += tf.reduce_mean(tf.square(val_real_images - val_fake_images)).numpy()
            callback_list.on_test_batch_end(val_step, logs=logs)

        val_mse /= (val_step + 1)
        print(f"Validation MSE after Epoch {epoch + 1}: {val_mse:.4f}")

        logs = {'d_loss': d_loss, 'g_loss': g_loss, 'val_mse': val_mse}

        callback_list.on_epoch_end(epoch, logs=logs)
        save_training_state(generator, discriminator, gen_optimizer, disc_optimizer, epoch, checkpoint_dir)

    callback_list.on_train_end(logs=logs)
