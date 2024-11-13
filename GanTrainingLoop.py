import json
import os
import pickle

import keras
import tensorflow as tf
from tqdm import tqdm


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


# Optimized Training Step Functions
@tf.function
def generator_step(generator, discriminator, gen_optimizer, dirty_images, accumulation_steps):
    accumulated_grads_g = [tf.zeros_like(var) for var in generator.trainable_variables]
    gen_loss = 0.0

    for _ in range(accumulation_steps):  # Loop through accumulation steps
        with tf.GradientTape() as tape_g:
            generated_images = generator(dirty_images, training=True)
            gan_output = discriminator([dirty_images, generated_images], training=True)
            g_loss = -tf.reduce_mean(gan_output)

        grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
        # Accumulate gradients
        accumulated_grads_g = [
            accum_grad + (grad if grad is not None else tf.zeros_like(accum_grad))
            for accum_grad, grad in zip(accumulated_grads_g, grads_g)
        ]
        gen_loss += g_loss

    # Apply accumulated gradients
    averaged_grads_g = [grad / accumulation_steps for grad in accumulated_grads_g]
    gen_optimizer.apply_gradients(zip(averaged_grads_g, generator.trainable_variables))
    gen_loss /= accumulation_steps  # Average loss over accumulated steps
    return gen_loss


# Define Discriminator Training Step with Gradient Accumulation
@tf.function
def discriminator_step(discriminator, generator, disc_optimizer, dirty_images, clean_images, lambda_gp, disc_steps,
                       accumulation_steps):
    disc_loss = 0.0

    for disc_step in range(disc_steps):
        accumulated_grads_d = [tf.zeros_like(var) for var in discriminator.trainable_variables]
        step_loss = 0.0

        for acc_step in range(accumulation_steps):
            with tf.GradientTape() as tape_d:
                transformed_images = generator(dirty_images, training=True)
                real_pred = discriminator([dirty_images, clean_images], training=True)
                fake_pred = discriminator([dirty_images, transformed_images], training=True)

                gp = gradient_penalty(discriminator, dirty_images, clean_images, transformed_images)
                d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + lambda_gp * gp

            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            accumulated_grads_d = [
                accum_grad + (grad if grad is not None else tf.zeros_like(accum_grad))
                for accum_grad, grad in zip(accumulated_grads_d, grads_d)
            ]
            step_loss += d_loss

        # Apply accumulated gradients after accumulation steps
        averaged_grads_d = [grad / accumulation_steps for grad in accumulated_grads_d]
        disc_optimizer.apply_gradients(zip(averaged_grads_d, discriminator.trainable_variables))
        disc_loss += step_loss / accumulation_steps  # Average loss over accumulation steps

    disc_loss /= disc_steps  # Average loss over discriminator steps
    return disc_loss


# Updated Train GAN Loop with Generator First
def train_gan(
        generator, discriminator, gen_optimizer, disc_optimizer,
        dataset, val_dataset, epochs, callbacks, log_interval=10,
        steps_per_epoch=250, steps_per_log=10, disc_steps=3, lambda_gp=10.0,
        accumulation_steps=4, resume=False, checkpoint_dir='checkpoints'
):
    # Load previous training state if required
    if resume:
        generator, discriminator, gen_optimizer, disc_optimizer, start_epoch = load_training_state(
            generator, discriminator, gen_optimizer, disc_optimizer, checkpoint_dir)
    else:
        start_epoch = 0  # Start from scratch if not resuming

    # Prepare callbacks and start training
    callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=generator)
    logs = {}
    callback_list.on_train_begin(logs=logs)

    # Initialize progress bar for monitoring
    for epoch in range(start_epoch, epochs):
        callback_list.on_epoch_begin(epoch, logs=logs)
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for step, (clean_images, dirty_images) in tqdm(enumerate(dataset.take(steps_per_epoch)), total=steps_per_epoch,
                                                       leave=False):
            # Generator training step with gradient accumulation first
            g_loss = generator_step(generator, discriminator, gen_optimizer, dirty_images, accumulation_steps)

            # Discriminator training step with gradient accumulation
            d_loss = discriminator_step(discriminator, generator, disc_optimizer, dirty_images, clean_images, lambda_gp,
                                        disc_steps, accumulation_steps)

            # Logging
            logs.update({'d_loss': d_loss, 'g_loss': g_loss})
            callback_list.on_train_batch_end(step, logs=logs)

            if step % log_interval == 0:
                tqdm.write(f"Step: {step}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        # Validation loop
        val_mse = tf.metrics.Mean()
        for val_step, (val_real_images, val_dirty_images) in enumerate(val_dataset.take(steps_per_log)):
            val_fake_images = generator(val_dirty_images, training=False)
            val_mse.update_state(tf.reduce_mean(tf.square(val_real_images - val_fake_images)))

        print(f"Validation MSE after Epoch {epoch + 1}: {val_mse.result():.4f}")
        logs.update({'val_mse': val_mse.result()})

        callback_list.on_epoch_end(epoch, logs=logs)

        # Save model state
        save_training_state(generator, discriminator, gen_optimizer, disc_optimizer, epoch, checkpoint_dir)

    callback_list.on_train_end(logs=logs)
