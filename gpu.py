import tensorflow as tf

# List all physical GPUs detected
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

# Show the logical devices (what TensorFlow actually uses)
logical_gpus = tf.config.list_logical_devices('GPU')
print("Logical GPUs:", logical_gpus)

# Check which device operations are running on
print("TensorFlow is using:")
print(tf.test.gpu_device_name())   # usually shows something like '/device:GPU:0'
