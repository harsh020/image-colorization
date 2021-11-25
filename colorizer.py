from absl import app
from absl import flags

from colorizer import colorizer

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", "train",
                    "Mode in which to run the model. Must be one of train, inference.")
flags.DEFINE_string("state_dict_path", None,
                     "Path of serialized state dict of trained model.")
flags.DEFINE_string("image_path", None,
                     "Path of image to color, when running in inference mode.")

# Required flag.
# flags.mark_flag_as_required("mode")

def main(argv):
    del argv  # Unused.

    mode = FLAGS.mode
    state_dict_path = FLAGS.state_dict_path
    image_path = FLAGS.image_path

    if mode == 'train':
       colorizer.train(state_dict_path)
    elif mode == 'inference':
        if image_path == None:
            raise ValueError('Please provide image path to run inference.')
        colorizer.inference(image_path, state_dict_path)
    else:
        raise ValueError(f'Invalid value {mode} for mode.')


if __name__ == '__main__':
    app.run(main)
