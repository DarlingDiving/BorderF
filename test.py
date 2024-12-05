import os
from config import opt
from test_wang import setup_seed  #from test_facehq import setup_seed
from test_wang import CombineTrainer #from test_facehq import CombineTrainer
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if __name__ == "__main__":
    setup_seed(opt.seed)
    trainer = CombineTrainer(opt)
    trainer.test(realname="0_real",fakename="1_fake")
