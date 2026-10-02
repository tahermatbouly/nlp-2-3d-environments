import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.train import main
import pipeline.config as cfg
cfg.EPOCHS = 1
cfg.SAVE_EVERY = 1
main()
