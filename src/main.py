from argparse import ArgumentParser

from seaborn import set_theme
set_theme(font_scale=0.75)

from torch import nn, optim, manual_seed, backends, mps
# from torch.optim import lr_scheduler
# from torchinfo import summary
# from torchvision import models
backends.mps.allow_tf32 = True  # Enable TF32 for better performance
mps.empty_cache()  # Clear unused memory

from src.utils import Config

# seed for reproducibility
manual_seed(Config.SEED)

from src.trainer import get_trainer
from src.utils import DataPreprocessor, Scheduler

def main():
    parser = ArgumentParser(description='Sign Language alphabet and digit recognition')
    parser.add_argument('-t', '--model_type', choices=['cnn', 'tl', 'vit'], required=True, help='Model type: cnn, tl, vit')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=True, help='Mode: train, test')
    parser.add_argument('-s', '--save', choices=['best', 'last'], default=Config.SAVE_MODEL, help='Save best or last model')
    parser.add_argument('-e', '--epochs', type=int, default=Config.EPOCHS, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LR, help='Learning rate')
    parser.add_argument('-p', '--patience', type=int, default=Config.PATIENCE, help='Patience for early stopping')
    parser.add_argument('-f', '--print_freq', type=int, default=Config.PRINT_EVERY, help='Print frequency')
    args = parser.parse_args()

    model_type = args.model_type # 'cnn' or 'tl' or 'vit'
    mode = args.mode # 'train' or 'test'
    Config.SAVE_MODEL = args.save # 'best' or 'last'
    Config.EPOCHS = args.epochs
    Config.LR = args.learning_rate
    Config.PATIENCE = args.patience
    Config.PRINT_EVERY = args.print_freq

    dp = DataPreprocessor(Config.DATA_PATH,
                        transform=Config.transform,
                        resize_shape=Config.INPUT_SHAPE)
    train, val, test = dp.preprocess(test_ratio=Config.TEST_RATIO,
                                    val_ratio=Config.VAL_RATIO,
                                    batch_size=Config.BATCH_SIZE,
                                    seed=Config.SEED)
    classes = dp.classes

    # Train
    if mode == 'train':
        trainer = get_trainer(model_type, classes)
        # summary(trainer.model, input_size=(Config.BATCH_SIZE, Config.NB_CHANNELS, *Config.INPUT_SHAPE),
        #     col_names = ('input_size', 'output_size', 'num_params'), verbose = 0) # from torchinfo

        train_criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING[model_type])
        val_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(trainer.model.parameters(), betas=Config.BETAS, eps=Config.EPS) \
            if model_type == 'vit' else optim.Adam(trainer.model.parameters(), lr=Config.LR[model_type])
        scheduler = Scheduler(optimizer, Config.D_MODEL, Config.WARMUP_STEPS)\
            if model_type == 'vit' else optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=Config.STEP_MODE[Config.STEP_METRIC],
                                                                            factor=Config.STEP_FACTOR, patience=Config.STEP_AFTER, min_lr=1e-5)

        train_losses, val_losses = trainer.train(train,
                                                 val,
                                                 train_criterion,
                                                 val_criterion,
                                                 optimizer,
                                                 scheduler = scheduler,
                                                 epochs = Config.EPOCHS, 
                                                 patience = Config.PATIENCE,
                                                 print_every = Config.PRINT_EVERY,
                                                 step_per = Config.STEP_PER[model_type],
                                                 step_after = Config.STEP_AFTER,
                                                 step_metric= Config.STEP_METRIC,
                                                 save = Config.SAVE_MODEL)

        # Plot the training and validation losses
        trainer.plot_losses(train_losses, val_losses,
                            model_type=model_type,
                            save_dir=Config.OUTPUT_DIR,
                            show=False)
    # Evaluate
    else:
        trainer = get_trainer(model_type, classes)
        trainer.load_model(Config.MODEL_SAVE_PATH[model_type])
        acc, y_true, y_pred = trainer.evaluate(test, is_test=True)
        print(f"Test Accuracy: {acc:.2f}%")

        trainer.plot_confusion_matrix(y_true, y_pred, classes,
                                    figsize=Config.FIG_SIZE, normalize=True, fmt='.0f',
                                    model_type=model_type,accuracy=acc,
                                    save_dir=Config.OUTPUT_DIR, show=False)


if __name__ == '__main__':
    main()