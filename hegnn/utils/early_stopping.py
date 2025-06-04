class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01, path="checkpoint.pt", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.path = path
        self.best_model_state = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            # self.save_checkpoint(model)

        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"EarlyStopping: No improvement for {self.counter} epochs.")
                self.early_stop = True
        else:
            # self.save_checkpoint(model)
            self.counter = 0


if __name__ == "__main__":
    # Example usage
    early_stopping = EarlyStopping(patience=5)

    for _ in range(10):
        early_stopping(1)
