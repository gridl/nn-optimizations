from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseTrainer:
    def __init__(self, model, train_iterator, val_iterator, optimizer,
                 criterion,
                 use_cuda, gpu_idx=0, lr_scheduler=None, disable_tqdm=False):
        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_cuda = use_cuda
        self.gpu_idx = gpu_idx
        self.lr_scheduler = lr_scheduler
        self.disable_tqdm = disable_tqdm
        self.is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    def to_gpu(self, tensor):
        """Helper method if we want to run our model on the CPU"""
        if self.use_cuda:
            return tensor.cuda(self.gpu_idx)
        else:
            return tensor

    def from_gpu(self, tensor):
        if self.use_cuda:
            return tensor.cpu()
        else:
            return tensor

    def train_or_val_epoch(self, data_iterator, train):
        """
        train: bool, train or evaluation mode
        """
        mode = 'Train' if train else 'Validation'
        data_iterator = tqdm(
            enumerate(data_iterator),
            total=len(data_iterator),
            disable=self.disable_tqdm,
            desc=mode)
        if train:
            if self.lr_scheduler and not self.is_reduce_on_plateau:
                self.lr_scheduler.step()
        running_loss = 0
        running_accuracy = 0
        for i, (inputs, targets) in data_iterator:
            inputs = Variable(self.to_gpu(inputs))
            targets = Variable(self.to_gpu(targets))
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            if train:
                self.model_based_optimization(loss)
            running_loss += loss.data
            running_accuracy += self.compute_accuracy(logits, targets)
        running_loss /= i
        running_accuracy /= i
        return running_loss, running_accuracy

    def model_based_optimization(self, loss):
        """This method should call optimization step for the required model.
        Can be updated for every model"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_accuracy(self, logits, targets):
        return 1

    def __call__(self, epochs):
        results = {
            'train': {
                'loss': [],
                'accuracy': [],
            },
            'val': {
                'loss': [],
                'accuracy': [],
            }
        }
        for epoch in epochs:
            for mode in ['train', 'val']:
                train_bool = mode == 'train'
                loss, acc = self.train_or_val_epoch(
                    self.train_iterator, train=train_bool)
                results[mode]['loss'].append(loss)
                results[mode]['acc'].append(acc)
        return results


if __name__ == '__main__':
    # TODO: I think this instances should be created from configs
    criterion = None
    optimizer = None
    lr_scheduler = None
    trainer = BaseTrainer()
    results_dict = trainer(epochs=22)
