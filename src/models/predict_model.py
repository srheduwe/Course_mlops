import argparse
import sys
import torch

sys.path.append('src/data/')
from make_dataset import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
            
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        parser.add_argument('load_data_from', default="")
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model = model.to(self.device)

        test_set = torch.load(args.load_data_from)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
        
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            
            preds = model(x.to(self.device))
            preds = preds.argmax(dim=-1)
            
            correct += (preds == y.to(self.device)).sum().item()
            total += y.numel()
            
        print(f"Test set accuracy {correct/total}")


if __name__ == '__main__':
    TrainOREvaluate()
