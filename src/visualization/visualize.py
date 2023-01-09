import argparse
import sys
import torch

sys.path.append('src/models/')
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Viz(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for Visualizing",
            usage="python visualize.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
                
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
            
    def visualize(self):
        print("Creating TSNE visualization")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        weights = model.classifier.state_dict()['3.weight'].numpy()
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random', perplexity=3).fit_transform(weights)

        plt.plot(X_embedded[:,0]);
        plt.plot(X_embedded[:,1]);
        plt.xlabel('Class')
        plt.ylabel('Value')
        plt.savefig("C:/Users/duweh/OneDrive/6_MLops/MLops/cookiecutter_test/reports/figures/TSNE_2.png")

if __name__ == '__main__':
    Viz()