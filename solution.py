import pickle
import matplotlib.pyplot as plt

class Solution(object):
    def __init__(self, metric, count_of_blocks, start_lr, sheduler, hid_size_in_block, frozen):
        self.data = {
            "metric" : metric,
            "count_of_blocks" : count_of_blocks,
            "start_lr" : start_lr,
            "sheduler" : sheduler,
            "hid_size_in_block" : hid_size_in_block,
            "frozen" : frozen
        }
    
    @classmethod
    def from_file(cls, dir, file_name):
        instance = cls(0, 0, 0, 0, 0, 0)
        with open(dir + "/" + file_name, 'rb') as f:
            instance.data = pickle.load(f)
        return instance

    def __repr__(self):
        print("start accuracy =", self.data["metric"]["start:accuracy ~ top#1"][-1])
        print("end accuracy =", self.data["metric"]["end:accuracy ~ top#1"][-1])
        print("accuracy =", self.data["metric"]["accuracy"][-1])
        print("count of blocks =", self.data["count_of_blocks"])
        print("hid_size =", self.data["hid_size_in_block"])
        if self.data["frozen"]:
            print("only blocks are trained")
        else:
            print("trained encoder and blocks")

        print("start lr =", self.data["start_lr"])
        print("sheduler =", self.data["sheduler"])
        return ""
    
    def save_to_file(self, dir, file):
        with open(dir + "/" + file, 'wb') as f:
            pickle.dump(self.data, f)

def draw_metrics(sol, ymin=0.6):
    metrics = sol.data["metric"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    print(sol)
    plt.ylim([ymin, 1.])
    plt.title(f'metrics')
    for name in metrics:
        plt.plot(metrics[name], '.-', label=name)
    plt.legend()
    plt.grid()
            
    plt.show()