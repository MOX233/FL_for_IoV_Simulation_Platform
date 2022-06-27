from utils.plot_utils import plot_loss_acc_curve
from utils.log_utils import save_training_log

class Evaluator():
    def __init__(self, args, evaluator_for_task) -> None:
        self.val_loss_list = []
        self.eval_metrices_list = []
        self.args = args
        self.evaluator = evaluator_for_task

    def eval_for_None(self, net, round):
        if len(self.val_loss_list) > 0:
            self.val_loss_list.append(self.val_loss_list[-1])
            self.eval_metrices_list.append(self.eval_metrices_list[-1])
        else:
            self.eval(net, round)

    def eval(self, net, round):
        net.eval()
        loss, eval_metrices = self.evaluator.eval_for_traj_pred(net, round)
        self.val_loss_list.append(loss)
        self.eval_metrices_list.append(eval_metrices)
        net.train()
        
    
    def record_eval_result(self, train_loss_list, round):
        plot_loss_acc_curve(self.args, train_loss_list, self.val_loss_list, self.eval_metrices_list, round)
        save_training_log(self.args, train_loss_list, self.val_loss_list, self.eval_metrices_list)
