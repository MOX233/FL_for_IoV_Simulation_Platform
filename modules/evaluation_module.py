from utils.log_utils import save_training_log

class Evaluator():
    def __init__(self, args, evaluator_for_task, plot_func) -> None:
        self.val_loss_list = []
        self.eval_metrices_list = []
        self.args = args
        self.evaluator_for_task = evaluator_for_task
        self.plot_func = plot_func

    def eval_for_None(self, net, round):
        if len(self.val_loss_list) > 0:
            self.val_loss_list.append(self.val_loss_list[-1])
            self.eval_metrices_list.append(self.eval_metrices_list[-1])
        else:
            self.eval(net, round)

    def eval(self, net, round):
        net.eval()
        loss, eval_metrices = self.evaluator_for_task.eval_a_round(net, round)
        self.val_loss_list.append(loss)
        self.eval_metrices_list.append(eval_metrices)
        net.train()
        
    
    def record_eval_result(self, train_loss_list, round):
        self.plot_func(self.args, train_loss_list, self.val_loss_list, self.eval_metrices_list, round)
        save_training_log(self.args, train_loss_list, self.val_loss_list, self.eval_metrices_list)
