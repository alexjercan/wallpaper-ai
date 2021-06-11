
# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/XinJCheng/CSPN/blob/b3e487bdcdcd8a63333656e69b3268698e543181/cspn_pytorch/utils.py#L19
# - https://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html
#

import torch


class MetricFunction():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.total_size = 0
        self.error_sum = {}
        self.error_avg = {}

    def evaluate(self, predictions, labels):
        _, predictions = torch.max(predictions, 1)

        error_val = evaluate_error_classification(predictions, labels)

        self.total_size += self.batch_size
        self.error_avg = avg_error(self.error_sum, error_val, self.total_size, self.batch_size)
        return self.error_avg

    def show(self):
        error = self.error_avg
        format_str = ('======CLASSIFICATION========\nACC=%.4f\n')
        return format_str % (error['C_ACC'])


def evaluate_error_classification(predictions, targets):
    error = {}

    correct = (predictions == targets).sum()
    length = targets.shape[0]

    error['C_ACC'] = correct / length
    return error


# avg the error
def avg_error(error_sum, error_val, total_size, batch_size):
    error_avg = {}
    for item, value in error_val.items():
        error_sum[item] = error_sum.get(item, 0) + value * batch_size
        error_avg[item] = error_sum[item] / float(total_size)
    return error_avg


def print_single_error(epoch, loss, error):
    format_str = ('%s\nEpoch: %d, loss=%s\n%s\n')
    print (format_str % ('eval_avg_error', epoch, loss, error))