import torch

def dice_coef(y_true, y_pred):
    # y_true_f = y_true.view(-1)
    # y_pred_f = y_pred.view(-1)
    # intersection = torch.sum(y_true_f * y_pred_f)
    # smooth = 0.0001
    # return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    intersection = torch.sum(y_true * y_pred)
    smooth = 0.0001
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_coef_multilabel(y_true, y_pred, num_classes, num_points):
    dice = 0
    #for index in range(num_classes):
    for index in range(num_points):
        dice_score_point = dice_coef(y_true[index, :], y_pred[index, :])
        # print("dice_score_point")
        # print(dice_score_point)
        dice += dice_score_point
    #return dice/num_classes # taking average
    return dice/num_points # taking average

def add_i_and_u(i, u, i_total, u_total, batch_idx):

    # Sum i and u along the batch dimension (gives value per class)
    i = torch.sum(i, dim=0) / i.shape[0]
    u = torch.sum(u, dim=0) / u.shape[0]

    if batch_idx == 0:
        i_total = i
        u_total = u
    else:
        i_total += i
        u_total += u

    return i_total, u_total


def get_mean_iou_per_class(i_total, u_total):

    i_total = i_total.type(torch.FloatTensor)
    u_total = u_total.type(torch.FloatTensor)

    mean_iou_per_class = i_total / u_total

    return mean_iou_per_class





