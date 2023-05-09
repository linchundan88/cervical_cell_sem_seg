'''
    the loss function only interested in the mask area.
'''

def loss_with_masks(loss_function, outputs, targets, masks):
    loss = loss_function(outputs, targets)
    loss = (loss * masks.float()).sum()  # the element of masks is 0 or 1

    loss = loss / masks.sum()

    return loss

