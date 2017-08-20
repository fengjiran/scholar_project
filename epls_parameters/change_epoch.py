import cPickle

with open('current_finetune_epoch.save', 'wb') as f:
    cPickle.dump(200, f, True)
