import yaml

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

print config
# print config['finetune_lr'][2]
activation = 0 if config['activation'] == 'relu' or \
    config['activation'] == 'Relu' or \
    config['activation'] == 'ReLu' \
    else 1
print activation

#
