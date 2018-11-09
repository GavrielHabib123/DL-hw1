from LenetOverFashionMnist import LenetOverFashionMnist

basic_lenet = LenetOverFashionMnist(name='basic Lenet')
dropout_lenet = LenetOverFashionMnist(dropout=True, name='With Dropout')
weight_decay_lenet = LenetOverFashionMnist(weight_decay=True, name='With weight decay')
bn_lenet = LenetOverFashionMnist(bn=True, name='With Batch norm')

lenet_models = [basic_lenet,
                dropout_lenet,
                weight_decay_lenet,
                bn_lenet]

for model in lenet_models:
    model.load_model_and_weights()
