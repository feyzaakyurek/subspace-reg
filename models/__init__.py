from .resnet_language import resnet12, resnet18, resnet24

model_pool = [
    'resnet12',
    'resnet18',
    'resnet24'
]

model_dict = {
    'resnet12': resnet12,
    'resnet18': resnet18,
    'resnet24': resnet24
}
