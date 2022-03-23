from vidaug import augmentors as va

def va_augment(aug_list):
    augment_list = []
    if "VerticalFlip" in aug_list:
        augment_list.append(va.VerticalFlip())
    if "HorizontalFlip" in aug_list:
        augment_list.append(va.HorizontalFlip())
    if "GaussianBlur" in aug_list:
        augment_list.append(va.GaussianBlur(2))
    if "Add" in aug_list:
        augment_list.append(va.Add(10))
    if "Multiply" in aug_list:
        augment_list.append(va.Multiply(1.05))
    if "ElasticTransform" in aug_list:
        augment_list.append(va.ElasticTransformation(alpha = 0.5, sigma = 1,cval = 50))
    if "Salt" in aug_list:
        augment_list.append(va.Salt(95))
    if "Pepper" in aug_list:
        augment_list.append(va.Pepper(95))
    if "RandomShear" in aug_list:
        augment_list.append(va.RandomShear(10,10))

    va_augmentation = va.Sequential(augment_list)
    return va_augmentation