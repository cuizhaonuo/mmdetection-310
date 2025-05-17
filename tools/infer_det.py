from mmdet.apis import DetInferencer

# models is a list of model names, and them will print automatically
models = DetInferencer.list_models('mmdet')
print(models)