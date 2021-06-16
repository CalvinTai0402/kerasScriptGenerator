data = [
    "keras.optimizers.SGD(0.01)",
    "keras.optimizers.RMSprop(0.01)",
    "keras.optimizers.Adam(0.01)",
    "keras.optimizers.Adadelta(0.01)",
    "keras.optimizers.Adagrad(0.01)",
    "keras.optimizers.Adamax(0.01)",
    "keras.optimizers.Nadam(0.01)",
    "keras.optimizers.ftrl(0.01)",
]

result = []
for datum in data:
    record = {"name": datum, "value": datum}
    result.append(record)
print(result)
