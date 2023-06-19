from model import compile_model
from dataset_process import train_dataset, val_dataset, test_dataset

model = compile_model()
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1,
)
model.evaluate(test_dataset, verbose=1)
model.save("./neural/model.h5")
