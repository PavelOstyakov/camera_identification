from sklearn.metrics import accuracy_score, log_loss

NUM_EPOCH = 100


def predict_test(model, test_loader):
    return model.predict_generator(test_loader)


def train(model, train_loader, val_loader, val_labels, model_save_path):
    epoch_id = 0
    best_loss = 1000
    while True:
        model.fit_generator(train_loader)
        y_pred = model.predict_generator(val_loader)

        loss = log_loss(val_labels, y_pred, eps=1e-6)
        accuracy = accuracy_score(val_labels, y_pred.argmax(axis=-1))

        print("Epoch {0}. Val accuracy {1}. Val loss {2}".format(epoch_id, accuracy, loss))
        model.scheduler_step(loss, epoch_id)
        if loss < best_loss:
            best_loss = loss
            model.save(model_save_path)

        epoch_id += 1

        if epoch_id == NUM_EPOCH:
            break

