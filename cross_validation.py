

def CV(model, X, Y, nr_folds = 10):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=nr_folds)
    accuracies = []
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train = [X[x] for x in train_index]
        X_test = [X[x] for x in test_index]
        Y_train = [Y[x] for x in train_index]
        Y_test = [Y[x] for x in test_index]

        # train the model:
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        # calculate the accuracy:
        miss = 0
        for i, y in enumerate(pred):
            if not y == Y_test[i]:
                miss += 1
        accuracies.append((len(pred)-miss)/len(pred))

    print(accuracies)
    print("avg", sum(accuracies)/len(accuracies))
    return accuracies, sum(accuracies)/len(accuracies)


if __name__ == '__main__':
    N = 1000
    X = [[i, i+5] for i in range(N)]
    Y = [0 if i[0]+i[1] < N/2 else 1 for i in X]
    import random
    l = [a for a in zip(X,Y)]
    random.shuffle(l)
    X,Y = list(zip(*l))
    from sklearn import svm
    model = svm.SVC()
    model.fit(X, Y)
    print(X)
    CV(model, X, Y, nr_folds=5)