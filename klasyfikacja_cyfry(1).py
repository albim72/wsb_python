# -*- coding: utf-8 -*-

# -- Sheet --

import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1,ncols=4, figsize=(10,3))
for ax,image,label in zip(axes,digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Obraz treningowy: %i" %(label))

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

clf = svm.SVC(gamma=0.001)
X_train, X_test,y_train,y_test = train_test_split(data,digits.target,test_size=0.5, shuffle=False)

clf.fit(X_train,y_train)

predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1,ncols=4, figsize=(10,3))
for ax,image,prediction in zip(axes,X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("cyfra przewidywana: %i" %(prediction))

print(f"Raport z klasyfikacji dla klasyfikatora cyfr {clf}:\n"
    f"{metrics.classification_report(y_test,predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted)
disp.figure_.suptitle("Macierz błędu")
print(f"Macierz błędu:\n {disp.confusion_matrix}")
plt.show()

