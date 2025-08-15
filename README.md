## This is a testcase for OsmoViewCVTestCase ##

The note book one may look for is main notebook.
The env I used is in environment.yml.

Model: DAViT finetuned on given dataset

Estimated accuracy: 88-89% on test set

Further info about fitting, preprocessing and mining is stored in the notebook and other files

Max Accuracy obtained on test set: 89.9%.
Human benchmark on IMAGENET-1K: max Error rate: 12.0%, least error rate: 5%, optimistic error rate 2.4%. (source: https://arxiv.org/abs/1409.0575 ) 
This means that the model's performance is comparable, but worse then the human one.

Possible improvements for the learning algorithm:
* k-fold cross-validation (class stratified) instead of uusing the static validation set.
* Early stopping to prevent the overfitting and to (possibly) accelerate the fine-tuning
* Optimize the hyperparameters (such as lr, gamma etc) using specific packages (e.g. optuna, ray ml) or use grid search as the most straightforward opiton
* Make more augmentations to the training set, such as Affine Transforms, in order to increase the models generalize
