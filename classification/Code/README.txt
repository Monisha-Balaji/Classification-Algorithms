README

Install python 3.7.4 version, pandas, numpy, sklearn, math, statistics, random & itertools.

Run the submitted python file in Command prompt. Keep the files project3_dataset1.txt, project3_dataset2.txt, project3_dataset3_train.txt, project3_dataset3_test.txt, project3_dataset4.txt in the same folder as the python files in order to avoid giving full path when prompted for filename.

For the Kaggle Competition part, keep the files train_features.csv, train_label.csv & test_features.csv in the same folder as the python file as the file name is hardcoded in the program.

K-Nearest Neighbors:

python3 KNN.py

- User prompt for specifying the type of input dataset; (Enter 0 for single file or Enter 1 to input train and test separately). Enter the appropriate choice.
- If choice is 0 then:
	- User prompt for input dataset. Enter the filename along with path.
	- User prompt for the number of folds. Enter the number of folds for cross-validation.Do not enter 0 for number of folds.
- If choice is 1, then:
	- User prompt for train and test dataset. Enter the filenames along with path. 
- User prompt for the number of neighbors. Enter the number of neighbors to consider.
- User prompt for (Enter 1) Normalization of data or (Enter 2) for original dataset.


Decision Tree:

python3 DecisionTree.py

- User prompt for input dataset. Enter the filename along with path.
- If entered filename is not project3_dataset4.txt, then:
	- User prompt for the number of folds.Enter the number of folds for cross-validation.Do not enter 0 for number of folds.


Random Forest:

python3 RandomForest.py

- User prompt for input dataset. Enter the filename along with path.
- User prompt for the number of folds. Enter the number of folds for cross-validation.Do not enter 0 for number of folds.
- User prompt for the the number of trees. Enter the number of trees to be generated per fold.
- User prompt for (Enter 1) selecting number of features to be randomly selected for each fold of random forest or (Enter 2) for the number of features selected for each fold to be square root of total number of features.


Naive Bayes:
	
python3 NaiveBayes.py

- User prompt for input dataset. Enter the filename along with path.
- User prompt for the number of folds (Enter 0 if entered filename is project3_dataset4.txt). Enter the number of folds for cross-validation.Do not enter 0 for number of folds unless project3_dataset4.txt is enterd.
- If project3_dataset4.txt is entered as filename, then:
	-User prompt for entering test case. Enter test case with values separated via commas. Eg.:sunny,cool,high,weak


Kaggle Competition:

python3 RF-Kag-Final.py


