
'''
@autor: DIEGO MARTIN CRESPO
@id: A20432558
To run the program python: NaiveBayes.py data/aclImdb 1.0 0.5
    Being alpha=1.0 and threshold=0.5
It is settle up to get the accuracy, precision and recal for the introduced values.
'''

import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata

#this method is used to print the progress at the prediction (it takes a while to process all dataset)
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print('\n')

class NaiveBayes:
    def __init__(self, data, ALPHA, THRESH):
        self.ALPHA = ALPHA
        self.probThresh = THRESH
        self.data = data # training data
        #TODO: Initalize parameters
        # self.vocab_len = 
        self.count_positive = []
        self.count_negative = []
        self.num_positive_reviews = 1
        self.num_negative_reviews = 1
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.positive_indices = []
        self.negative_indices = []
        # self.deno_pos = 
        # self.deno_neg =
        self.Train(data.X,data.Y)
        self.ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]
        self.THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        self.positive_indices = np.argwhere(Y == 1.0).flatten() # len 12500
        self.negative_indices = np.argwhere(Y == -1.0).flatten() #len 12500
        #print('positive index', len(positive_indices))
        #print('negative index', len(negative_indices))
        self.num_positive_reviews = len(self.positive_indices)
        self.num_negative_reviews = len(self.negative_indices)

        self.count_positive = np.ravel(X[self.positive_indices,:].sum(axis=0))
        self.count_negative = np.ravel(X[self.negative_indices,:].sum(axis=0))

        self.total_positive_words = self.count_positive.sum()
        self.total_negative_words = self.count_negative.sum()

        self.deno_pos = 1.0
        self.deno_neg = 1.0

        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        print('\nFor alfa:', self.ALPHA, 'and Threshold:\n', self.probThresh)
        #TODO: Implement Naive Bayes Classification
        self.P_positive = self.num_positive_reviews / \
            (self.num_positive_reviews + self.num_negative_reviews)
        self.P_negative = self.num_negative_reviews / \
            (self.num_positive_reviews + self.num_negative_reviews)
        
        pred_labels = []
        cnpos=0
        cnneg=0
        sh = X.shape[0]
        for i in range(sh): #length of matrix (number of documents)
            y_pos_score = log(self.P_positive)
            y_neg_score = log(self.P_negative)
            z = X[i].nonzero() #returns index of non zero colums of the matrix
            for j in range(len(z[0])):
                # Look at each feature
                # if not in the train data index prob=0
                if (len(self.count_positive)) < z[1][j]:
                    print('New word that is not in the training set ')
                    y_pos_score += X[i, z[1][j]] * \
                        log((self.ALPHA) /
                           (self.total_positive_words + self.ALPHA*2))
                    y_neg_score += X[i, z[1][j]] * \
                        log((self.ALPHA) /
                           (self.total_negative_words + self.ALPHA*2))
                else:
                    y_pos_score += X[i, z[1][j]] * log((self.count_positive[z[1][j]] + self.ALPHA) / (
                        self.total_positive_words + self.ALPHA*2))
                    y_neg_score += X[i, z[1][j]] * log((self.count_negative[z[1][j]] + self.ALPHA) / (
                        self.total_negative_words + self.ALPHA*2))
            
            predicted_prob_positive = exp(y_pos_score-self.LogSum(
                y_pos_score, y_neg_score))
            predicted_prob_negative = exp(y_neg_score-self.LogSum(
                y_pos_score, y_neg_score))

            printProgressBar(i, sh, prefix='Progress predicting Label:',
                             suffix='Complete', length=50)
           
            if predicted_prob_positive > self.probThresh:                     # comment for classification and evaluation with threshold (accuracy)
            #if predicted_prob_positive > predicted_prob_negative:            # Uncomment for classification and evaluation without threshold (precision and recall)
                pred_labels.append(1.0)
                cnpos+=1
            else:                                    # Predict negative
                pred_labels.append(-1.0)
                cnneg+=1
        return pred_labels

    def LogSum(self, logx, logy):   
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):
        # These two following senteces are repeated just in case this function is executed before the predictlabel() function
        self.P_positive = self.num_positive_reviews / \
            (self.num_positive_reviews + self.num_negative_reviews)
        self.P_negative = self.num_negative_reviews / \
            (self.num_positive_reviews + self.num_negative_reviews)
        sum_positive=0
        sum_negative=0
        pred_labels=[]
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            z = test.X[i].nonzero()
            sum_pred_pos = log(self.P_positive)
            sum_pred_neg = log(self.P_negative)
            for j in range(len(z[0])):
                pos = test.X[i, z[1][j]] * log(
                    (self.count_positive[z[1][j]] + self.ALPHA) / (self.total_positive_words + self.ALPHA*2))
                neg = test.X[i, z[1][j]] * log(
                    (self.count_negative[z[1][j]] + self.ALPHA) / (self.total_negative_words + self.ALPHA*2))
                sum_pred_pos += pos
                sum_pred_neg += neg
                
            predicted_prob_positive = exp(sum_pred_pos-self.LogSum(
                sum_pred_pos, sum_pred_neg))
            predicted_prob_negative = exp(sum_pred_neg-self.LogSum(
                sum_pred_pos, sum_pred_neg))

            if predicted_prob_positive > predicted_prob_negative:
                predicted_label = 1.0
                pred_labels.append(predicted_label)
                sum_positive += 1
            else:
                predicted_label = -1.0
                pred_labels.append(predicted_label)
                sum_negative += 1
            
            # TO DO: Comment the line above, and uncomment the line below
            # print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i],'\n')
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative,'\n')
        return pred_labels
    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        
        ev = Eval(Y_pred, test.Y)
        acuracy = ev.Accuracy()

        precision = self.EvalPrecision(test, Y_pred)
        recal = self.EvalRecal(test, Y_pred)
        return acuracy, precision, recal
    

    def EvalPrecision(self, test, Y_pred):
        tp=0.0
        fp=0.0
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1 and test.Y[i] == 1:
                tp+=1.0
            if Y_pred[i] == 1 and test.Y[i] == -1:
                fp+=1.0
        return tp/(tp+fp)

    def EvalRecal(self, test, Y_pred):
        tp = 0.0
        fn = 0.0
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1 and test.Y[i] == 1:
                tp += 1.0
            if Y_pred[i] == -1 and test.Y[i] == 1:
                fn += 1.0
        return tp/(tp+fn)

    def evalAll(self, test):
        accuracy=[]
        precision=[]
        recal=[]
        
        for i in self.ALPHAS:
            self.ALPHA=i
            Y_pred = self.PredictLabel(test.X)
            ev = Eval(Y_pred, test.Y)
            accuracy.append((self.ALPHA, ev.Accuracy()))
        
        #self.ALPHA = sorted(accuracy, key=lambda x: x[1], reverse=True)[0][1]
        self.ALPHA = 1.0
        for i in self.THRESHOLDS:
            self.probThresh=i
            Y_pred = self.PredictLabel(test.X)
            p = self.EvalPrecision(test, Y_pred)
            r = self.EvalRecal(test, Y_pred)
            precision.append((self.probThresh, p))
            recal.append((self.probThresh, r))
        
        return accuracy, precision, recal

# Print out the n most positive and n most negative words in the vocabulary  sorted by their weight according to your model
    def getFeatures(self, n, traindata):
        mostPos=[]
        mostNeg=[]
        posNorm = []
        negNorm = []
        for i in range(len(self.count_positive)):
            posNorm.append((i, self.count_positive[i]/self.total_positive_words))
            negNorm.append((i, self.count_negative[i]/self.total_negative_words))

        posNorm20 = sorted(posNorm, key=lambda x: x[1], reverse=True)[:n]
        negNorm20 = sorted(negNorm, key=lambda x: x[1], reverse=True)[:n]
        
        for i in range(n):
            mostPos.append((traindata.vocab.GetWord(posNorm20[i][0]),posNorm20[i][1]))
            mostNeg.append((traindata.vocab.GetWord(negNorm20[i][0]), negNorm20[i][1]))
        
        return mostPos, mostNeg

if __name__ == "__main__":

    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]), float(sys.argv[3]))
    print("Evaluating")
    print("\nPredict Probability:\n")
    nb.PredictProb(testdata, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # UNCOMMENT THIS IF WHAT TO EVALUATE FOR 5 Alpha values and 5 Thresholds
    '''
    Accuracy, Precision, Recall = nb.evalAll(testdata)
    print('These are the (Alfa, accuracy) pairs:', Accuracy)
    print('These are the (Thresholds, precision) pairs:', Precision)
    print('These are the (Thresholds, recall) pairs:', Recall)
    '''

    Accuracy, Precision, Recall = nb.Eval(testdata)
    print("\n\nTest Accuracy: ", Accuracy) #chage the threshold value at the PredictLbael function to se the actual accuracy
    print("\nTest Precision: ", Precision)
    print("\nTest Recall: ", Recall)
    
    mostPos, mostNeg = nb.getFeatures(20, traindata)
    print('\nMost Positive 20 words:\n', mostPos)
    print('Most Negative 20 words:\n', mostNeg)
