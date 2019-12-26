import svm_sentiment
import bayes_sentiment
import decsiontree_sentiment
import knn_sentiment
import mlp_sentiment
import lstm_sentiment
import cnn_sentiment


print("<---Accuracy Summary--->")
lstm_sentiment.lstm_test()
svm_sentiment.svm_test()
cnn_sentiment.cnn_test()
bayes_sentiment.bayes_test()
decsiontree_sentiment.decisiontree_test()
knn_sentiment.knn_test()
mlp_sentiment.mlp_test()


