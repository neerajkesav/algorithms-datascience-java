/**
 * NaiveBayesClassifier
 */
package com.neeraj.algorithms;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Class NaiveBayesClassifier. Implementation of Naive Bayes Classifier.
 * 
 * @author neeraj
 *
 */
public class NaiveBayesClassifier {

	/**
	 * Class NaiveBayesClassifier implements many methods for loading data,
	 * splitting data, summarizing data, calculating probability, testing model
	 * and for making prediction. This implementation is for numeric data set.
	 * Last attribute should be the class values.
	 */

	/**
	 * @classVariable train: Training data set.
	 * @classVariable test: Testing data set.
	 * @classVariable predictions: Predicted class values.
	 */
	private ArrayList<Float[]> train = new ArrayList<Float[]>();
	private ArrayList<Float[]> test = new ArrayList<Float[]>();
	private ArrayList<Float> predictions = new ArrayList<Float>();

	/**
	 * Function to test the model by calculating accuracy of
	 * prediction on a given data set.
	 * 
	 * @param fileName
	 *            File path to the data set file.
	 * @param splitRatio
	 *            Ratio at which data set is to be split.
	 * @throws IOException
	 */
	private void testModel(String fileName, double splitRatio) throws IOException {
		/* Loading data set */
		ArrayList<Float[]> data = loadFile(fileName);
		/* Splitting data set. */
		splitData(data, splitRatio);
		/* Prediction */
		performPredictions();
		/* Printing Prediction. */
		showPredictions();
		/* Accuracy */
		calculateAccuracy();
	}

	/**
	 * Function loads the data set from file.
	 * 
	 * @param filename
	 *            Data set file.
	 * @return dataSet
	 * @throws IOException
	 */
	private ArrayList<Float[]> loadFile(String filename) throws IOException {

		ArrayList<String[]> dataLines = new ArrayList<String[]>();
		ArrayList<Float[]> dataSet = new ArrayList<Float[]>();
		/* Reading from file */
		@SuppressWarnings("resource")
		Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset());
		/* Splitting data to attributes. */
		lines.forEach(line -> dataLines.add(line.split(",")));

		for (String[] line : dataLines) {
			Float[] lineData = new Float[line.length];
			int i = 0;
			for (String value : line) {
				lineData[i++] = Float.valueOf(value);
			}
			dataSet.add(lineData);
		}
		return dataSet;
	}

	/**
	 * Function splits data into train set and test set.
	 * 
	 * @param data
	 *            data set.
	 * @param ratio
	 *            split ratio.
	 */
	@SuppressWarnings("unchecked")
	private void splitData(ArrayList<Float[]> data, double ratio) {

		/* Train set size using split ratio. */
		int train_size = (int) (data.size() * ratio);

		/* Splitting data set to train set and test set */
		for (int i = 0; i < train_size; i++) {
			int random_index = new Random().nextInt(data.size());
			this.train.add(data.get(random_index));
			data.remove(data.get(random_index));
		}
		this.test = (ArrayList<Float[]>) data.clone();
	}

	/**
	 * Function maps each record in data set to its class value.
	 * 
	 * @return classData
	 */
	private HashMap<Float, ArrayList<Float[]>> dataByClass() {

		HashMap<Float, ArrayList<Float[]>> classData = new HashMap<Float, ArrayList<Float[]>>();
		/* Index of last/class attribute */
		int last = train.get(0).length - 1;

		/* Maps each record to its class value. */
		for (int i = 0; i < train.size(); i++) {
			Float[] value = train.get(i);
			float key = value[last];
			if (classData.containsKey(key)) {
				ArrayList<Float[]> temp = classData.get(key);
				temp.add(value);
				classData.put(key, temp);
			} else {
				ArrayList<Float[]> temp = new ArrayList<Float[]>();
				temp.add(value);
				classData.put(key, temp);
			}
		}

		return classData;
	}

	/**
	 * Function to prepare the class summary.
	 * 
	 * @return summary
	 */
	private HashMap<Float, ArrayList<Float[]>> summaryByClass() {

		HashMap<Float, ArrayList<Float[]>> summary = new HashMap<Float, ArrayList<Float[]>>();
		HashMap<Float, ArrayList<Float[]>> classData = dataByClass();
		Set<Float> keys = classData.keySet();

		/* Preparing class summary */
		for (float key : keys) {
			ArrayList<Float[]> values = new ArrayList<Float[]>();
			/* Summarize data for a particular class */
			values = summarize(classData.get(key));
			summary.put(key, values);
		}

		return summary;
	}

	/**
	 * Function prepares the summary of each class by using mean
	 * and standard deviation of each attribute/column.
	 * 
	 * @param data
	 *            Data records of a particular class.
	 * @return classSummary
	 */
	private ArrayList<Float[]> summarize(ArrayList<Float[]> data) {

		/* Mean for each attribute/column */
		Float[] mean = calculateMean(data);
		/* Standard deviation for each attribute/column */
		Float[] stdDev = calculateStdDev(data, mean);
		ArrayList<Float[]> classSummary = new ArrayList<Float[]>();
		classSummary.add(mean);
		classSummary.add(stdDev);

		return classSummary;
	}

	/**
	 * Function calculates standard deviation of each
	 * columns/attributes.
	 * 
	 * @param data
	 *            Data records of a particular class.
	 * @param mean
	 *            mean of each attributes/columns.
	 * @return stdDev
	 */
	private Float[] calculateStdDev(ArrayList<Float[]> data, Float[] mean) {

		int length = data.get(0).length - 1;
		int dataSize = data.size();
		Float[] stdDev = new Float[length];
		float sum = 0;
		/* Calculates standard deviation of each column/attribute. */
		for (int i = 0; i < length; i++) {
			for (Float[] value : data) {
				sum += Math.pow((value[i] - mean[i]), 2);
			}
			stdDev[i] = (float) Math.sqrt(sum / (float) dataSize);
			sum = 0;
		}

		return stdDev;
	}

	/**
	 * Function calculates mean of each columns/attributes.
	 * 
	 * @param data
	 *            Data records of a particular class.
	 * @return mean
	 */
	private Float[] calculateMean(ArrayList<Float[]> data) {

		int length = data.get(0).length - 1;
		int dataSize = data.size();
		Float[] mean = new Float[length];
		float sum = 0;
		/* Calculates mean of each column/attribute. */
		for (int i = 0; i < length; i++) {
			for (Float[] value : data) {
				sum += value[i];
			}
			mean[i] = (float) (sum / dataSize);
			sum = 0;
		}

		return mean;
	}

	/**
	 * Function calculate probability of each value.
	 * 
	 * @param value
	 *            Attribute value of a record.
	 * @param mean
	 *            Mean of a particular column/attribute.
	 * @param stdDev
	 *            Standard deviation of a particular column/attribute.
	 * @return
	 */
	private double calculateProbability(float value, float mean, float stdDev) {

		double exponent = Math
				.exp(-1 * (Math.pow(((double) value - (double) mean), 2) / (2 * Math.pow((double) stdDev, 2))));
		double prob = (1 / (Math.sqrt(2 * Math.PI) * (double) stdDev)) * exponent;

		return prob;
	}

	/**
	 * Function calculates probability of each class.
	 * 
	 * @param testRecord
	 *            One record from test set.
	 * @return probabilities
	 */
	private HashMap<Float, Float> classProbabilities(Float[] testRecord) {

		HashMap<Float, Float> probabilities = new HashMap<Float, Float>();
		/* Getting class summary. */
		HashMap<Float, ArrayList<Float[]>> summary = summaryByClass();
		Set<Float> classKeys = summary.keySet();
		int length = testRecord.length - 1;
		/* Calculating class probabilities. */
		for (float classKey : classKeys) {
			float classProbability = 1;
			ArrayList<Float[]> values = summary.get(classKey);
			for (int i = 0; i < length; i++) {
				/* Mean of ith column/attribute. */
				float mean = values.get(0)[i];
				/* Standard deviation of ith column/attribute. */
				float stdDev = values.get(1)[i];
				/* ith column/attribute value of a specific test set record. */
				float attribute = testRecord[i];
				/*
				 * Calculating class probability by multiplying probability of
				 * each values.
				 */
				classProbability *= calculateProbability(attribute, mean, stdDev);
			}
			probabilities.put(classKey, classProbability);
		}

		return probabilities;
	}

	/**
	 * Function performs prediction on each record in test
	 * set.
	 */
	private void performPredictions() {

		predictions.clear();
		for (Float[] record : test) {
			float prediction = predict(record);
			predictions.add(prediction);
		}
	}

	/**
	 * Function performs class prediction of a record.
	 * 
	 * @param record
	 *            Single record in test set.
	 * @return predictedClass
	 */
	private float predict(Float[] record) {

		/* Getting class probabilities for prediction. */
		HashMap<Float, Float> probabilities = classProbabilities(record);
		float predictedClass = -1;
		float probability = -1;
		for (float classKey : probabilities.keySet()) {
			float prob = probabilities.get(classKey);
			if ((predictedClass == -1) || (probability < prob)) {
				predictedClass = classKey;
				probability = prob;
			}
		}

		return predictedClass;
	}

	/**
	 * Function calculates accuracy of prediction.
	 */
	private void calculateAccuracy() {

		int truePredction = 0;
		int size = test.size();
		int last = test.get(0).length - 1;
		for (int i = 0; i < size; i++) {
			/* Checking for correct prediction. */
			if (test.get(i)[last] == (float) predictions.get(i)) {
				truePredction++;
			}
		}
		float accuracy = (truePredction / (float) size) * 100;
		System.out.println("\nAccuracy = " + accuracy + "%");
	}

	/**
	 * Function prints prediction.
	 */
	private void showPredictions() {

		int i = 0;
		System.out.println("Class\tPrediction");
		for (Float[] values : test) {

			System.out.println(values[values.length - 1] + "\t" + predictions.get(i));
			i++;
		}

	}

	/**
	 * Function makes prediction on a given training data and
	 * input data.
	 * 
	 * @param trainFile
	 *            Training data set.
	 * @param inputFile
	 *            Data in which prediction is to be made.
	 * @throws IOException
	 */
	private void makePredictions(String trainFile, String inputFile) throws IOException {

		/* Loading training data set */
		this.train = loadFile(trainFile);
		/* Loading input data for prediction. */
		this.test = loadFile(inputFile);
		/* Prediction */
		performPredictions();
		System.out.println("\n-------------Prediction-----------------");
		showPredictions();
	}

	/**
	 * Function describes how to use 'NaiveBayesClassifier'.
	 * 
	 * @param args
	 *            Takes nothing.
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		/* Testing model. */
		NaiveBayesClassifier naiveBayes1 = new NaiveBayesClassifier();
		naiveBayes1.testModel("resource/pima-indians-diabetes.data.csv", 0.70);

		/* Making prediction. */
		NaiveBayesClassifier naiveBayes2 = new NaiveBayesClassifier();
		String trainFile = "resource/pima-indians-diabetes.data.csv";
		/*
		 * In case of the input file for prediction class attribute should not
		 * be empty, set the class value to some numeric value. example: -1.
		 */
		String inputFile = "resource/input.csv";
		naiveBayes2.makePredictions(trainFile, inputFile);

	}

}
