package com.github.perceptron;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.Perceptron;

import java.util.Arrays;

/**
 * @Author: Leon
 * @CreateDate: 2017/5/17
 * @Description: 训练感知器学会 AND 运算和 OR 元算 <br/>
 * 单层感知器无法解决 XOR 问题
 * @Version: 1.0.0
 */
public class Test1 {

    public static void main(String[] args) {
        // 1.逻辑元算 AND 中的基本规则：
        // 0 and 0 = 0
        // 0 and 1 = 0
        // 1 and 0 = 0
        // 1 and 1 = 1

        DataSet trainingSet1 = new DataSet(2, 1); // 建立训练集，有两个输入一个输出
        trainingSet1.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet1.addRow(new DataSetRow(new double[]{0, 1}, new double[]{0}));
        trainingSet1.addRow(new DataSetRow(new double[]{1, 0}, new double[]{0}));
        trainingSet1.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));

        // 2.逻辑运算 OR 中基本规则
        // 0 or 0 = 0
        // 0 or 1 = 1
        // 1 or 0 = 1
        // 1 or 1 = 1
        DataSet trainingSet2 = new DataSet(2, 1); // 建立训练集，有两个输入一个输出
        trainingSet2.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet2.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet2.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet2.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));

        // 3.XOR 问题，基本规则，两个值相等时返回0，否则返回1：
        // 0 xor 0 = 0
        // 0 xor 1 = 1
        // 1 xor 0 = 1
        // 1 xor 1 = 0
        DataSet trainingSet3 = new DataSet(2, 1); // 建立训练集，有两个输入一个输出
        trainingSet3.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet3.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet3.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet3.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));

        // 建立一个单层感知器，定义输入刺激是2个，感知器输出是1个，这里用 Neuroph 提供的 Perceptron 类
        NeuralNetwork myPerceptron = new Perceptron(2, 1);
        LearningRule learningRule = myPerceptron.getLearningRule(); // 学习规则
        learningRule.addListener(new LearningEventListener() {
            private int count = 0;
            @Override
            public void handleLearningEvent(LearningEvent learningEvent) {
                if (LearningEvent.Type.EPOCH_ENDED == learningEvent.getEventType()) {
                    System.out.println(String.format("第%d次训练完", ++ count)); // 因为初始权值是随机的，所以学习迭代次数每次都不一样
                } else if (LearningEvent.Type.LEARNING_STOPPED == learningEvent.getEventType()) {
                    System.out.println("停止学习");
                }
            }
        });

        DataSet dataSet = trainingSet1; // trainingSet1, trainingSet2, trainingSet3
        // 开始学习训练集
        myPerceptron.learn(dataSet); // 使用上述的训练数据进行训练
        System.out.println("训练结果，测试感知器是否正确输出：");
        checkNeuralNetwork(myPerceptron, dataSet);
    }

    // 提供感知器和输入，输出结果
    public static void checkNeuralNetwork(NeuralNetwork neuralNetwork, DataSet dataSet) {
        for(DataSetRow dataRow : dataSet.getRows()) {
            neuralNetwork.setInput(dataRow.getInput());
            neuralNetwork.calculate();
            double[] networkOutput = neuralNetwork.getOutput();
            System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
            System.out.println(" Output: " + Arrays.toString(networkOutput) );

        }
    }

}
