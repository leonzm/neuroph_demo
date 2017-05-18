package com.github.perceptron;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

/**
 * @Author: Leon
 * @CreateDate: 2017/5/17
 * @Description: 使用多层感知器解决 XOR 问题
 * @Version: 1.0.0
 */
public class Test2 {

    public static void main(String[] args) {
        // XOR 问题，基本规则，两个值相等时返回0，否则返回1：
        // 0 xor 0 = 0
        // 0 xor 1 = 1
        // 1 xor 0 = 1
        // 1 xor 1 = 0
        DataSet trainingSet = new DataSet(2, 1); // 建立训练集，有两个输入一个输出
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));

        // 创建多层感知器，输入层2个神经元，隐含层3个神经元，最后输出层为1个隐含神经元，我们使用 TANH 传输函数用于最后格式化输出
        MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron(TransferFunctionType.TANH, 2, 3, 1);
        multiLayerPerceptron.getLearningRule().addListener(new LearningEventListener() {
            private int count = 0;
            @Override
            public void handleLearningEvent(LearningEvent learningEvent) {
                if (LearningEvent.Type.EPOCH_ENDED == learningEvent.getEventType()) {
                    System.out.println(String.format("第%d次训练完", ++ count));
                } else if (LearningEvent.Type.LEARNING_STOPPED == learningEvent.getEventType()) {
                    System.out.println("停止学习");
                }
            }
        });
        System.out.println("开始训练");
        multiLayerPerceptron.learn(trainingSet);

        System.out.println("训练结果，测试感知器是否正确输出：");
        Test1.checkNeuralNetwork(multiLayerPerceptron, trainingSet);
    }

}
