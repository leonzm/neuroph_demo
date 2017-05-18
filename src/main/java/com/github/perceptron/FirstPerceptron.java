package com.github.perceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.ThresholdNeuron;
import org.neuroph.nnet.learning.BinaryDeltaRule;
import org.neuroph.util.*;

/**
 * @Author: Leon
 * @CreateDate: 2017/5/17
 * @Description: 简单感知器，参考 Perceptron
 * @Version: 1.0.0
 */
public class FirstPerceptron extends NeuralNetwork {

    public FirstPerceptron(int inputNeuronsCount, int outputNeuronsCount) {
        createNetwork(inputNeuronsCount, outputNeuronsCount, TransferFunctionType.STEP);
    }

    public FirstPerceptron(int inputNeuronsCount, int outputNeuronsCount, TransferFunctionType transferFunctionType) {
        createNetwork(inputNeuronsCount, outputNeuronsCount, transferFunctionType);
    }

    /**
     * 创建感知器
     * @param inputNeuronsCount 输入链接个数
     * @param outputNeuronsCount 输出链接个数
     * @param transferFunctionType 传输函数（激活函数）类型
     */
    private void createNetwork(int inputNeuronsCount, int outputNeuronsCount, TransferFunctionType transferFunctionType) {
        // 设置神经网络类型，这里将类型设置为感知器
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        // 初始化神经元输入刺激设置
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR); // 线性

        // 创建输入刺激
        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);

        // 初始化神经元输出刺激设置
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("neuronType", ThresholdNeuron.class); // 阀值神经元
        outputNeuronProperties.setProperty("thresh", new Double(Math.abs(Math.random())));
        outputNeuronProperties.setProperty("transferFunction", transferFunctionType);
        // 为sigmoid和tanh传输函数设置斜率属性
        outputNeuronProperties.setProperty("transferFunction.slope", new Double(1));

        // 创建一个神经元的输出
        Layer outputLayer = LayerFactory.createLayer(outputNeuronsCount, outputNeuronProperties);
        this.addLayer(outputLayer);

        // 在输入和输出层中建立全链接
        ConnectionFactory.fullConnect(inputLayer, outputLayer);

        // 为神经网络设置默认输入输出
        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new BinaryDeltaRule());
    }

}
