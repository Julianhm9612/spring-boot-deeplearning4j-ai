package com.example.spring_boot_deeplearning4j_ai.service;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;

@Service
public class ImageClassificationService {

    private MultiLayerNetwork model;

    @PostConstruct
    public void init() throws IOException {
        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 123;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(1e-3))
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(channels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(height, width, channels))
            .build();

        model = new MultiLayerNetwork(conf);
        model.init();

        // Train the model (in a real-world scenario, you'd use a larger dataset and more epochs)
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);
        }
    }

    public int classifyImage(INDArray image) {
        INDArray output = model.output(image);
        return output.argMax(1).getInt(0);
    }
}
