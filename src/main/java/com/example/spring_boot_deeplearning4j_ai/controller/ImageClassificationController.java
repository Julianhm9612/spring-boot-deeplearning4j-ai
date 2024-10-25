package com.example.spring_boot_deeplearning4j_ai.controller;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.spring_boot_deeplearning4j_ai.service.ImageClassificationService;

@RestController
@RequestMapping("/api/image-classification")
public class ImageClassificationController {

    @Autowired
    private ImageClassificationService classificationService;

    int height = 28;
    int width = 28;
    int channels = 1;

    @PostMapping("/classify")
    public ResponseEntity<Integer> classifyImage(@RequestBody byte[] imageBytes) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
        INDArray image = loader.asMatrix(img);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        // INDArray image = convertBytesToINDArray(imageBytes);
        int classification = classificationService.classifyImage(image);
        return ResponseEntity.ok(classification);
    }

    // private INDArray convertBytesToINDArray(byte[] imageBytes) throws IOException {
    //     BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
    //     double[][][] data = new double[1][28][28];
    //     for (int i = 0; i < 28; i++) {
    //         for (int j = 0; j < 28; j++) {
    //             data[0][i][j] = (double) (img.getRGB(j, i) & 0xFF) / 255.0;
    //         }
    //     }
    //     return Nd4j.create(data);
    // }
}
