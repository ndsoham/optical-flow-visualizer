#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<float> toGrayscale(const cv::Mat& img) {
    
    auto intensity = [](int b, int g, int r){return 0.114*b + 0.587*g + 0.299*r;};

    std::vector<float> grayScaled;
    grayScaled.reserve(img.rows * img.cols);
    uint8_t* pixel;

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            int stride = img.cols*3;
            pixel = img.data + row * stride + col * 3;
            int blue = (int)*(pixel);
            int green = (int)*(pixel+1);
            int red = (int)*(pixel+2);
            grayScaled.push_back(intensity(blue, green, red));
        }
    }

    return grayScaled;
}

cv::Mat toMat(const std::vector<float>& grayScaled, int rows, int cols) {
    cv::Mat result(rows, cols, CV_8UC1);
    for (int i = 0; i < rows*cols; i++) {
        result.data[i] = (uint8_t)std::min(255.0f, std::max(0.0f, grayScaled[i]));
    }
    return result;
}



int main() {
    cv::Mat frame1 = cv::imread("../eval-data/Basketball/frame10.png");
    cv::Mat frame2 = cv::imread("../eval-data/Basketball/frame11.png");

    if (frame1.empty() || frame2.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return 1;
    }

    std::cout << "Width: " << frame1.cols << std::endl;
    std::cout << "Height: " << frame1.rows << std::endl;

    std::cout << "Channels: " << frame1.channels() << std::endl;

    std::cout << "Type: " << frame1.type() << std::endl;

    std::vector<float> grayScaled = toGrayscale(frame1);
    cv::Mat grayFrame1 = toMat(grayScaled, frame1.rows, frame1.cols);

    cv::imshow("Grayscaled", grayFrame1);
    cv::waitKey(0);

    return 0;
}