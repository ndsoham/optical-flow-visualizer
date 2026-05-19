#include <iostream>
#include <opencv2/opencv.hpp>

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

    uint8_t* data = frame1.data;

    std::cout << "First Pixel (BGR): "
              << (int)data[0] << " "
              << (int)data[1] << " "
              << (int)data[2] << std::endl;

    cv::imshow("Frame 1", frame1);
    cv::waitKey(0);

    return 0;
}