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

cv::Mat gradientToMat(const std::vector<float>& grad, int rows, int cols) {
    float minVal = *std::min_element(grad.begin(), grad.end());
    float maxVal = *std::max_element(grad.begin(), grad.end());
    float range = maxVal - minVal;

    cv::Mat result(rows, cols, CV_8UC1);
    for (int i = 0; i < rows*cols; i++) {
        result.data[i] = (uint8_t)(255.0f * (grad[i] - minVal)/range);
    }

    return result;
}

void computeSpatialGradients(const std::vector<float>& gray, int rows, int cols,
                      std::vector<float>& Ix, std::vector<float>& Iy) {
    int kX[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int kY[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    auto kIdx = [](int r, int c){return r*3 + c;};
    auto grayIdx = [&cols](int r, int c){return r*cols + c;};
    
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            // apply the filter
            float sX = 0;
            float sY = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int imgX = std::min(cols-1, std::max(0, col+dx));
                    int imgY = std::min(rows-1, std::max(0, row+dy));
                    int kx = dx+1;
                    int ky = dy+1;

                    sX += gray.at(grayIdx(imgY, imgX)) * kX[kIdx(ky, kx)];
                    sY += gray.at(grayIdx(imgY, imgX)) * kY[kIdx(ky, kx)];

                }  
            }

            Ix.push_back(sX);
            Iy.push_back(sY);
        }
    }
}

std::vector<float> computeTemporalGradient(const std::vector<float>& grayFrame1, 
                                           const std::vector<float>& grayFrame2) {
    std::vector<float> It;
    It.reserve(grayFrame1.size());

    for (int i = 0; i < grayFrame1.size(); i++) {
        It.push_back(grayFrame2[i] - grayFrame1[i]);
    }

    return It;
}



int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cout << "usage: <filename> <frame1> <frame2>" << std::endl;
        return 1;
    }

    cv::Mat frame1 = cv::imread(argv[1]);
    cv::Mat frame2 = cv::imread(argv[2]);

    if (frame1.empty() || frame2.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return 1;
    }

    std::cout << "Width: " << frame1.cols << std::endl;
    std::cout << "Height: " << frame1.rows << std::endl;

    std::cout << "Channels: " << frame1.channels() << std::endl;

    std::cout << "Type: " << frame1.type() << std::endl;

    std::vector<float> grayFrame1 = toGrayscale(frame1);
    std::vector<float> grayFrame2 = toGrayscale(frame2);

    std::vector<float> Ix;
    std::vector<float> Iy;

    Ix.reserve(grayFrame1.size());
    Iy.reserve(grayFrame1.size());
    computeSpatialGradients(grayFrame1, frame1.rows, frame1.cols, Ix, Iy);
    
    std::vector<float> It = computeTemporalGradient(grayFrame1, grayFrame2);

    return 0;
}