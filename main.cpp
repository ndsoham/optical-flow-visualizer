#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

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

    for (size_t i = 0; i < grayFrame1.size(); i++) {
        It.push_back(grayFrame2[i] - grayFrame1[i]);
    }

    return It;
}

std::vector<float> gaussianKernel(int windowSize, float sigma) {
    std::vector<float> kernel;
    kernel.reserve(windowSize*windowSize);
    int half = windowSize/2;

    float sum = 0;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            float weight = std::exp(-(dx*dx + dy*dy)/(2*sigma*sigma));
            kernel.push_back(weight);
            sum += weight;
        }
    }

    for (float& w : kernel) {
        w /= sum;
    }

    return kernel;
}

void lucasKanade(const std::vector<float>& Ix, const std::vector<float>&Iy,
                 const std::vector<float>& It, int rows, int cols, int windowSize,
                 std::vector<float>& u, std::vector<float>& v, float sigma) {

    auto idx = [&cols](int r, int c){return r*cols + c;};
    int half = windowSize/2;

    std::vector<float> gauss = gaussianKernel(windowSize, sigma);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {

            float IxIx = 0, IxIy = 0, IyIy = 0, IxIt = 0, IyIt = 0;
            
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    int cX = std::min(cols-1, std::max(0, col+dx));
                    int cY = std::min(rows-1, std::max(0, row+dy));
                    int i = idx(cY, cX);
                    int idx = (dy + half) * windowSize + (dx + half);
                    float w = gauss.at(idx);

                    IxIx += w * Ix.at(i) * Ix.at(i);
                    IxIy += w * Ix.at(i) * Iy.at(i);
                    IyIy += w * Iy.at(i) * Iy.at(i);
                    IxIt += w * Ix.at(i) * It.at(i);
                    IyIt += w * Iy.at(i) * It.at(i);
                }
            }

            float det = IxIx*IyIy - IxIy*IxIy;
            float u_i = 0, v_i = 0;

            if (std::abs(det) > 1e-6) {
                u_i = (1/det) * (-IyIy*IxIt + IxIy*IyIt);
                v_i = (1/det) * (IxIy*IxIt  - IxIx*IyIt);
            }

            float mag = std::sqrt(u_i*u_i + v_i*v_i);
            if (mag < 0.3f) {
                u_i = 0;
                v_i = 0;
            }

            u.push_back(u_i);
            v.push_back(v_i);

        }
    }
}

cv::Mat flowToHSV(const std::vector<float>& u, const std::vector<float>&v,
                  int rows, int cols){
    cv::Mat hsv(rows, cols, CV_8UC3);

    float maxMag = 0;
    for (size_t i = 0; i < u.size(); i++) {
        float mag = std::sqrt(u[i]*u[i] + v[i]*v[i]);
        maxMag = std::max(maxMag, mag);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            float angle = std::atan2(v[i], u[i]);
            float deg = angle * 180.0f / M_PI + 180.0f;
            float mag = std::sqrt(u[i]*u[i] + v[i]*v[i]);
            float normalizedMag = maxMag > 0 ? std::sqrt(mag / maxMag) : 0;

            uint8_t* pixel = hsv.data + i*3;
            pixel[0] = (uint8_t)(deg / 2.0f);
            pixel[1] = 255;
            pixel[2] = (uint8_t)(normalizedMag * 255);
        }
    }

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}



int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "usage: <frames directory>" << std::endl;
        return 1;
    }

    std::vector<std::string> frameFiles;

    for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
        frameFiles.push_back(entry.path());
    }

    std::sort(frameFiles.begin(), frameFiles.end());

    cv::Mat frame1 = cv::imread(frameFiles.at(0));
    std::vector<float> grayFrame1 = toGrayscale(frame1);
    int nRows = frame1.rows;
    int nCols = frame1.cols;
    
    std::string outputPath = strcat(argv[1], "/output.mp4");
    cv::VideoWriter writer(outputPath, 
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           30.0, 
                           cv::Size(nCols, nRows));

    for (auto it = frameFiles.begin()+1; it != frameFiles.end(); it++) {
        std::cout << "Processing " << *it << std::endl;
        cv::Mat frame2 = cv::imread(*it);

        if (frame1.empty() || frame2.empty()) {
            std::cerr << "Failed to load images" << std::endl;
            return 1;
        }
        
        std::vector<float> grayFrame2 = toGrayscale(frame2);

        std::vector<float> Ix;
        std::vector<float> Iy;

        Ix.reserve(grayFrame1.size());
        Iy.reserve(grayFrame1.size());
        computeSpatialGradients(grayFrame1, nRows, nCols, Ix, Iy);
        std::vector<float> It = computeTemporalGradient(grayFrame1, grayFrame2);

        std::vector<float> u;
        std::vector<float> v;
        u.reserve(Ix.size());
        v.reserve(Iy.size());

        int windowSize = 15;
        float sigma = (float)windowSize/6.0;
        lucasKanade(Ix, Iy, It, nRows, nCols, windowSize, u, v, sigma);
        cv::Mat flowBgr = flowToHSV(u, v, nRows, nCols);

        writer.write(flowBgr);

        grayFrame1 = grayFrame2;
    }
    
    return 0;
}