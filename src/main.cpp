#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#include "run_darknet.h"

#define POSE_MAX_PEOPLE 96
#define NET_OUT_CHANNELS 57 // 38 for pafs, 19 for parts

template<typename T>
inline int intRound(const T a)
{
    return int(a+0.5f);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

void render_pose_keypoints
    (
    Mat& frame,
    const vector<float>& keypoints,
    vector<int> keyshape,
    const float threshold,
    float scale
    )
{
    const auto num_keypoints = keyshape[1];
    vector<unsigned int> pairs{ 1, 2, 1, 5, 2, 3, 3,
    4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12,
    12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17 };
    vector<float> colors{ 255.f, 0.f, 85.f,
    255.f, 0.f, 0.f, 255.f, 85.f, 0.f, 255.f, 170.f, 0.f,
    255.f, 255.f, 0.f, 170.f, 255.f, 0.f, 85.f, 255.f, 0.f,
    0.f, 255.f, 0.f, 0.f, 255.f, 85.f, 0.f, 255.f, 170.f,
    0.f, 255.f, 255.f, 0.f, 170.f, 255.f, 0.f, 85.f, 255.f,
    0.f, 0.f, 255.f, 255.f, 0.f, 170.f, 170.f, 0.f, 255.f,
    255.f, 0.f, 255.f, 85.f, 0.f, 255.f };
    const auto number_colors = colors.size();

    for (auto person = 0; person < keyshape[0]; ++person)
        {
        // Draw lines
        for (auto pair = 0u; pair < pairs.size(); pair += 2)
            {
            const auto index1 = (person * num_keypoints + pairs[pair]) * keyshape[2];
            const auto index2 = (person * num_keypoints + pairs[pair + 1]) * keyshape[2];
            if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
                {
                const auto color_index = pairs[pair + 1] * 3;
                Scalar color { colors[(color_index + 2) % number_colors],
                               colors[(color_index + 1) % number_colors],
                               colors[(color_index + 0) % number_colors]};
                Point keypoint1{ intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale) };
                Point keypoint2{ intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale) };
                line(frame, keypoint1, keypoint2, color, 2);
                }
            }
        // Draw circles
        for (auto part = 0; part < num_keypoints; ++part)
            {
            const auto index = (person * num_keypoints + part) * keyshape[2];
            if (keypoints[index + 2] > threshold)
                {
                const auto color_index = part * 3;
                Scalar color { colors[(color_index + 2) % number_colors],
                               colors[(color_index + 1) % number_colors],
                               colors[(color_index + 0) % number_colors]};
                Point center{ intRound(keypoints[index] * scale), intRound(keypoints[index + 1] * scale) };
                circle(frame, center, 3, color, -1);
                }
            }
        }
}

void connect_bodyparts
    (
    vector<float>& pose_keypoints,
    const float* const map,
    const float* const peaks,
    int mapw,
    int maph,
    const int inter_min_above_th,
    const float inter_th,
    const int min_subset_cnt,
    const float min_subset_score,
    vector<int>& keypoint_shape
    )
{
    keypoint_shape.resize(3);
    const int POSE_COCO_PAIRS[] =
    { 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11,
      11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17 };
    const int POSE_COCO_MAP_IDX[] =
    { 31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25,
      26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46 };
    const int *body_part_pairs = POSE_COCO_PAIRS;
    const int *limb_idx = POSE_COCO_MAP_IDX;
    const int num_body_parts = 18; // COCO part number
    const int num_body_part_pairs = num_body_parts + 1;

    std::vector<std::pair<std::vector<int>, double>> subset;
    const int subset_counter_index = num_body_parts;
    const int subset_size = num_body_parts + 1;
    const int peaks_offset = 3 * (POSE_MAX_PEOPLE + 1);
    const int map_offset = mapw * maph;

    for (auto pair_index = 0u; pair_index < num_body_part_pairs; ++pair_index)
        {
        const auto body_partA = body_part_pairs[2 * pair_index];
        const auto body_partB = body_part_pairs[2 * pair_index + 1];
        const auto* candidateA = peaks + body_partA*peaks_offset;
        const auto* candidateB = peaks + body_partB*peaks_offset;
        const auto nA = (int)(candidateA[0]); // number of part A candidates
        const auto nB = (int)(candidateB[0]); // number of part B candidates

        // add parts into the subset in special case
        if (nA == 0 || nB == 0)
            {
            // Change w.r.t. other
            if (nA == 0) // nB == 0 or not
                {
                for (auto i = 1; i <= nB; ++i)
                    {
                    bool num = false;
                    const auto indexB = body_partB;
                    for (auto j = 0u; j < subset.size(); ++j)
                        {
                        const auto off = (int)body_partB*peaks_offset + i * 3 + 2;
                        if (subset[j].first[indexB] == off)
                            {
                            num = true;
                            break;
                            }
                        }
                    if (!num)
                        {
                        std::vector<int> row_vector(subset_size, 0);
                        // store the index
                        row_vector[body_partB] = body_partB*peaks_offset + i * 3 + 2;
                        // the parts number of that person
                        row_vector[subset_counter_index] = 1;
                        // total score
                        const auto subsetScore = candidateB[i * 3 + 2];
                        subset.emplace_back(std::make_pair(row_vector, subsetScore));
                        }
                    }
                }
            else // if (nA != 0 && nB == 0)
                {
                for (auto i = 1; i <= nA; i++)
                    {
                    bool num = false;
                    const auto indexA = body_partA;
                    for (auto j = 0u; j < subset.size(); ++j)
                        {
                        const auto off = (int)body_partA*peaks_offset + i * 3 + 2;
                        if (subset[j].first[indexA] == off)
                            {
                            num = true;
                            break;
                            }
                        }
                    if (!num)
                        {
                        std::vector<int> row_vector(subset_size, 0);
                        // store the index
                        row_vector[body_partA] = body_partA*peaks_offset + i * 3 + 2;
                        // parts number of that person
                        row_vector[subset_counter_index] = 1;
                        // total score
                        const auto subsetScore = candidateA[i * 3 + 2];
                        subset.emplace_back(std::make_pair(row_vector, subsetScore));
                        }
                    }
                }
            }
        else // if (nA != 0 && nB != 0)
            {
            std::vector<std::tuple<double, int, int>> temp;
            const auto num_inter = 10;
            // limb PAF x-direction heatmap
            const auto* const mapX = map + limb_idx[2 * pair_index] * map_offset;
            // limb PAF y-direction heatmap
            const auto* const mapY = map + limb_idx[2 * pair_index + 1] * map_offset;
            // start greedy algorithm
            for (auto i = 1; i <= nA; i++)
                {
                for (auto j = 1; j <= nB; j++)
                    {
                    const auto dX = candidateB[j * 3] - candidateA[i * 3];
                    const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
                    const auto norm_vec = float(std::sqrt(dX*dX + dY*dY));
                    // If the peaksPtr are coincident. Don't connect them.
                    if (norm_vec > 1e-6)
                        {
                        const auto sX = candidateA[i * 3];
                        const auto sY = candidateA[i * 3 + 1];
                        const auto vecX = dX / norm_vec;
                        const auto vecY = dY / norm_vec;
                        auto sum = 0.;
                        auto count = 0;
                        for (auto lm = 0; lm < num_inter; lm++)
                            {
                            const auto mX = fastMin(mapw - 1, intRound(sX + lm*dX / num_inter));
                            const auto mY = fastMin(maph - 1, intRound(sY + lm*dY / num_inter));
                            const auto idx = mY * mapw + mX;
                            const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
                            if (score > inter_th)
                                {
                                sum += score;
                                ++count;
                                }
                            }

                        // parts score + connection score
                        if (count > inter_min_above_th)
                            {
                            temp.emplace_back(std::make_tuple(sum / count, i, j));
                            }
                        }
                    }
                }
            // select the top minAB connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (!temp.empty())
                {
                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());
                }
            std::vector<std::tuple<int, int, double>> connectionK;

            const auto minAB = fastMin(nA, nB);
            // assuming that each part occur only once, filter out same part1 to different part2
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);
            auto counter = 0;
            for (auto row = 0u; row < temp.size(); row++)
                {
                const auto score = std::get<0>(temp[row]);
                const auto aidx = std::get<1>(temp[row]);
                const auto bidx = std::get<2>(temp[row]);
                if (!occurA[aidx - 1] && !occurB[bidx - 1])
                    {
                    // save two part score "position" and limb mean PAF score
                    connectionK.emplace_back(std::make_tuple(body_partA*peaks_offset + aidx * 3 + 2,
                        body_partB*peaks_offset + bidx * 3 + 2, score));
                    ++counter;
                    if (counter == minAB)
                        {
                        break;
                        }
                    occurA[aidx - 1] = 1;
                    occurB[bidx - 1] = 1;
                    }
                }
            // Cluster all the body part candidates into subset based on the part connection
            // initialize first body part connection
            if (pair_index == 0)
                {
                for (const auto connectionKI : connectionK)
                    {
                    std::vector<int> row_vector(num_body_parts + 3, 0);
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    const auto score = std::get<2>(connectionKI);
                    row_vector[body_part_pairs[0]] = indexA;
                    row_vector[body_part_pairs[1]] = indexB;
                    row_vector[subset_counter_index] = 2;
                    // add the score of parts and the connection
                    const auto subset_score = peaks[indexA] + peaks[indexB] + score;
                    subset.emplace_back(std::make_pair(row_vector, subset_score));
                    }
                }
            // Add ears connections (in case person is looking to opposite direction to camera)
            else if (pair_index == 17 || pair_index == 18)
                {
                for (const auto& connectionKI : connectionK)
                    {
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    for (auto& subsetJ : subset)
                        {
                        auto& subsetJ_first = subsetJ.first[body_partA];
                        auto& subsetJ_first_plus1 = subsetJ.first[body_partB];
                        if (subsetJ_first == indexA && subsetJ_first_plus1 == 0)
                            {
                            subsetJ_first_plus1 = indexB;
                            }
                        else if (subsetJ_first_plus1 == indexB && subsetJ_first == 0)
                            {
                            subsetJ_first = indexA;
                            }
                        }
                    }
                }
            else
                {
                if (!connectionK.empty())
                    {
                    for (auto i = 0u; i < connectionK.size(); ++i)
                        {
                        const auto indexA = std::get<0>(connectionK[i]);
                        const auto indexB = std::get<1>(connectionK[i]);
                        const auto score = std::get<2>(connectionK[i]);
                        auto num = 0;
                        // if A is already in the subset, add B
                        for (auto j = 0u; j < subset.size(); j++)
                            {
                            if (subset[j].first[body_partA] == indexA)
                                {
                                subset[j].first[body_partB] = indexB;
                                ++num;
                                subset[j].first[subset_counter_index] = subset[j].first[subset_counter_index] + 1;
                                subset[j].second = subset[j].second + peaks[indexB] + score;
                                }
                            }
                        // if A is not found in the subset, create new one and add both
                        if (num == 0)
                            {
                            std::vector<int> row_vector(subset_size, 0);
                            row_vector[body_partA] = indexA;
                            row_vector[body_partB] = indexB;
                            row_vector[subset_counter_index] = 2;
                            const auto subsetScore = peaks[indexA] + peaks[indexB] + score;
                            subset.emplace_back(std::make_pair(row_vector, subsetScore));
                            }
                        }
                    }
                }
            }
        }

    // Delete people below thresholds, and save to output
    auto number_people = 0;
    std::vector<int> valid_subset_indexes;
    valid_subset_indexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
    for (auto index = 0u; index < subset.size(); ++index)
        {
        const auto subset_counter = subset[index].first[subset_counter_index];
        const auto subset_score = subset[index].second;
        if (subset_counter >= min_subset_cnt && (subset_score / subset_counter) > min_subset_score)
            {
            ++number_people;
            valid_subset_indexes.emplace_back(index);
            if (number_people == POSE_MAX_PEOPLE)
                {
                break;
                }
            }
        }

    // Fill and return poseKeypoints
    keypoint_shape = { number_people, (int)num_body_parts, 3 };
    if (number_people > 0)
        {
        pose_keypoints.resize(number_people * (int)num_body_parts * 3);
        }
    else
        {
        pose_keypoints.clear();
        }
    for (auto person = 0u; person < valid_subset_indexes.size(); ++person)
        {
        const auto& subsetI = subset[valid_subset_indexes[person]].first;
        for (auto bodyPart = 0u; bodyPart < num_body_parts; bodyPart++)
            {
            const auto baseOffset = (person*num_body_parts + bodyPart) * 3;
            const auto bodyPartIndex = subsetI[bodyPart];
            if (bodyPartIndex > 0)
                {
                pose_keypoints[baseOffset] = peaks[bodyPartIndex - 2];
                pose_keypoints[baseOffset + 1] = peaks[bodyPartIndex - 1];
                pose_keypoints[baseOffset + 2] = peaks[bodyPartIndex];
                }
            else
                {
                pose_keypoints[baseOffset] = 0.f;
                pose_keypoints[baseOffset + 1] = 0.f;
                pose_keypoints[baseOffset + 2] = 0.f;
                }
            }
        }
}

void find_heatmap_peaks
    (
    float *map,
    float *_peaks,
    int mapw,
    int maph,
    int mapc,
    float threshold
    )
{
    float *peaks = _peaks;
    int map_offset = mapw * maph;
    int peaks_offset = 3 * (POSE_MAX_PEOPLE + 1);
    for (int c = 0; c < mapc; ++c)
        {
        int num_peaks = 0;
        for (int y = 1; y < maph - 1 && num_peaks != POSE_MAX_PEOPLE; ++y)
            {
            for (int x = 1; x < mapw - 1 && num_peaks != POSE_MAX_PEOPLE; ++x)
                {
                float value = map[y*mapw + x];
                if (value > threshold)
                    {
                    const float topLeft = map[(y - 1)*mapw + x - 1];
                    const float top = map[(y - 1)*mapw + x];
                    const float topRight = map[(y - 1)*mapw + x + 1];
                    const float left = map[y*mapw + x - 1];
                    const float right = map[y*mapw + x + 1];
                    const float bottomLeft = map[(y + 1)*mapw + x - 1];
                    const float bottom = map[(y + 1)*mapw + x];
                    const float bottomRight = map[(y + 1)*mapw + x + 1];
                    if (value > topLeft && value > top &&
                        value > topRight && value > left &&
                        value > right && value > bottomLeft &&
                        value > bottom && value > bottomRight)
                        {
                        float xAcc = 0;
                        float yAcc = 0;
                        float scoreAcc = 0;
                        for (int kx = -3; kx <= 3; ++kx)
                            {
                            int ux = x + kx;
                            if (ux >= 0 && ux < mapw)
                                {
                                for (int ky = -3; ky <= 3; ++ky)
                                    {
                                    int uy = y + ky;
                                    if (uy >= 0 && uy < maph)
                                        {
                                        float score = map[uy * mapw + ux];
                                        xAcc += ux * score;
                                        yAcc += uy * score;
                                        scoreAcc += score;
                                        }
                                    }
                                }
                            }
                        xAcc /= scoreAcc;
                        yAcc /= scoreAcc;
                        scoreAcc = value;
                        peaks[(num_peaks + 1) * 3 + 0] = xAcc;
                        peaks[(num_peaks + 1) * 3 + 1] = yAcc;
                        peaks[(num_peaks + 1) * 3 + 2] = scoreAcc;
                        ++num_peaks;
                        }
                    }
                }
            }
        peaks[0] = num_peaks;
        map += map_offset;
        peaks += peaks_offset;
        }
}

Mat create_netsize_im
    (
    const Mat &im,
    const int netw,
    const int neth,
    float *scale
    )
{
    // for tall image
    int newh = neth;
    float s = newh / (float)im.rows;
    int neww = im.cols * s;
    if (neww > netw)
        {
        //for fat image
        neww = netw;
        s = neww / (float)im.cols;
        newh = im.rows * s;
        }

    *scale = 1 / s;
    Rect dst_area(0, 0, neww, newh);
    Mat dst = Mat::zeros(neth, netw, CV_8UC3);
    resize(im, dst(dst_area), Size(neww, newh));
    return dst;
}

int main
    (
    int ac,
    char **av
    )
{
    if (ac != 6)
        {
        cout << "usage: ./bin [image file] [cfg file] [weight file] [net input width] [net input height]" << endl;
        return 1;
        }

    // 0. read args
    char *im_path = av[1];
    char *cfg_path = av[2];
    char *weight_path = av[3];
    int net_inw = atoi(av[4]);
    int net_inh = atoi(av[5]);
    Mat im = imread(im_path);
    if (im.empty())
        {
        cout << "failed to read image" << endl;
        return 1;
        }

    // 1. resize to net input size, put scaled image on the top left
    float scale = 0.0f;
    Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

    // 2. normalized to float type
    netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

    // 3. split channels
    float *netin_data = new float[net_inw * net_inh * 3]();
    float *netin_data_ptr = netin_data;
    vector<Mat> input_channels;
    for (int i = 0; i < 3; ++i)
        {
        Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
        input_channels.emplace_back(channel);
        netin_data_ptr += (net_inw * net_inh);
        }
    split(netim, input_channels);

    // 4. feed forward
    int net_outw = 0;
    int net_outh = 0;
    float *netoutdata = run_net(cfg_path, weight_path, netin_data, &net_outw, &net_outh);


    // 5. resize net output back to input size to get heatmap
    float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
    for (int i = 0; i < NET_OUT_CHANNELS; ++i)
        {
        Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
        Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
        resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
        }

    // 6. get heatmap peaks
    float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * 56];
    find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

    // 7. link parts
    vector<float> keypoints;
    vector<int> shape;
    connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);

    // 8. draw result
    render_pose_keypoints(im, keypoints, shape, 0.05, scale);

    // 9. show and save result
    cout << "people: " << shape[0] << endl;
    imshow("demo", im);
    waitKey(0);

    delete [] heatmap_peaks;
    delete [] heatmap;
    delete [] netin_data;
    return 0;
}
