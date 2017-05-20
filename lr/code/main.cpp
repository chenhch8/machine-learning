#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <vector>
#include <ctime>
#include <cmath>

#define SIZE 385
#define STEP 0.05 // 步长

using namespace std;

// 参数初始化
void initTheta(double (*theta)[SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        theta[0][i] = 1;
}

void clearData(vector<double*> &samples) {
    for (int i = 0; i < samples.size(); ++i)
        delete []samples[i];
}

double toDouble(string &str) {
    istringstream iss(str);
    double num;
    iss >> num;
    return num;
}

void paraseStr(string &str, vector<double*> &samples) {
    if (str == "") return;
    double *feature = new double[SIZE + 1];
    feature[0] = 1;
    string substr;
    str += ',';
    for (int i = str.find(',') + 1, j = 1; i < str.size(); ++i) {
        if (str[i] == ',') {
            feature[j++] = toDouble(substr);
            // cout << "[1]: " << substr << ", [2]: " << feature[j - 1] << endl;
            substr = "";
        } else {
            substr += str[i];
        }
    }
    samples.push_back(feature);
}

void calcFactor(vector<double*> &samples, vector<double> &factors, double *theta, int &count) {
    double factor;
    int i, j;
    factors.clear();
    for (i = 0; i < count; ++i) {
        factor = 0;
        for (j = 0; j < SIZE; ++j) {
            factor += samples[i][j] * theta[j];
        }
        factors.push_back(factor - samples[i][SIZE]);
    }
}

void calcError(vector<double*> &samples, double *theta, int &count) {
    double errs = 0, value;
    for (int i = count, j; i < samples.size(); ++i) {
        value = theta[0];
        for (j = 1; j < SIZE; ++j)
            value += theta[j] * samples[i][j];
        // cout << "预测值为：" << value << ", 真实值为：" << samples[i][SIZE] << endl;
        errs += sqrt(pow(value - samples[i][SIZE], 2));
    }
    cout << "误差为 " << errs / (samples.size() - count) << endl;
    return;
}

void linearRegression(vector<double*> &samples, double *theta, int &count, int times) {
    int i, j;
    double temp, alltime;  // 梯度下降因子
    vector<double> factors;  // 梯度下降时的因子

    clock_t start = clock(), finish;

    for (int k = 0; k < times; ++k) {
        // 计算因子
        calcFactor(samples, factors, theta, count);
        for (i = 0; i < SIZE; ++i) {
            temp = 0;
            for (j = 0; j < count; ++j) {
                temp += samples[j][i] * factors[j];
            }
            theta[i] = theta[i] - STEP * temp / count;
        }
        cout << "进行第 " << k + 1 << " 次迭代" << endl;
        calcError(samples, theta, count);
    }
    finish = clock();
    alltime = (finish - start) / CLOCKS_PER_SEC;
    cout << "总用时: " << alltime / 60 << " min; 平均用时: " << alltime / times << " s"<< endl;
}

void saveModel(double *theta) {
    ofstream file("../data/model.csv", ios::binary);
    file.write((char *)theta, sizeof(double) * SIZE);
    file.close();
}

int main() {
    string str;
    int index = 0;
    int count = 22000; // 使用count个作为训练样本，剩余的作为测试样本
    double theta[SIZE];

    vector<double*> samples;  // 所有训练样本

    ifstream file("../data/save_train.csv");

    if (file.is_open()) {
        getline(file, str);
        // cout << str << endl;
        while(!file.eof()) {
            // 读入并保存样本
            getline(file, str);
            if (str == "") break;
            cout << "读入第 " << ++index << " 个样本" << endl;
            // cout << str << endl;
            paraseStr(str, samples);
        }
        file.close();

        // 线性回归计算
        linearRegression(samples, theta, count, 10000);
        // 保存模型
        saveModel(theta);
        // 测试误差
        calcError(samples, theta, count);
        // 清除数据
        clearData(samples);
    } else {
        cout << "文件读取失败" << endl;
    }
    return 0;
}
