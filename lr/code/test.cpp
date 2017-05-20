#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <sstream>

using namespace std;

#define SIZE 385

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
    double *feature = new double[SIZE];
    feature[0] = 1;
    string substr;
    str += ',';
    for (int i = str.find(',') + 1, j = 1; i < str.size(); ++i) {
        if (str[i] == ',') {
            feature[j++] = toDouble(substr);
            substr = "";
        } else {
            substr += str[i];
        }
    }
    samples.push_back(feature);
}

int main() {
    ifstream file("../data/model.csv", ios::binary);
    double theta[SIZE], expect;
    string str;
    vector<double*> samples;  // 所有训练样本
    if (file.is_open()) {
        file.read((char*)theta, sizeof(double)*SIZE);
        file.close();
        file.open("../data/save_test.csv");
        getline(file, str);
        while (!file.eof()) {
            getline(file, str);
            paraseStr(str, samples);
        }
        file.close();

        cout << "SIZE: " << samples.size() << endl;
        
        ofstream out("../data/result.csv");
        if (out.is_open()) {
            out << "Id,reference" << endl;
            for (int i = 0; i < samples.size(); ++i) {
                expect = 0;
                for (int j = 0; j < SIZE; ++j) {
                    expect += theta[j] * samples[i][j];
                }
                out << i << "," << expect << endl;
                cout << "进行第 " << i + 1 << " 个预测, 预测结果为：" << expect << endl;
            }
            out.close();
        } else {
            cout << "文件错误" << endl;
        }

        clearData(samples);
    } else {
        cout << "文件错误" << endl;
    }
    
    return 0;
}
