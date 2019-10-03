// classifier.cpp
// wumengxi@umich.edu

#include <glob.h>
#include <libgen.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <cassert>
#include <string.h>
#include <sstream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <stack>
#include <cstdlib>

using namespace std;

class PreProcess{
    
public:
    
    PreProcess(){}
    
    // Function to print the confusion matrix.
    // Argument 1: "actual" is a list of integer class labels, one for each test example.
    // Argument 2: "predicted" is a list of integer class labels, one for each test example.
    // "actual" is the list of actual (ground truth) labels.
    // "predicted" is the list of labels predicted by your classifier.
    // "actual" and "predicted" MUST be in one-to-one correspondence.
    // That is, actual[i] and predicted[i] stand for testfile[i].
	
    void printConfMat(vector<int>actual, vector<int>predicted){
        vector<int> all_labels;
        assert(actual.size() == predicted.size());
        for (vector<int>::iterator i = actual.begin(); i != actual.end(); i++)
            all_labels.push_back((*i));
        for (vector<int>::iterator i = predicted.begin(); i != predicted.end(); i++)
            all_labels.push_back((*i));
        sort( all_labels.begin(), all_labels.end() );
        all_labels.erase( unique( all_labels.begin(), all_labels.end() ), all_labels.end() );
        map<pair<int,int>, unsigned> confmat;  // Confusion Matrix
        int itt = 0;
        for (vector<int>::iterator i = actual.begin(); i != actual.end(); i++){
            int a = (*i);
            pair<int, int> pp = make_pair(a, predicted[itt]);
            if (confmat.find(pp) == confmat.end()) confmat[pp] = 1;
            else confmat[pp] += 1;
            itt++;
        }
        cout << "\n\n";
        cout << "0 ";  // Actual labels column (aka first column)
        vector<int> tmp_labels;
        for (vector<int>::iterator i = all_labels.begin(); i != all_labels.end(); i++){
            int label2 = (*i);
            cout << label2 << " ";
            tmp_labels.push_back(label2);
        }
        cout << "\n";
        for (vector<int>::iterator i = all_labels.begin(); i != all_labels.end(); i++){
            int label = (*i);
            cout << label << " ";
            for (vector<int>::iterator i2 = tmp_labels.begin(); i2 != tmp_labels.end(); i2++){
                int label2 = (*i2);
                pair<int, int> pp = make_pair(label, label2);
                if (confmat.find(pp) == confmat.end()) cout << "0 ";
                else cout << confmat[pp] << " ";
            }
            cout << "\n";
        }
    }
    
    // Function to remove leading, trailing, and extra space from a string.
    // Inputs a string with extra spaces.
    // Outputs a string with no extra spaces.
    string remove_extra_space(string str){
        string buf; // Have a buffer string
        stringstream ss(str); // Insert the string into a stream
        vector<string> tokens; // Create vector to hold our words
        while (ss >> buf) tokens.push_back(buf);
        const char* const delim = " ";
        ostringstream imploded;
        copy(tokens.begin(), tokens.end(), ostream_iterator<string>(imploded, delim));
        return imploded.str();
    }
    
    // Tokenizer.
    // Input: string
    // Output: list of lowercased words from the string
    vector<string> word_tokenize(string input_string){
        string punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        string extra_space_removed = remove_extra_space(input_string);
        string punctuation_removed = "";
        for (unsigned i = 0; i < extra_space_removed.length(); i++) {
            char curr = extra_space_removed[i];
            if (punctuations.find(curr) == string::npos) punctuation_removed += curr;
        }
        transform(punctuation_removed.begin(), punctuation_removed.end(), punctuation_removed.begin(), ::tolower);
        string buf; // Have a buffer string
        stringstream ss(punctuation_removed); // Insert the string into a stream
        vector<string> tokens; // Create vector to hold our words
        while (ss >> buf) tokens.push_back(buf);
        return tokens;
    }
};


class BernoulliNaiveBayes{
    
private:
    
    string train_test_dir;
    vector<string> features; //stopwords
    PreProcess p;
    
public:
    
    BernoulliNaiveBayes(string train_test_dirr, vector<string> vocabb, PreProcess pp){
        train_test_dir = train_test_dirr;
        features = vocabb;
        p = pp;
    }
    
    void featureCurve(int numFea, string problemN, const vector<int>& actual,
                      unordered_set<int>& all_labels, ofstream &output) {
        map<string, int> frequency;
        vector<pair<string, int>> temp;
        vector<int> results;
        map<int, double> prior;
        map<string, map<int, double>> condprob;

		for (size_t i = 0; i < features.size(); ++i) {
			frequency.insert({features[i], 0});
		}

        for (int i = 1; i < 14; ++i) {
            for (int j = 1; j < 5; ++j) {
                ifstream file;
                string fileN = train_test_dir + problemN + "train";
                if (i < 10) fileN += "0";
                fileN += to_string(i) + "-";
                if (problemN == "G" || problemN == "H") fileN += "0";
                fileN += to_string(j) + ".txt";
                file.open(fileN);
                if (file.is_open()) {
                    vector<string> words;
                    string line;
                    while (getline(file, line)) {
                        line = p.remove_extra_space(line);
                        vector<string> temp = p.word_tokenize(line);
                        words.insert(words.end(), temp.begin(), temp.end());
                    }
                    
                    for (size_t k = 0; k < words.size(); ++k) {
                        auto found = find(features.begin(), features.end(), words[k]);
                        if (found != features.end()) {
                            frequency[features[found - features.begin()]]++;
                        }
                    }
                }
            }
        }
        
        for (auto iter = frequency.begin(); iter != frequency.end(); ++iter)
            temp.push_back({iter->first, iter->second});
        
        sort(temp.begin(), temp.end(),
             [](auto &pair1, auto &pair2) {
                 return pair1.second > pair2.second;
             });

        for (size_t i = 0; i < numFea; ++i) {
            features[i] = temp[i].first;
        }

        features.erase(features.begin() + numFea, features.end());
        
        this->train(prior, condprob, all_labels);
    
        for(int i =1; i < 14; ++i){
            string fileN = train_test_dir + problemN + "sample";
            if(i < 10) fileN += "0";
            fileN += to_string(i) + ".txt";
            results.push_back(this->test(prior, condprob, all_labels, fileN));
        }
        
        results.erase(remove(results.begin(), results.end(), -1), results.end());
        
        int correct = 0;
        for(size_t i = 0; i < actual.size(); ++i){
            if (actual[i] == results[i]) correct++;
        }
        cout << endl;
        cout << "The accuracy of problem" << problemN << " using "<<numFea <<" features";
        double accurate = (double(correct)/double(actual.size()))*100;
        cout <<" is "<<accurate<<"%"<<endl<<endl;
        output<<accurate;
        output<<endl;
    }
    
    //Define Train function
    void train(map<int, double>& prior, map<string, map<int, double>>& condprob,
               unordered_set<int>& all_labels) {
        map<string, map<int, int>> count;
        map<int, int> Nc;
        string problemN;
        problemN = train_test_dir.substr(train_test_dir.size()-2);
        problemN.erase(problemN.size()-1);
    
        int numFile = 0;
        for(int i =1; i < 14; ++i){
            for(int j = 1; j < 5; ++j){
                ifstream file;
                string fileN = train_test_dir + problemN + "train";
                if(i < 10) fileN += "0";
                fileN += to_string(i) + "-";
                if (problemN == "G" || problemN == "H") fileN += "0";
                fileN += to_string(j) + ".txt";
                file.open(fileN);
                if(file.is_open()){
                    vector<string> words;
                    string line;
                    while(getline(file, line)){
                        vector<string> temp = p.word_tokenize(line);
                        words.insert(words.end(), temp.begin(), temp.end());
                    }
                    all_labels.insert(i);
                    Nc[i]++; //File written by the author
                    numFile++; //Total number of files increase
                    for(size_t k = 0; k < features.size(); ++k){
                        auto found = find(words.begin(), words.end(), features[k]);
                        if (found != words.end()){
                            count[features[k]][i]++;
                        }
                    }
                }
            }
        }
        
        for(auto iter = Nc.begin(); iter != Nc.end(); ++iter){
            prior[(*iter).first] = double((*iter).second)/double(numFile);
        }
        
        for(size_t i = 0; i < features.size(); ++i){
            for (auto iter = all_labels.begin(); iter != all_labels.end(); iter++) {
                condprob[features[i]][*iter] = double(count[features[i]][*iter] + 1)/double(Nc[*iter] + 2);
            }
        }
    }
    
    //Define Test function
    int test(map <int, double>& prior, map <string, map<int, double> >& condprob,
             unordered_set<int>& all_labels, string testfilename){
        ifstream file;
        vector<std::pair<int, double>> results;
        vector<string> words;
        
        file.open(testfilename);
        if(!file.is_open()) return -1;
        
        string line;
        while(getline(file, line)){
            vector<string> temp = p.word_tokenize(line);
            words.insert(words.end(), temp.begin(), temp.end());
        }
        
        for(auto iter = all_labels.begin(); iter != all_labels.end(); ++iter){
            double feaSum = log2(prior[*iter]);
            for(size_t i = 0; i < features.size(); ++i){
                auto found = find(words.begin(), words.end(), features[i]);
                if(found == words.end()) feaSum += log2(1 - condprob[features[i]][*iter]);
                else if(found != words.end()) feaSum += log2(condprob[features[i]][*iter]);
            }
            results.push_back({*iter, feaSum});
        }
        
        sort(results.begin(), results.end(),
             [](auto &pair1,  auto &pair2){
                 return pair1.second > pair2.second;
             });
        
        auto iter = results.begin();
        return (*iter).first;
    }
};

struct FeatureRank {
    int featureIndex;
    double entropy;
    FeatureRank() {};
};

void sortFeatures(map <int, double>& prior, map <string, map<int, double> >& condprob,
                  unordered_set<int>& all_labels, vector<string> &vocabb) {
    vector<FeatureRank> fr;
    
    for (size_t i = 0; i < vocabb.size(); ++i) {
        FeatureRank newF;
        newF.featureIndex = i;
        newF.entropy = 0.0;
        for (auto iter = all_labels.begin(); iter != all_labels.end(); ++iter) {
            newF.entropy += prior[*iter] * condprob[vocabb[i]][*iter] * (log2(condprob[vocabb[i]][*iter]));
        }
        newF.entropy = (-1.0)*newF.entropy;
        fr.push_back(newF);
    }
    
    sort(fr.begin(), fr.end(),
         [](const FeatureRank &f1, const FeatureRank &f2) {
             return f1.entropy > f2.entropy;
         });
    
    cout << "Top 20 features:\n";
    for(size_t i = 0; i < 20; ++i) {
        cout << vocabb[fr[i].featureIndex] <<" "<<fr[i].entropy <<endl;
    }
    cout<<endl;
    return;
}

int main(int argc, char** argv) {
    string train_test_dirr;
    string currDirr(getenv("PWD"));
    vector<string> vocabb;
    PreProcess pp;
    map<int, double> prior;
    map<string, double> feaRank;
    map<string, map<int, double>> condprob;
    unordered_set<int> all_labels;
    vector<int>actual;
    vector<int>predicted;
    
    train_test_dirr = argv[1];
    ifstream fileFea;
    string word;
    string feaF = currDirr + "/stopwords.txt"; //Change to the directory of my code
    fileFea.open(feaF);
    if(fileFea.is_open()){
        while(getline(fileFea,word)){
            word = pp.remove_extra_space(word);
            vector<string> temp = pp.word_tokenize(word);
            vocabb.insert(vocabb.end(), temp.begin(), temp.end());
        }
    }
    
    BernoulliNaiveBayes bnb(train_test_dirr, vocabb, pp);
    bnb.train(prior, condprob, all_labels);
    
    string problemN;
    problemN = train_test_dirr.substr(train_test_dirr.size()-2);
    problemN.erase(problemN.size()-1);
    for(int i =1; i < 14; ++i){
        string fileN = train_test_dirr + problemN + "sample";
        if(i < 10) fileN += "0";
        fileN += to_string(i) + ".txt";
        predicted.push_back(bnb.test(prior, condprob, all_labels, fileN));
    }
    
    predicted.erase(remove(predicted.begin(), predicted.end(), -1), predicted.end());
    
    //Get the actual files
    ifstream fileG;
    string groudF = currDirr + "/test_ground_truth.txt";
    fileG.open(groudF);
    if(fileG.is_open()){
        string line;
        while (getline(fileG, line)) {
            string answer;
            if (line.substr(0, 8) == "problem" + problemN) {
                stringstream ss(line);
                ss >> line >> answer;
                answer.erase(answer.begin(), answer.begin() + 6);
                if (answer[0] == '0') { answer.erase(answer.begin()); actual.push_back(stoi(answer)); }
                else actual.push_back(stoi(answer));
            }
        }
    }

    
    int correct = 0;
    for(size_t i = 0; i < actual.size(); ++i){
        if (actual[i] == predicted[i]) correct++;
    }
    cout<<endl;
    cout << "The accuracy of problem" << problemN << " is "<<(double(correct)/double(actual.size()))*100<<"%";
    pp.printConfMat(actual, predicted);
    cout << endl;

	ofstream output;
	output.open("featureCurve" + problemN + ".csv");
	for (size_t i = 10; i < vocabb.size(); i += 10) {
		BernoulliNaiveBayes bnb1(train_test_dirr, vocabb, pp);
		bnb1.featureCurve(i, problemN, actual, all_labels, output);
	}
	output.close();
    sortFeatures(prior, condprob, all_labels, vocabb);
    return 0;
}

