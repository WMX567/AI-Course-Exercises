// Markov Decision Process simulation
// wumengxi@umich.edu

#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include <numeric>
#include <iomanip>
#include <cctype>
#include <cstdlib>
#include <algorithm>
using namespace std;

struct Point {
    size_t x;
    size_t y;
};

struct Policy{
    vector<vector<double>> util = vector<vector<double>>(4, vector<double>(5,0.0));
    vector<vector<string>> actions = vector<vector<string>>(4, vector<string>(5, " "));
    double rs = 0.0;
};

struct Policy3{
    vector<vector<double>> util = vector<vector<double>>(4, vector<double>(4,0.0));
    vector<vector<string>> actions = vector<vector<string>>(4, vector<string>(4, " "));
    double ga = 0.0;
};

bool notWall(size_t i, size_t j){
    return !(i == 2 && j == 2);
}

bool isTerminal(size_t x, size_t y){
    if(y!=4) return false;
    else if(x == 1) return true;
    else if(x == 2) return true;
    return false;
}

class MDP{
    map<size_t, string> actList;
    vector<Policy> policies;
    double rewards[4][5];
    Policy result;
    Point terminal_1;
    Point terminal_2;
    double gamma = 1;
	string direct;

public:
    
    void printMatrix(const Policy &r){
        Policy Fig17_3;
        Fig17_3.util = {{0,0,0,0,0},{0, 0.812, 0.868, 0.918, 1},{0, 0.762, 0, 0.660, -1},
            {0, 0.705, 0.655, 0.611, 0.388}};
        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 5; ++j)
                cout<<r.actions[i][j]<<"("<<r.util[i][j]<<")"<<" ";
            cout<<endl;
        }
    }
    
    void setUp(double RS, string dir){
        result.rs = RS;
		direct = dir;
        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 5; ++j){
                rewards[i][j] = RS; //A range of R(x)
            }
        }
        
        terminal_1.x = 1;
        terminal_1.y = 4;
        terminal_2.x = 2;
        terminal_2.y = 4;
        
        result.util[1][4] = rewards[1][4]= 1; //terminal 1
        result.util[2][4] = rewards[2][4]= -1; //terminal 2
        
        rewards[2][2]= 0; //Wall
        
        actList[0] = "^"; //Up
        actList[1] = "v"; //Down
        actList[2] = ">"; //Right
        actList[3] = "<"; //Left
        result.actions[2][2] = "X";
        result.actions[1][4] = "+1";
        result.actions[2][4] = "-1";
        
        policies.reserve(11);
    }
    
    double maxPU(size_t i, size_t j, const vector<vector<double>> &U){
        vector<double> fourDirect(4,U[i][j]);
        vector<double> predDirect(4, 0.0);
        //Direction availability
        if(notWall(i-1, j)&& 1<=i-1) fourDirect[0] = U[i-1][j]; //Up
        if(notWall(i+1, j)&& i+1<=3) fourDirect[1] = U[i+1][j]; //Down
        if(notWall(i, j+1)&& j+1<=4) fourDirect[2] = U[i][j+1]; //Right
        if(notWall(i, j-1)&& 1<=j-1) fourDirect[3] = U[i][j-1]; //Left
        
        predDirect[0] = fourDirect[0]*0.8 + (fourDirect[2] + fourDirect[3])*0.1; //Up
        predDirect[1] = fourDirect[1]*0.8 + (fourDirect[2] + fourDirect[3])*0.1; //Down
        predDirect[2] = fourDirect[2]*0.8 + (fourDirect[0] + fourDirect[1])*0.1; //Right
        predDirect[3] = fourDirect[3]*0.8 + (fourDirect[0] + fourDirect[1])*0.1; //Left
        
        auto maxEle = max_element(predDirect.begin(), predDirect.end());
        result.actions[i][j] = actList[maxEle - predDirect.begin()];
        return predDirect[maxEle - predDirect.begin()];
    }
    
    bool validTail(size_t i, size_t j){
        if(!notWall(i, j)) return false; //encounter Wall
        else if(i == terminal_1.x && j == terminal_1.y) return false; //encounter terminal 1
        else if(i == terminal_2.x && j == terminal_2.y) return false;
        return true;
    }
    
    void optimalSolu(const vector<vector<double>> &U){
        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 5; ++j){
                if(validTail(i, j)){
                    maxPU(i, j, U);
                }
            }
        }
        //printMatrix(result);
    }
    
    void valueIteration(){
        double delta = 100;
        vector<vector<double>> uNew = vector<vector<double>>(4, vector<double>(5, 0.0));
        vector<vector<double>> uTemp = vector<vector<double>>(4, vector<double>(5, 0.0));
        int numIter = 0;
        
        uNew[1][4] = uTemp[1][4] = 1;
        uNew[2][4] = uTemp[2][4] = -1;
        uTemp[2][2] = 0;
        
        do{
            numIter += 1;
            delta = 0;
            uNew = uTemp;
            for(size_t i = 1; i < 4; ++i){
                for(size_t j = 1; j < 5; ++j){
                    if(validTail(i, j)){
                        double max = maxPU(i, j, uNew);
                        uTemp[i][j] = rewards[i][j] + max;
                        if(fabs(uTemp[i][j] - uNew[i][j]) > delta)
                            delta = fabs(uTemp[i][j] - uNew[i][j]);
                    }
                }
            }
            
        }while(0.0000001 <= delta);
        result.util = uNew;
        optimalSolu(result.util);
    }
    
    void searchThreshold(double startP, double endP, string dir){
        if(policies.size() == 8 || fabs(endP - startP) <= 0.0001) return;
        double mid = (startP + endP)/2.0;
        Policy p1, p2;
        p1.rs = mid - 0.0001;
        p2.rs = mid + 0.0001;
        this->setUp(p1.rs, dir);
        this->valueIteration();
        p1 = result;
        this->setUp(p2.rs, dir);
        this->valueIteration();
        p2 = result;
        
        if(p1.actions != p2.actions){
            bool add = true;
            for(size_t i = 0; i < policies.size(); ++i){
                if(policies[i].actions == p2.actions) {add = false; break;}
            }
            if(add == true) {p2.rs = mid; policies.push_back(p2);}
        }
        
        if(policies.size() == 8) return;
        
        if(fabs(mid - startP) > 0.0001) searchThreshold(startP, mid, dir);
        if(fabs(endP - mid) > 0.0001) searchThreshold(mid, endP, dir);
        
        return;
    }
    
    void outPut(){
        ofstream os;
        os.open(direct + "P1-output.txt");
        this->setUp(0, direct);
        this->valueIteration();
        policies.push_back(result);
        os<<"Problem 1:"<<endl<<endl;
        reverse(policies.begin(), policies.end());
        this->setUp(policies[policies.size()-1].rs - 0.0001, direct);
        this->valueIteration();
        policies.push_back(result);
        for(size_t i = 1; i < policies.size(); ++i){
            if(1 <= i) os<<fixed<<setprecision(4)<<policies[i-1].rs<<endl;
            //if(i + 1 == policies.size()) os<<"-Infinity \n";
            //else os<<policies[i].rs<<endl;
			os << "Optimal Policy:" << endl;
            for(size_t j = 1; j < 4; ++j){
                for(size_t k = 1; k < 5; ++k){
                    os<<policies[i].actions[j][k];
                    if(1 <= i){
                        if(policies[i-1].actions[j][k] != policies[i].actions[j][k])
                            os<<"*";
                    }
                    os<<" ";
                }
                os<<endl;
            }
            os<<endl;
        }
    }
    
    Policy getPolicy(){
        return result;
    }
};


class MDP3{
    map<size_t, string> actList;
    vector<Policy3> policies;
    double rewards[4][4];
    Policy3 result;
    Point terminal;
    double gamma = 1;
	string direct;
    
public:
    
    void printMatrix(const Policy3 &r){
        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 4; ++j)
                cout<<r.actions[i][j]<<"("<<r.util[i][j]<<")"<<" ";
            cout<<endl;
        }
    }
    
    void setUp(double GA, string dir){
        
        result.ga = GA;
        gamma = GA;
		direct = dir;

        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 4; ++j){
                rewards[i][j] = -1; //A range of R(x)
            }
        }
        
        terminal.x = 1;
        terminal.y = 3;
        
        result.util[1][3] = rewards[1][3]= 10; //terminal 1
        rewards[1][1] = 3;
        
        actList[0] = "^"; //Up
        actList[1] = "v"; //Down
        actList[2] = ">"; //Right
        actList[3] = "<"; //Left
        result.actions[1][3] = "+10";
        
        policies.reserve(7);
    }
    
    double maxPU(size_t i, size_t j, const vector<vector<double>> &U){
        vector<double> fourDirect(4,U[i][j]);
        vector<double> predDirect(4, 0.0);
        //Direction availability
        if(1<=i-1) fourDirect[0] = U[i-1][j]; //Up
        if(i+1<=3) fourDirect[1] = U[i+1][j]; //Down
        if(j+1<=3) fourDirect[2] = U[i][j+1]; //Right
        if(1<=j-1) fourDirect[3] = U[i][j-1]; //Left
        
        predDirect[0] = fourDirect[0]*0.8 + (fourDirect[2] + fourDirect[3])*0.1; //Up
        predDirect[1] = fourDirect[1]*0.8 + (fourDirect[2] + fourDirect[3])*0.1; //Down
        predDirect[2] = fourDirect[2]*0.8 + (fourDirect[0] + fourDirect[1])*0.1; //Right
        predDirect[3] = fourDirect[3]*0.8 + (fourDirect[0] + fourDirect[1])*0.1; //Left
        
        auto maxEle = max_element(predDirect.begin(), predDirect.end());
        result.actions[i][j] = actList[maxEle - predDirect.begin()];
        return predDirect[maxEle - predDirect.begin()];
    }
    
    bool validTail(size_t i, size_t j){
        if(i == terminal.x && j == terminal.y) return false;
        return true;
    }
    
    void optimalSolu(const vector<vector<double>> &U){
        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 4; ++j){
                if(validTail(i, j)){
                    maxPU(i, j, U);
                }
            }
        }
        //printMatrix(result);
    }
    
    void valueIteration(){
        double delta = 100;
        vector<vector<double>> uNew = vector<vector<double>>(4, vector<double>(4, 0.0));
        vector<vector<double>> uTemp = vector<vector<double>>(4, vector<double>(4, 0.0));
        int numIter = 0;
        
        uNew[1][3] = uTemp[1][3] = 10;
        
        do{
            numIter += 1;
            delta = 0;
            uNew = uTemp;
            for(size_t i = 1; i < 4; ++i){
                for(size_t j = 1; j < 4; ++j){
                    if(validTail(i, j)){
                        double max = maxPU(i, j, uNew);
                        uTemp[i][j] = rewards[i][j] + gamma*max;
                        if(fabs(uTemp[i][j] - uNew[i][j]) > delta)
                            delta = fabs(uTemp[i][j] - uNew[i][j]);
                    }
                }
            }
            
        }while(0.0000001 <= delta);
        result.util = uNew;
        optimalSolu(result.util);
    }
    
    
    bool isBadTail(size_t i, size_t j){
        if(i == 1 && j == 1) return true;
        else if(i == 1 && j == 3) return true;
        return false;
    }
    
    bool compPolicy(const Policy3 p1, const Policy3 p2){
        bool equal = true;
        for(size_t i = 1; i < 4; ++i){
            for(size_t j = 1; j < 4; ++j){
                if(!isBadTail(i, j)){
                    if(p1.actions[i][j] != p2.actions[i][j]){
                        equal = false;
                    }
                }
            }
        }
        return equal;
    }
    
    void searchThreshold(double startP, double endP, string dir){
        if(policies.size() == 5 || fabs(endP - startP) < 0.00001) return;
        double mid = (startP + endP)/2.0;
        Policy3 p1, p2;
        p1.ga = mid - 0.00001;
        p2.ga = mid + 0.00001;
        this->setUp(p1.ga, dir);
        this->valueIteration();
        p1 = result;
        this->setUp(p2.ga, dir);
        this->valueIteration();
        p2 = result;
        
        if(!compPolicy(p1, p2)){
            bool add = true;
            for(size_t i = 0; i < policies.size(); ++i){
                if(compPolicy(policies[i], p2)) {add = false; break;}
            }
            if(add == true) {p2.ga = mid; policies.push_back(p2);}
        }
        
        if(policies.size() == 5) return;
        
        if(fabs(mid - startP) > 0.00001) searchThreshold(startP, mid, dir);
        if(fabs(endP - mid) > 0.00001) searchThreshold(mid, endP, dir);
        
        return;
    }
    
    void outPut(){
        ofstream os;
        os.open(direct + "P3-output.txt");
        this->setUp(0.00001, direct);
        this->valueIteration();
        policies.insert(policies.begin(),result);
        os<<"Problem 3:"<<endl<<endl;
        for(size_t i = 0; i < policies.size(); ++i){
            os<<fixed<<setprecision(5)<<policies[i].ga<<endl;
			os << "Optimal Policy:" << endl;
            for(size_t j = 1; j < 4; ++j){
                for(size_t k = 1; k < 4; ++k){
                    os<<policies[i].actions[j][k];
                    if(1 <= i){
                        if(policies[i-1].actions[j][k] != policies[i].actions[j][k])
                            os<<"*";
                    }
                    os<<" ";
                }
                os<<endl;
            }
			os << endl;
			os << "Utilities:" << endl;
			for (size_t j = 1; j < 4; ++j) {
				for (size_t k = 1; k < 4; ++k) {
					os << policies[i].util[j][k] << " ";
				}
				os << endl;
			}
			os << endl << endl;
        }
    }
    
    Policy3 getPolicy(){
        return result;
    }
};


int main(){
    double startP = -2;
    double endP = -0.0001;
    double test = -0.0400;

	string currDir(getenv("PWD"));
	cout << currDir << endl;
	currDir.erase(currDir.end()-4, currDir.end());
	currDir += "Generated/";
	cout << currDir << endl;

    MDP mdp;
    cout<<"Problem 1\n";
    mdp.searchThreshold(startP, endP, currDir);
    mdp.outPut();
    
    cout<<"Problem 2\n";
    mdp.setUp(test, currDir);
    mdp.valueIteration();
    
    Policy bestPolicy;
    bestPolicy = mdp.getPolicy();
    vector<double> iterResult;
    
    vector<vector<double>> rewards = vector<vector<double>>(4, vector<double>(5, -0.0400));
    rewards[1][4]= 1;
    rewards[2][4]= -1;
    rewards[2][2]= 0;
    
    Point state;
    int numIter = 10;
    int dice = -1;
    string optAct = " ";
    ofstream outTxt;
    outTxt.open(currDir + "P2-output.txt");
	outTxt << fixed << setprecision(4);
    outTxt<<"Expected utility: "<<bestPolicy.util[3][4]<<endl;
    
    while(numIter < 10000){

        iterResult = vector<double>(numIter, 0.00000);
        ofstream outputX, outputY, data;
        outputX.open(currDir + "Run" + to_string(numIter) + "X.csv");
        outputY.open(currDir + "Run" + to_string(numIter) + "Y.csv");
		data.open(currDir + "P2-data-" + to_string(numIter) + ".txt");
		outputX << fixed << setprecision(4);
		outputY << fixed << setprecision(4);
		data << fixed << setprecision(4);

        for(size_t i = 0; i < numIter; ++i){
            state.x = 3;
            state.y = 4;
            while(!isTerminal(state.x, state.y)){
                dice = rand() % 10;
                optAct = bestPolicy.actions[state.x][state.y];
                Point target, side1, side2;
                target = side1 = side2 = state;
                
                if(optAct == "^"){
                    if(notWall(state.x-1, state.y) && 1 <= state.x-1) target.x -= 1;
                }
                else if(optAct == "v"){
                    if(notWall(state.x+1, state.y) && state.x+1 <= 3) target.x += 1;
                }
                else if(optAct == "<"){
                    if(notWall(state.x, state.y-1) && 1 <= state.y-1) target.y -= 1;
                }
                else if(optAct == ">"){
                    if(notWall(state.x, state.y+1) && state.y+1 <= 4) target.y += 1;
                }
                
                if(target.x != state.x){
                    if(notWall(state.x, state.y-1) && 1 <= state.y-1) side1.y -= 1;
                    if(notWall(state.x, state.y+1) && state.y+1 <= 4) side2.y += 1;
                }
                else if(target.y != state.y){
                    if(notWall(state.x-1, state.y) && 1 <= state.x-1) side1.x -= 1;
                    if(notWall(state.x+1, state.y) && state.x+1 <= 3) side2.x += 1;
                }
                
                if(dice == 0){
                    iterResult[i] += rewards[side1.x][side1.y];
                    state = side1;
                }
                else if(dice == 1){
                    iterResult[i] += rewards[side2.x][side2.y];
                    state = side2;
                }
                else{
                    iterResult[i] += rewards[target.x][target.y];
                    state = target;
                }
            }
        }
        
        map<double, size_t> valueTable;
        for(size_t i = 0; i < iterResult.size(); ++i){
            valueTable[iterResult[i]]++;
			data << iterResult[i] << endl;
        }
        
        for(auto iter = valueTable.begin(); iter != valueTable.end(); ++iter){
            outputX<<(*iter).first<<endl;
            outputY<<(*iter).second<<endl;
        }
        
        outputX.close();
        outputY.close();
		data.close();
        //Average
        double sum = accumulate(iterResult.begin(), iterResult.end(), 0.0);
        double mean = sum/iterResult.size();
        
        //Standard Deviation
        vector<double>diff(iterResult.size());
        transform(iterResult.begin(), iterResult.end(), diff.begin(), bind2nd(minus<double>(), mean));
        double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double std = sqrt(sq_sum/iterResult.size());
        
        outTxt<<numIter<<"-run average utility: "<<mean<<endl;
        outTxt<<numIter<<"-run standard dev: "<<std<<endl;
        numIter *= 10;
    }
    
    outTxt.close();
	cout << "Problem 3\n";
    MDP3 mdp3;
    mdp3.searchThreshold(0.00001, 1.00000, currDir);
    mdp3.outPut();
    return 0;
}
