//  solver.cpp
//  wumengxi@umich.edu

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace std;

struct Relation {
    vector<string>otherShape_id;
    string relation_type;
};

struct Shape{
    string id;
    string type;
    string sz;
    int center_x;
    int center_y;
    int sides = 0;
    Shape(void){}
};

struct Diagram{
    vector<Shape> shapes;
    vector<Relation> relation;
    string dia;
    Diagram(void){}
};

struct eleTransit{
    string former_id;
    string later_id;
    string action;
    int rank;
    eleTransit(void){};
};

struct diaTransit{
    vector<eleTransit> diaT;
    string diaA;
    string diaB;
    size_t A;
    size_t B;
    int rankSum;
    diaTransit(void){};
};

class Problem{
public:
    vector<Diagram>sampleA;
    vector<Diagram>sampleB;
    vector<Diagram>targetC;
    vector<Diagram>choiceK;
    diaTransit temp;
    diaTransit T_AB;
	diaTransit T_CK;
	string outFile;
    bool isContain(const eleTransit &t);
    bool isSame(const Shape &a, const Shape &b);
    bool isResize(const Shape &a, const Shape &b);
    bool isShift(const Shape &a, const Shape &b);
    bool isResizeThenShift(const Shape &a, const Shape &b);
    bool idVisited(const vector<string> &names, vector<bool> &visied, const string &s1, size_t &pos);
    bool isAdded(const vector<diaTransit> &C, const diaTransit &b);
    void Delete(const Shape &a);
    void Create(const Shape &b);
    void enumerate(const Diagram &A, const Diagram &B);
    void genComb(const vector<string> &names, vector<bool> visited, diaTransit &T);
    void processing(vector<Diagram> &AorC, vector<Diagram> &BorK, vector<diaTransit> &ABorCK);
    void readIn(string filename);
    void output();
};

void printT(const eleTransit &t){
    cout<<t.action<<" "<<t.former_id<<" "<<t.later_id<<endl;
}
bool Problem::isContain(const eleTransit &t){
    for(size_t i = 0; i < temp.diaT.size(); ++i){
        bool test = temp.diaT[i].former_id == t.former_id;
        test &= temp.diaT[i].later_id == t.later_id;
        test &= temp.diaT[i].rank == t.rank;
        test &= temp.diaT[i].action == t.action;
        if(test == true) return true;
    }
    return false;
}

bool Problem::isSame(const Shape &a,const Shape &b){
    bool test = a.type == b.type;//all same
    test &= a.center_x == b.center_x;
    test &= a.center_y == b.center_y;
    test &= a.sz == b.sz;
    test &= a.sides == b.sides;
    if(test == true){
        eleTransit t;
        t.former_id = a.id;
        t.later_id = b.id;
        t.action = "same";
        t.rank = 9;
        printT(t);
        if(!isContain(t)) {temp.diaT.push_back(t); printT(t);}
        return true;
    }
    return false;
}

bool Problem::isResize(const Shape &a, const Shape &b){
    bool test = a.sz != b.sz;//diff at size
    test &= a.center_x == b.center_x;
    test &= a.center_y == b.center_y;
    test &= a.sides == b.sides;
    test &= a.type == b.type;
    if(test == true){
        eleTransit t;
        t.former_id = a.id;
        t.later_id = b.id;
        t.action = "resize";
        t.rank = 7;
        if(!isContain(t)) {temp.diaT.push_back(t); printT(t);}
        return true;
    }
    return false;
}

bool Problem::isShift(const Shape &a, const Shape &b){
    if(a.center_x == b.center_x && a.center_y == b.center_y) return false;//diff at position
    bool test = a.sz == b.sz;
    test &= a.sides == b.sides;
    test &= a.type == b.type;
    if(test == false) return false;
    eleTransit t;
    t.former_id = a.id;
    t.later_id = b.id;
    t.rank = 5;
    t.action = "shift";
    if(b.center_x < a.center_x) t.action += "-left";
    if(b.center_x > a.center_x) t.action += "-right";
    if(b.center_y < a.center_y) t.action += "-up";
    if(b.center_y > a.center_y) t.action += "-down";
    if(!isContain(t)) {temp.diaT.push_back(t); printT(t);}
    return true;
}

bool Problem::isResizeThenShift(const Shape &a, const Shape &b){
    if(a.type != b.type || a.sides != b.sides) return false; //same type and sides fails
    if(a.sz == b.sz) return false;//size diff fails
    if(a.center_x == b.center_x && a.center_y == b.center_y) return false;//center diff fails
    eleTransit t;
    t.former_id = a.id;
    t.later_id = b.id;
    t.rank = 4;
    t.action = "resize_Then_shift";
    if(b.center_x < a.center_x) t.action += "-left";
    if(b.center_x > a.center_x) t.action += "-right";
    if(b.center_y < a.center_y) t.action += "-up";
    if(b.center_y > a.center_y) t.action += "-down";
    if(!isContain(t)) {temp.diaT.push_back(t); printT(t);}
    return true;
}

void Problem::Delete(const Shape &a){
    eleTransit t;
    t.former_id = a.id;
    t.later_id = "0";
    t.rank = 1;
    t.action = "delete";
    if(!isContain(t)) {temp.diaT.push_back(t); printT(t);}
    return;
}

void Problem::Create(const Shape &b){
    eleTransit t;
    t.former_id = "0";
    t.later_id = b.id;
    t.rank = 0;
    t.action = "create";
    if(!isContain(t)) {temp.diaT.push_back(t); printT(t);}
    return;
}

void Problem::enumerate(const Diagram &A, const Diagram &B){
    for(size_t i = 0; i < A.shapes.size(); ++i){
        for(size_t j = 0; j < B.shapes.size(); ++j){
            if(isSame(A.shapes[i], B.shapes[j])){;}
            else if(isResize(A.shapes[i], B.shapes[j])){;}
            else if(isShift(A.shapes[i], B.shapes[j])){;}
            else if(isResizeThenShift(A.shapes[i], B.shapes[j])){;}
            Delete(A.shapes[i]);
            Create(B.shapes[j]);
        }
    }
    std::sort(temp.diaT.begin(), temp.diaT.end(),
              [](const eleTransit &lhs, const eleTransit &rhs){return lhs.rank > rhs.rank;});
    return;
}

bool Problem::idVisited(const vector<string> &names, vector<bool> &visited,
                        const string &s1, size_t &pos){
    auto found = std::find(names.begin(), names.end(), s1);
    if(visited[found-names.begin()] == true) return true;
    pos = found - names.begin();//get the position
    return false;
}

void Problem::genComb(const vector<string> &names, vector<bool> visited, diaTransit &T){
    for(size_t i = 0; i < temp.diaT.size(); ++i){
        size_t pos1 = 10000;
        size_t pos2 = 10000;
        bool add = true;
        if(temp.diaT[i].former_id != "0"){
            if(idVisited(names, visited, temp.diaT[i].former_id, pos1)) add = false;
        }
        if(temp.diaT[i].later_id != "0"){
            if(idVisited(names, visited, temp.diaT[i].later_id, pos2)) add = false;
        }
        if(add == true) {
            T.diaT.push_back(temp.diaT[i]);
            if(pos1 != 10000) visited[pos1] = true;
            if(pos2 != 10000) visited[pos2] = true;
        }
    }
    std::sort(T.diaT.begin(), T.diaT.end(),
              [](const eleTransit &lhs, const eleTransit &rhs){return lhs.rank > rhs.rank;});
    return;
}

bool Problem::isAdded(const vector<diaTransit> &C, const diaTransit &b){
    for(size_t i = 0; i < C.size(); i++){
        const diaTransit &a = C[i];
        bool test = true;
       if(a.diaT.size()== b.diaT.size()){
        for(size_t j = 0; j < a.diaT.size(); ++j){
            test &= a.diaT[j].action == b.diaT[j].action;
            test &= a.diaT[j].former_id == b.diaT[j].former_id;
            test &= a.diaT[j].later_id == b.diaT[j].later_id;
            test &= a.diaT[j].rank == b.diaT[j].rank;
        }
        test &= a.diaA == b.diaA;
        test &= a.A == b.A;
        test &= a.B == b.B;
        if(test == true) return true;
       }
    }
    return false;
}

void Problem::processing(vector<Diagram> &AorC, vector<Diagram> &BorK, vector<diaTransit> &ABorCK){
    //Enumerate action
    for(size_t i = 0; i < AorC.size(); ++i){
        for(size_t j = 0; j < BorK.size(); ++j){
            temp = diaTransit();
            temp.diaA = AorC[i].dia;
            temp.A = i;
            temp.diaB = BorK[j].dia;
            temp.B = j;
            enumerate(AorC[i], BorK[j]);
            cout<<endl<<endl;
            cout<<temp.diaA<<" "<<temp.diaB<<endl;;
            for_each(temp.diaT.begin(), temp.diaT.end(), [this](const eleTransit &t){
                cout<<t.former_id<<" "<<t.later_id<<" "<<t.rank<<endl;
            });
            //Generate correspondance
            vector<string>idInAB;
            for(size_t k = 0; k < AorC[i].shapes.size(); ++k){
                idInAB.push_back(AorC[i].shapes[k].id);
            }
            for(size_t k = 0; k < BorK[j].shapes.size(); ++k){
                idInAB.push_back(BorK[j].shapes[k].id);
            }
            for(size_t k = 0; k < temp.diaT.size(); ++k){
                diaTransit T;
                size_t pos;
                T.diaA = AorC[i].dia;
                T.diaB = BorK[j].dia;
				T.A = i;
				T.B = j;
                vector<bool>visited(idInAB.size(), false);
                if(temp.diaT[k].former_id != "0"){
                    if(!idVisited(idInAB, visited, temp.diaT[k].former_id, pos)){
                        visited[pos] = true;
                    }
                }
                if(temp.diaT[k].later_id != "0"){
                    if(!idVisited(idInAB, visited, temp.diaT[k].later_id, pos)){
                        visited[pos] = true;
                    }
                }
                T.diaT.push_back(temp.diaT[k]);
                genComb(idInAB, visited, T);
                if(isAdded(ABorCK, T) == false) ABorCK.push_back(T);
            }
        }
    }
}

void Problem::readIn(string filename){
    ifstream file;
    string line = " ";
    file.open(filename);
    if(!file.is_open()){cout<<"Error openning file!\n"; return;}
    Diagram diagram;
    while (getline(file, line)){
        if(line[0] == 'A'||line[0]=='B'||line[0]=='C'||line[0]=='K'){
            if(diagram.dia[0] =='A') sampleA.push_back(diagram);
            else if(diagram.dia[0] =='B') sampleB.push_back(diagram);
            else if(diagram.dia[0] =='C') targetC.push_back(diagram);
            else if(diagram.dia[0] =='K') choiceK.push_back(diagram);
            diagram = Diagram();
            diagram.dia = line;
            cout<<diagram.dia<<endl;
        }
       else if(line[0] == 'r'||line[0]=='c'||line[0]=='d'||line[0]=='s'||line[0]=='d'||line[0]=='t'){
            Shape s;
            stringstream ss(line);
            string parse = " ";
            ss>>s.id>>parse;
            ss>>parse;
            std::size_t found = parse.find_first_not_of("abcdefghijklmnopqrstuvwxyz");
            s.type = parse.substr(0, found);
            parse.erase(parse.end()-2, parse.end());
            parse.erase(parse.begin(), parse.begin()+found+2);
            s.sz = parse;
           ss>>s.center_x>>parse>>s.center_y>>parse;
            if(s.type == "scc") ss>>s.sides;
            cout<<s.type<<" "<<s.sz<<" "<<s.center_x<<" "<<s.center_y<<" "<<s.id<<" "<<s.sides<<endl;
           diagram.shapes.push_back(s);
        }
        else if(line[0]=='l' || line[0] == 'a' || line[0] == 'o'||line[0]=='i'){
            Relation r;
            stringstream ss(line);
            string parse = " ";
            ss>>parse;
            std::size_t found = parse.find_first_not_of("abcdefghijklmnopqrstuvwxyz_");
            r.relation_type = parse.substr(0, found);
            parse.erase(parse.end()-1,parse.end());
            r.otherShape_id.push_back(parse.substr(found+1));
            ss>>parse;
            parse.erase(parse.end()-1, parse.end());
            r.otherShape_id.push_back(parse);
            cout<<r.relation_type<<" "<<r.otherShape_id[0]<<" "<<r.otherShape_id[1]<<endl;
            diagram.relation.push_back(r);
        }
    }
    choiceK.push_back(diagram);
    return;
}

void Problem::output(){
	ofstream os;
	os.open(outFile + ".txt");
    os<<"Answer: "<<T_CK.diaB<<endl;
    os<<"T_"<<T_AB.diaA<<T_AB.diaB<<endl;
    os<<"[";
    for(size_t i = 0; i < T_AB.diaT.size(); ++i){
        os<<"("<<T_AB.diaT[i].former_id<<", "<<T_AB.diaT[i].later_id<<")";
        if(i+1 != T_AB.diaT.size()) os<<", ";
    }
    os<<"]"<<endl;
    for(size_t i = 0; i < T_AB.diaT.size(); ++i){
        os<<"action("<<"'"<<T_AB.diaT[i].action<<"'"<<", ";
        if(T_AB.diaT[i].action[0] == 'd') os<<T_AB.diaT[i].former_id;
        else if(T_AB.diaT[i].action[0] == 'c') os<<T_AB.diaT[i].later_id;
        else  os<<T_AB.diaT[i].former_id<<", "<<T_AB.diaT[i].later_id;
        os<<")";
        os<<endl;
    }
    os<<endl;
    os<<"T_"<<T_CK.diaA<<T_CK.diaB<<endl;
    os<<"[";
    for(size_t i = 0; i < T_CK.diaT.size(); ++i){
        os<<"("<<T_CK.diaT[i].former_id<<", "<<T_CK.diaT[i].later_id<<")";
        if(i+1 != T_CK.diaT.size()) os<<", ";
    }
    os<<"]"<<endl;
    for(size_t i = 0; i < T_CK.diaT.size(); ++i){
        os<<"action("<<"'"<<T_CK.diaT[i].action<<"'"<<", ";
        if(T_CK.diaT[i].action[0] == 'd') os<<T_CK.diaT[i].former_id;
        else if(T_CK.diaT[i].action[0] == 'c') os<<T_CK.diaT[i].later_id;
        else  os<<T_CK.diaT[i].former_id<<", "<<T_CK.diaT[i].later_id;
        os<<")";
        os<<endl;
    }
	os.close();
    return;
}

int main(int argc, const char *argv[]) {
    string filename = " ";
	filename = argv[1];
	Problem problem;
	problem.outFile = argv[2];
    vector<diaTransit> AB;
    vector<diaTransit> CK;
    problem.readIn(filename);
    problem.processing(problem.sampleA, problem.sampleB, AB);
    problem.processing(problem.targetC, problem.choiceK, CK);
    std::sort(AB.begin(), AB.end(),
              [](const diaTransit &lhs, const diaTransit &rhs){return lhs.diaT.size() < rhs.diaT.size();});
    std::sort(CK.begin(), CK.end(),
              [](const diaTransit &lhs, const diaTransit &rhs){return lhs.diaT.size() < rhs.diaT.size();});
 
   cout << "Show correspondence:\n";
   for (size_t i = 0; i < AB.size(); ++i) {
	   cout << AB[i].diaA << " " << AB[i].diaB << endl;
	   for (size_t j = 0; j < AB[i].diaT.size(); ++j) {
		   cout << AB[i].diaT[j].action << " " << AB[i].diaT[j].former_id << " " << AB[i].diaT[j].later_id << " " << AB[i].diaT[j].rank << endl;
	   }
	   cout << endl;
   }
    
   for (size_t i = 0; i < CK.size(); ++i) {
	   cout << CK[i].diaA << " " << CK[i].diaB << endl;
	   for (size_t j = 0; j < CK[i].diaT.size(); ++j) {
		   cout << CK[i].diaT[j].action << " " << CK[i].diaT[j].former_id << " " << CK[i].diaT[j].later_id << " " << CK[i].diaT[j].rank << endl;
	   }
	   cout << endl;
   }

   problem.T_AB = AB[0];
    for(size_t i = 0; i < CK.size(); ++i){
          bool test0 = true;
        for(size_t j = 0; j < problem.T_AB.diaT.size(); ++j){
            bool test1 = false;
            const eleTransit &a = problem.T_AB.diaT[j];
            for(size_t k = 0; k < CK[i].diaT.size(); ++k){
                const eleTransit &b = CK[i].diaT[k];
                if(a.action == b.action) test1 = true;
            }
            test0 &= test1;
        }
        if(test0 == true && CK[i].diaT.size() == problem.T_AB.diaT.size()){
			for (size_t j = 0; j < CK[i].diaT.size(); ++j) {
				if(CK[i].diaT[j].rank != problem.T_AB.diaT[j].rank)
					test0 = false;
			}

			cout << problem.T_AB.A << endl;
			for (size_t j = 0; j < problem.sampleA[problem.T_AB.A].relation.size(); ++j) {
				bool test2 = false;
				const Relation &a = problem.sampleA[problem.T_AB.A].relation[j];
				for (size_t k = 0; k < problem.targetC[CK[i].A].relation.size(); ++k) {
					const Relation &b = problem.targetC[CK[i].A].relation[k];
					if (a.relation_type == b.relation_type) test2 = true;
				}
				test0 &= test2;
			}


			for (size_t j = 0; j < problem.sampleB[problem.T_AB.B].relation.size(); ++j) {
				bool test3 = false;
				const Relation &a = problem.sampleB[problem.T_AB.B].relation[j];
				for (size_t k = 0; k < problem.choiceK[CK[i].B].relation.size(); ++k) {
					const Relation &b = problem.choiceK[CK[i].B].relation[k];
					if (a.relation_type == b.relation_type) test3 = true;
				}
				test0 &= test3;
			}

			if (test0 == true) {
				problem.T_CK = CK[i];
				break;
			}
        }
    }
    problem.output();
    return 0;
}


