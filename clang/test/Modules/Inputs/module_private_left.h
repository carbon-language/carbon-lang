__module_private__ struct HiddenStruct;

struct HiddenStruct {
};


int &f0(int);

template<typename T>
__module_private__ void f1(T*);

template<typename T>
void f1(T*);

template<typename T>
__module_private__ class vector;

template<typename T>
class vector {
};

vector<float> vec_float;

typedef __module_private__ int Integer;
typedef int Integer;

