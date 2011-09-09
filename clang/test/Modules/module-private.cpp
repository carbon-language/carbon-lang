// RUN: mkdir -p %t
// RUN: %clang_cc1 -x c++ -emit-module -o %t/left.pcm %s -D MODULE_LEFT
// RUN: %clang_cc1 -x c++ -emit-module -o %t/right.pcm %s -D MODULE_RIGHT
// RUN: %clang_cc1 -I %t %s -verify

#if defined(MODULE_LEFT)

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

#elif defined(MODULE_RIGHT)
__module_private__ double &f0(double);
double &f0(double);

__module_private__ int hidden_var;

inline void test_f0_in_right() {
  double &dr = f0(hidden_var);
}
#else
__import_module__ left;
__import_module__ right;

void test() {
  int &ir = f0(1.0); // okay: f0() from 'right' is not visible
}

int test_broken() {
  HiddenStruct hidden; // expected-error{{use of undeclared identifier 'HiddenStruct'}}

  Integer i; // expected-error{{use of undeclared identifier 'Integer'}}

  int *ip = 0;
  f1(ip); // expected-error{{use of undeclared identifier 'f1'}}

  vector<int> vec; // expected-error{{use of undeclared identifier 'vector'}} \
  // expected-error{{expected '(' for function-style cast or type construction}} \
  // expected-error{{use of undeclared identifier 'vec'}}

  return hidden_var; // expected-error{{use of undeclared identifier 'hidden_var'}}
}

#endif
