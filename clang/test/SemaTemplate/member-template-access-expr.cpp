// RUN: clang-cc -fsyntax-only -verify %s
template<typename U, typename T>
U f0(T t) {
  return t.template get<U>();
}

template<typename U, typename T>
int &f1(T t) {
  // FIXME: When we pretty-print this, we lose the "template" keyword.
  return t.U::template get<int&>();
}

struct X {
  template<typename T> T get();
};

void test_f0(X x) {
  int i = f0<int>(x);
  int &ir = f0<int&>(x);
}

struct XDerived : public X {
};

void test_f1(XDerived xd) {
  int &ir = f1<X>(xd);
}

// PR5213
template <class T>
struct A {};

template<class T>
class B
{
  A<T> a_;
  
public:
  void destroy();
};

template<class T>
void
B<T>::destroy()
{
  a_.~A<T>();
}

void do_destroy_B(B<int> b) {
  b.destroy();
}

struct X1 {
  int* f1(int);
  template<typename T> float* f1(T);
  
  static int* f2(int);
  template<typename T> static float* f2(T);
};

void test_X1(X1 x1) {
  float *fp1 = x1.f1<>(17);
  float *fp2 = x1.f1<int>(3.14);
  int *ip1 = x1.f1(17);
  float *ip2 = x1.f1(3.14);
  
  float* (X1::*mf1)(int) = &X1::f1;
  float* (X1::*mf2)(int) = &X1::f1<>;
  float* (X1::*mf3)(float) = &X1::f1<float>;
  
  float* (*fp3)(int) = &X1::f2;
  float* (*fp4)(int) = &X1::f2<>;
  float* (*fp5)(float) = &X1::f2<float>;  
  float* (*fp6)(int) = X1::f2;
  float* (*fp7)(int) = X1::f2<>;
  float* (*fp8)(float) = X1::f2<float>;  
}

template<int A> struct X2 { 
  int m;
};

template<typename T>
struct X3 : T { };

template<typename T>
struct X4 {
  template<typename U>
  void f(X2<sizeof(X3<U>().U::m)>);
};

void f(X4<X3<int> > x4i) {
  X2<sizeof(int)> x2;
  x4i.f<X2<sizeof(int)> >(x2);
}
