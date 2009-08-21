// RUN: clang-cc -fsyntax-only -verify %s

struct X {
  template<typename T> T& f0(T);
  
  void g0(int i, double d) {
    int &ir = f0(i);
    double &dr = f0(d);
  }
  
  template<typename T> T& f1(T);
  template<typename T, typename U> U& f1(T, U);
  
  void g1(int i, double d) {
    int &ir1 = f1(i);
    int &ir2 = f1(d, i);
    int &ir3 = f1(i, i);
  }
};

void test_X_f0(X x, int i, float f) {
  int &ir = x.f0(i);
  float &fr = x.f0(f);
}

void test_X_f1(X x, int i, float f) {
  int &ir1 = x.f1(i);
  int &ir2 = x.f1(f, i);
  int &ir3 = x.f1(i, i);
}
