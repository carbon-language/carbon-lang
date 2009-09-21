// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

struct Base1 {
  int member1;
  float member2;
};

struct Base2 {
  int member1;
  double member3;
  void memfun1(int);
};

struct Base3 : Base1, Base2 {
  void memfun1(float);
  void memfun1(double);
  void memfun2(int);
};

struct Derived : Base3 {
  int member4;
  int memfun3(int);
};

class Proxy {
public:
  Derived *operator->() const;
};

void test(const Proxy &p) {
  // CHECK-CC1: member4 : 0
  // CHECK-CC1: memfun3 : 0
  // CHECK-CC1: memfun1 : 1
  // CHECK-CC1: memfun1 : 1
  // CHECK-CC1: memfun2 : 1
  // CHECK-CC1: member1 : 2
  // CHECK-CC1: member1 : 2
  // CHECK-CC1: member2 : 2
  // CHECK-CC1: member3 : 2
  // CHECK-CC1: memfun1 : 2 (Hidden) : Base2::memfun1(<#int#>)
  p->