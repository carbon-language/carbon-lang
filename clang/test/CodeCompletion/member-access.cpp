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
  p->
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:29:6 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: member1 : 0 : [#Base1::#]member1
  // CHECK-CC1: member1 : 0 : [#Base2::#]member1
  // CHECK-CC1: member2 : 0 : [#Base1::#]member2
  // CHECK-CC1: member3 : 0
  // CHECK-CC1: member4 : 0
  // CHECK-CC1: memfun1 : 0 : [#Base3::#]memfun1(<#float#>)
  // CHECK-CC1: memfun1 : 0 : [#Base3::#]memfun1(<#double#>)
  // CHECK-CC1: memfun2 : 0 : [#Base3::#]memfun2(<#int#>)
  // CHECK-CC1: memfun3 : 0 : memfun3(<#int#>)
  // CHECK-CC1: memfun1 : 0 (Hidden) : Base2::memfun1(<#int#>)
  // RUN: true
  
