// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true
void f(float x, float y);
void f(int i, int j, int k);
struct X { };
void f(X);
namespace N {
  struct Y { 
    Y(int = 0); 
    
    operator int() const;
  };
  void f(Y y);
}
typedef N::Y Y;
void f();

void test() {
  // CHECK-CC1: f : 0 : f(<#struct N::Y y#>)
  // CHECK-NEXT-CC1: f : 0 : f(<#int i#>, <#int j#>, <#int k#>)
  // CHECK-NEXT-CC1: f : 0 : f(<#float x#>, <#float y#>)
  f(Y(),
