// Note: the run lines follow their respective tests, since line/column
// matter in this test.
void f(float x, float y);
void f(int i, int j, int k);
struct X { };
void f(X);
namespace N {
  struct Y { 
    Y(int = 0); 
    
    operator int() const;
  };
  void f(Y y, int);
}
typedef N::Y Y;
void f();

void test() {
  f(Y(), 0, 0);
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:19:9 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: f : 0 : f(<#struct N::Y y#>, <#int#>)
  // CHECK-NEXT-CC1: f : 0 : f(<#int i#>, <#int j#>, <#int k#>)
  // CHECK-NEXT-CC1: f : 0 : f(<#float x#>, <#float y#>)
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:19:13 %s -o - | FileCheck -check-prefix=CC2 %s &&
  // CHECK-NOT-CC2: f : 0 : f(<#struct N::Y y#>, <#int#>)
  // CHECK-CC2: f : 0 : f(<#int i#>, <#int j#>, <#int k#>)
  // CHECK-NEXT-CC2: f : 0 : f(<#float x#>, <#float y#>)
  // RUN: true
}
