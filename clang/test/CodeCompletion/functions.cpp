void f(int i, int j = 2, int k = 5);
void f(float x, float y...);
       
void test() {
  ::
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:5:5 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: f(<#int i#>{#, <#int j#>{#, <#int k#>#}#})
  // CHECK-CC1: f(<#float x#>, <#float y#><#, ...#>)
  // RUN: true
