// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

namespace std {
  template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first, RandomAccessIterator last);
  
  template<class X, class Y>
  X* dyn_cast(Y *Val);
}

void f() {
  // CHECK-CC1: dyn_cast<<#class X#>>(<#Y *Val#>)
  // CHECK-CC1: sort(<#RandomAccessIterator first#>, <#RandomAccessIterator last#>)
  std::
  
