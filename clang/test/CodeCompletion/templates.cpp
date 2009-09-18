// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

namespace std {
  template<typename T>
  class allocator;
  
  template<typename T, typename Alloc = std::allocator<T> >
  class vector;
}

void f() {
  // CHECK-CC1: allocator<<#typename T#>>
  // CHECK-CC1: vector<<#typename T#>{#, <#typename Alloc#>#}>
  std::


