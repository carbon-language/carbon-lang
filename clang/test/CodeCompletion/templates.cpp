namespace std {
  template<typename T>
  class allocator;
  
  template<typename T, typename Alloc = std::allocator<T> > class vector;
  template<typename Alloc> class vector<bool, Alloc>;
}

void f() {
  std::
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:10:8 %s -o - | FileCheck -check-prefix=CC1 %s
  // CHECK-CC1: allocator<<#typename T#>>
  // CHECK-CC1-NEXT: vector<<#typename T#>{#, <#typename Alloc#>#}>
  
  

