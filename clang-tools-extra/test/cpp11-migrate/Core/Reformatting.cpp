// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: not cpp11-migrate -format-style=FOO -use-auto %t.cpp -- -std=c++11
// RUN: not cpp11-migrate -format-style=/tmp/ -use-auto %t.cpp -- -std=c++11
// RUN: cpp11-migrate -format-style=LLVM -replace-auto_ptr -loop-convert \
// RUN:                                  -use-auto -use-nullptr %t.cpp -- \
// RUN:                                  -std=c++11
// RUN: FileCheck --strict-whitespace -input-file=%t.cpp %s

#include <iostream>
#include <memory>
#include <vector>

void take_auto_ptrs(std::auto_ptr<int>, std::auto_ptr<int>);

void f_1() {
  std::auto_ptr<int> aaaaaaaa;
  std::auto_ptr<int> bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
  // CHECK:      {{^\ \ std::unique_ptr<int>\ aaaaaaaa;$}}
  // CHECK-NEXT: std::unique_ptr<int>
  // CHECK-NEXT: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;

  aaaaaaaa = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
  // CHECK:      aaaaaaaa =
  // CHECK-Next:     std::move(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);

  std::auto_ptr<int> cccccccccccccccccccccccccccc, dddddddddddddddddddddddddddd;
  // CHECK:      std::unique_ptr<int> cccccccccccccccccccccccccccc,
  // CHECK-NEXT:     dddddddddddddddddddddddddddd;

  take_auto_ptrs(cccccccccccccccccccccccccccc, dddddddddddddddddddddddddddd);
  // CHECK:      take_auto_ptrs(std::move(cccccccccccccccccccccccccccc),
  // CHECK-NEXT:                std::move(dddddddddddddddddddddddddddd));
}

// Test loop-convert (and potentially use-auto)
void f_2(const std::vector<int> &Vect) {
  for (std::vector<int>::const_iterator I = Vect.begin(), E = Vect.end();
       I != E; ++I)
    std::cout << *I << std::endl;
  // CHECK:      for (auto const &elem : Vect)
  // CHECK-NEXT:   std::cout << elem << std::endl;
}
