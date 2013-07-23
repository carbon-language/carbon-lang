// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: not cpp11-migrate -format-style=FOO -use-auto %t.cpp -- -std=c++11
// RUN: not cpp11-migrate -format-style=/tmp/ -use-auto %t.cpp -- -std=c++11
// RUN: cpp11-migrate -format-style=LLVM -use-auto %t.cpp -- -std=c++11
// RUN: FileCheck --strict-whitespace -input-file=%t.cpp %s

class MyType012345678901234567890123456789 {};

int f() {
  MyType012345678901234567890123456789 *a =
      new MyType012345678901234567890123456789();
  // CHECK: {{^\ \ auto\ a\ \=\ new\ MyType012345678901234567890123456789\(\);}}

  delete a;
}
