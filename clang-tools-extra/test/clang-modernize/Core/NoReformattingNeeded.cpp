// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -format-style=LLVM -use-auto %t.cpp -- -std=c++11
// RUN: FileCheck --strict-whitespace -input-file=%t.cpp %s

class C {};

void f() { //
  C *a = new C();
  // CHECK: {{^\ \ auto\ a\ \=\ new\ C\(\);}}
}
