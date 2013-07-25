// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: not cpp11-migrate -format-style=non_existent_file.yaml -use-auto %t.cpp -- -std=c++11
// RUN: touch %T/non_format_config.yaml
// RUN: not cpp11-migrate -format-style=%T/non_format_config.yaml -use-auto %t.cpp -- -std=c++11
// RUN: cpp11-migrate -format-style=LLVM -use-auto %t.cpp -- -std=c++11
// RUN: FileCheck --strict-whitespace -input-file=%t.cpp %s

class MyType012345678901234567890123456789 {};

int f() {
  MyType012345678901234567890123456789 *a =
      new MyType012345678901234567890123456789();
  // CHECK: {{^\ \ auto\ a\ \=\ new\ MyType012345678901234567890123456789\(\);}}

  delete a;
}
