// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -fix --
// RUN: FileCheck -input-file=%t.cpp %s

namespace i {
}
// CHECK: } // namespace i

class A { A(int i); };
// CHECK: class A { explicit A(int i); };
