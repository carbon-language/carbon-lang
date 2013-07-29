// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -fix -checks=llvm.* --
// RUN: FileCheck -input-file=%t.cpp %s

namespace i {
}
// CHECK: } // namespace i

class A { A(int i); }; // Not fixing this, because the check is in google-.
// CHECK: class A { A(int i); };
