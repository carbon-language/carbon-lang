// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -- > %t2.cpp
// RUN: FileCheck -input-file=%t2.cpp %s

namespace i {
}
// CHECK: warning: namespace not terminated with a closing comment
