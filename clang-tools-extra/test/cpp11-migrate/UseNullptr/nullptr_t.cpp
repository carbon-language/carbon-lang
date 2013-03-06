// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -final-syntax-check -use-nullptr %t.cpp -- --std=c++11 -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *

namespace std { typedef decltype(nullptr) nullptr_t; }

// Just to make sure make_null() could have side effects.
void external();

std::nullptr_t make_null() { external(); return nullptr; }

void *call_make_null()
{
  return make_null();
  // CHECK: return make_null();
}
