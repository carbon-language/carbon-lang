// XFAIL: *
//
// Without inline namespace:
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -replace-auto_ptr %t.cpp -- -I %S/Inputs std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
//
// With inline namespace:
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -replace-auto_ptr %t.cpp -- -I %S/Inputs std=c++11 \
// RUN:                                           -DUSE_INLINE_NAMESPACE=1
// RUN: FileCheck -input-file=%t.cpp %s

#include "memory_stub.h"

// Fail to modify when the template is never instantiated.
//
// This might not be an issue. If it's never used it doesn't really matter if
// it's changed or not. If it's a header and one of the source use it, then it
// will still be changed.
template <typename X>
void f() {
  std::auto_ptr<X> p;
  // CHECK: std::unique_ptr<X> p;
}

// Alias template could be replaced if a matcher existed.
template <typename T> using aaaaaaaa = auto_ptr<T>;
// CHECK: template <typename T> using aaaaaaaa = unique_ptr<T>;
