// RUN: mkdir -p %t.h.gch
// RUN: %clang -x c-header %S/pch-dir.h -o %t.h.gch/c.gch 
// RUN: %clang -x c++-header %S/pch-dir.h -o %t.h.gch/cpp.gch 
// RUN: %clang -include %t.h -fsyntax-only %s -Xclang -print-stats 2> %t.clog
// RUN: FileCheck -check-prefix=C %s < %t.clog
// RUN: %clang -x c++ -include %t.h -fsyntax-only %s -Xclang -print-stats 2> %t.cpplog
// RUN: FileCheck -check-prefix=CPP %s < %t.cpplog
// RUN: not %clang -x c++ -std=c++11 -include %t.h -fsyntax-only %s 2> %t.cpp11log
// RUN: FileCheck -check-prefix=CPP11 %s < %t.cpp11log


int get() {
#ifdef __cplusplus
  // CHECK-CPP: .h.gch/cpp.gch
  return i;
#else
  // CHECK-C: .h.gch/c.gch
  return j;
#endif
}

// CHECK-CPP11: no suitable precompiled header file found in directory
