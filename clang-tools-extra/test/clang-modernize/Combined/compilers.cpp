// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=clang-2.9 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=CLANG-29 -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=clang-2.9 -override-macros %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=CLANG-29-OV-MACROS -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=clang-3.0 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=CLANG-30 -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=gcc-4.6 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=GCC-46 -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=gcc-4.7 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=GCC-47 -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=icc-13 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=ICC-13 -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=icc-14 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=ICC-14 -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=msvc-8 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=MSVC-8 -input-file=%t.cpp %s
//
// Test multiple compilers
// RUN: grep -Ev "// *[A-Z0-9-]+:" %s > %t.cpp
// RUN: clang-modernize -for-compilers=clang-3.0,gcc-4.6,gcc-4.7 %t.cpp -- -std=c++11
// RUN: FileCheck -check-prefix=MULTIPLE -input-file=%t.cpp %s
//
// Test unknown platform
// RUN: not clang-modernize -for-compilers=foo-10 %t.cpp -- -std=c++11
//
// Test when no transforms can be selected because the compiler lacks support of
// the needed C++11 features
// RUN: not clang-modernize -for-compilers=clang-2.0 %t.cpp -- -std=c++11

// Test add overrides
struct A {
  virtual A *clone() = 0;
};

#define LLVM_OVERRIDE override

struct B : A {
  virtual B *clone();
  // CLANG-29-OV-MACROS: virtual B *clone() LLVM_OVERRIDE;
  // CLANG-29: virtual B *clone();
  // CLANG-30: virtual B *clone() override;
  // GCC-46: virtual B *clone();
  // GCC-47: virtual B *clone() override;
  // ICC-13: virtual B *clone();
  // ICC-14: virtual B *clone() override;
  // MSVC-8: virtual B *clone() override;
  // MULTIPLE: virtual B *clone();
};
