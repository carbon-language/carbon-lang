// RUN: clang-cc -arch i386 -emit-llvm -o %t %s &&

// RUN: grep '@g0_ext = extern_weak global i32' %t &&
extern int g0_ext __attribute__((weak));
// RUN: grep 'declare extern_weak i32 @g1_ext()' %t &&
extern int __attribute__((weak)) g1_ext (void);

// RUN: grep '@g0_common = weak global i32' %t &&
int g0_common __attribute__((weak));

// RUN: grep '@g0_def = weak global i32' %t &&
int g0_def __attribute__((weak)) = 52;
// RUN: grep 'define weak i32 @g1_def()' %t &&
int __attribute__((weak)) g1_def (void) {}

// Force _ext references
void f0() {
  int a = g0_ext;
  int b = g1_ext();
}

// RUN: true
