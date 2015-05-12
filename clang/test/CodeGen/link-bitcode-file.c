// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -mlink-bitcode-file %t.bc -O3 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NO-BC %s
// RUN: not %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE -mlink-bitcode-file %t.bc -O3 -emit-llvm -o - %s 2>&1 | FileCheck -check-prefix=CHECK-BC %s
// Make sure we deal with failure to load the file.
// RUN: not %clang_cc1 -triple i386-pc-linux-gnu -mlink-bitcode-file no-such-file.bc \
// RUN:    -emit-llvm -o - %s 2>&1 | FileCheck -check-prefix=CHECK-NO-FILE %s

int f(void);

#ifdef BITCODE

// CHECK-BC: fatal error: cannot link module {{.*}}'f': symbol multiply defined
int f(void) {
  return 42;
}

#else

// CHECK-NO-BC-LABEL: define i32 @g
// CHECK-NO-BC: ret i32 42
int g(void) {
  return f();
}

// CHECK-NO-BC-LABEL: define i32 @f

#endif

// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file.bc'
