// RUN: clang -emit-llvm < %s | grep 't1.*noreturn' &&
void t1() __attribute__((noreturn));
void t1() {}

// RUN: clang -emit-llvm < %s | grep 't2.*nounwind' &&
void t2() __attribute__((nothrow));
void t2() {}

// RUN: clang -emit-llvm < %s | grep 'weak.*t3' &&
void t3() __attribute__((weak));
void t3() {}

// RUN: clang -emit-llvm < %s | grep 'hidden.*t4' &&
void t4() __attribute__((visibility("hidden")));
void t4() {}

// RUN: clang -emit-llvm < %s | grep 't5.*weak' &&
int t5 __attribute__((weak)) = 2;

// RUN: clang -emit-llvm < %s | grep 't6.*protected' &&
int t6 __attribute__((visibility("protected")));

// RUN: clang -emit-llvm < %s | grep 't7.*noreturn' &&
// RUN: clang -emit-llvm < %s | grep 't7.*nounwind' &&
void t7() __attribute__((noreturn, nothrow));
void t7() {}

// RUN: clang -emit-llvm < %s | grep 't9.*alias.*weak.*t8'
void __t8() {}
void t9() __attribute__((weak, alias("__t8")));
