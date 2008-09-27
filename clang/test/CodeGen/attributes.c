// RUN: clang -emit-llvm -o %t %s &&
// RUN: grep 't1.*noreturn' %t &&
// RUN: grep 't2.*nounwind' %t &&
// RUN: grep 'weak.*t3' %t &&
// RUN: grep 'hidden.*t4' %t &&
// RUN: grep 't5.*weak' %t &&
// RUN: grep 't6.*protected' %t &&
// RUN: grep 't7.*noreturn' %t &&
// RUN: grep 't7.*nounwind' %t &&
// RUN: grep 't9.*alias.*weak.*t8' %t

void t1() __attribute__((noreturn));
void t1() {}

void t2() __attribute__((nothrow));
void t2() {}

void t3() __attribute__((weak));
void t3() {}

void t4() __attribute__((visibility("hidden")));
void t4() {}

int t5 __attribute__((weak)) = 2;

int t6 __attribute__((visibility("protected")));

void t7() __attribute__((noreturn, nothrow));
void t7() {}

void __t8() {}
void t9() __attribute__((weak, alias("__t8")));
