// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o %t %s
// RUN: grep 'declare i32 @f0() readnone$' %t
// RUN: grep 'declare i32 @f1() readonly$' %t
// RUN: grep 'declare void @f2(.* noalias sret)$' %t
// RUN: grep 'declare void @f3(.* noalias sret)$' %t
// RUN: grep 'declare void @f4(.* byval)$' %t
// RUN: grep 'declare void @f5(.* byval)$' %t
// PR3835

typedef int T0;
typedef struct { int a[16]; } T1;

T0 __attribute__((const)) f0(void);
T0 __attribute__((pure)) f1(void);
T1 __attribute__((const)) f2(void);
T1 __attribute__((pure)) f3(void);
void __attribute__((const)) f4(T1 a);
void __attribute__((pure)) f5(T1 a);

void *ps[] = { f0, f1, f2, f3, f4, f5 };
