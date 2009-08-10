// RUN: clang-cc -emit-llvm -triple i386-linux-gnu -o %t %s &&
// RUN: grep 't1.*noreturn' %t &&
// RUN: grep 't2.*nounwind' %t &&
// RUN: grep 'weak.*t3' %t &&
// RUN: grep 'hidden.*t4' %t &&
// RUN: grep 't5.*weak' %t &&
// RUN: grep 't6.*protected' %t &&
// RUN: grep 't7.*noreturn' %t &&
// RUN: grep 't7.*nounwind' %t &&
// RUN: grep 't9.*alias.*weak.*t8' %t &&
// RUN: grep '@t10().*section "SECT"' %t &&
// RUN: grep '@t11().*section "SECT"' %t &&
// RUN: grep '@t12 =.*section "SECT"' %t &&
// RUN: grep '@t13 =.*section "SECT"' %t &&
// RUN: grep '@t14.x =.*section "SECT"' %t &&
// RUN: grep 'declare extern_weak i32 @t15()' %t &&
// RUN: grep '@t16 = extern_weak global i32' %t &&

void t1() __attribute__((noreturn));
void t1() { while (1) {} }

void t2() __attribute__((nothrow));
void t2() {}

void t3() __attribute__((weak));
void t3() {}

void t4() __attribute__((visibility("hidden")));
void t4() {}

int t5 __attribute__((weak)) = 2;

int t6 __attribute__((visibility("protected")));

void t7() __attribute__((noreturn, nothrow));
void t7() { while (1) {} }

void __t8() {}
void t9() __attribute__((weak, alias("__t8")));

void t10(void) __attribute__((section("SECT")));
void t10(void) {}
void __attribute__((section("SECT"))) t11(void) {}

int t12 __attribute__((section("SECT")));
struct s0 { int x; };
struct s0 t13 __attribute__((section("SECT"))) = { 0 };

void t14(void) {
  static int x __attribute__((section("SECT"))) = 0;
}

int __attribute__((weak_import)) t15(void);
extern int t16 __attribute__((weak_import));
int t17() {
  return t15() + t16;
}

// RUN: grep '@t18 = global i[0-9]* 1, align .*' %t &&
extern int t18 __attribute__((weak_import));
int t18 = 1;

// RUN: grep 'define i[0-9]* @t19()' %t &&
extern int t19(void) __attribute__((weak_import));
int t19(void) {
  return 10;
}

// RUN: true
