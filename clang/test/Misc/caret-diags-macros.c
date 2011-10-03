// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#define M1(x) x
#define M2 1;
void foo() {
  M1(
    M2);
  // CHECK: :7:{{[0-9]+}}: warning: expression result unused
  // CHECK: :4:{{[0-9]+}}: note: expanded from:
  // CHECK: :3:{{[0-9]+}}: note: expanded from:
}

#define A 1
#define B A
#define C B
void bar() {
  C;
  // CHECK: :17:3: warning: expression result unused
  // CHECK: :15:11: note: expanded from:
  // CHECK: :14:11: note: expanded from:
  // CHECK: :13:11: note: expanded from:
}

// rdar://7597492
#define sprintf(str, A, B) \
__builtin___sprintf_chk (str, 0, 42, A, B)

void baz(char *Msg) {
  sprintf(Msg,  "  sizeof FoooLib            : =%3u\n",   12LL);
}


// PR9279: comprehensive tests for multi-level macro back traces
#define macro_args1(x) x
#define macro_args2(x) macro_args1(x)
#define macro_args3(x) macro_args2(x)

#define macro_many_args1(x, y, z) y
#define macro_many_args2(x, y, z) macro_many_args1(x, y, z)
#define macro_many_args3(x, y, z) macro_many_args2(x, y, z)

void test() {
  macro_args3(1);
  // CHECK: {{.*}}:43:15: warning: expression result unused
  // Also check that the 'caret' printing agrees with the location here where
  // its easy to FileCheck.
  // CHECK-NEXT: macro_args3(1);
  // CHECK-NEXT: ~~~~~~~~~~~~^~
  // CHECK: {{.*}}:36:36: note: expanded from:
  // CHECK: {{.*}}:35:36: note: expanded from:
  // CHECK: {{.*}}:34:24: note: expanded from:

  macro_many_args3(
    1,
    2,
    3);
  // CHECK: {{.*}}:55:5: warning: expression result unused
  // CHECK: {{.*}}:40:55: note: expanded from:
  // CHECK: {{.*}}:39:55: note: expanded from:
  // CHECK: {{.*}}:38:35: note: expanded from:

  macro_many_args3(
    1,
    M2,
    3);
  // CHECK: {{.*}}:64:5: warning: expression result unused
  // CHECK: {{.*}}:4:12: note: expanded from:
  // CHECK: {{.*}}:40:55: note: expanded from:
  // CHECK: {{.*}}:39:55: note: expanded from:
  // CHECK: {{.*}}:38:35: note: expanded from:

  macro_many_args3(
    1,
    macro_args2(2),
    3);
  // CHECK: {{.*}}:74:17: warning: expression result unused
  // This caret location needs to be printed *inside* a different macro's
  // arguments.
  // CHECK-NEXT: macro_args2(2),
  // CHECK-NEXT: ~~~~~~~~~~~~^~~
  // CHECK: {{.*}}:35:36: note: expanded from:
  // CHECK: {{.*}}:34:24: note: expanded from:
  // CHECK: {{.*}}:40:55: note: expanded from:
  // CHECK: {{.*}}:39:55: note: expanded from:
  // CHECK: {{.*}}:38:35: note: expanded from:
}

#define variadic_args1(x, y, ...) y
#define variadic_args2(x, ...) variadic_args1(x, __VA_ARGS__)
#define variadic_args3(x, y, ...) variadic_args2(x, y, __VA_ARGS__)

void test2() {
  variadic_args3(1, 2, 3, 4);
  // CHECK: {{.*}}:93:21: warning: expression result unused
  // CHECK-NEXT: variadic_args3(1, 2, 3, 4);
  // CHECK-NEXT: ~~~~~~~~~~~~~~~~~~^~~~~~~~
  // CHECK: {{.*}}:90:53: note: expanded from:
  // CHECK: {{.*}}:89:50: note: expanded from:
  // CHECK: {{.*}}:88:35: note: expanded from:
}

#define variadic_pasting_args1(x, y, z) y
#define variadic_pasting_args2(x, ...) variadic_pasting_args1(x ## __VA_ARGS__)
#define variadic_pasting_args2a(x, y, ...) variadic_pasting_args1(x, y ## __VA_ARGS__)
#define variadic_pasting_args3(x, y, ...) variadic_pasting_args2(x, y, __VA_ARGS__)
#define variadic_pasting_args3a(x, y, ...) variadic_pasting_args2a(x, y, __VA_ARGS__)

void test3() {
  variadic_pasting_args3(1, 2, 3, 4);
  // CHECK: {{.*}}:109:32: warning: expression result unused
  // CHECK: {{.*}}:105:72: note: expanded from:
  // CHECK: {{.*}}:103:68: note: expanded from:
  // CHECK: {{.*}}:102:41: note: expanded from:

  variadic_pasting_args3a(1, 2, 3, 4);
  // CHECK: {{.*}}:115:30: warning: expression result unused
  // CHECK: {{.*}}:106:71: note: expanded from:
  // CHECK: {{.*}}:104:70: note: expanded from:
  // CHECK: {{.*}}:102:41: note: expanded from:
}
