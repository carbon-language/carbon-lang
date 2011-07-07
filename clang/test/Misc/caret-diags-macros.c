// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#define M1(x) x
#define M2 1;
void foo() {
  M1(
    M2);
  // CHECK: :7:{{[0-9]+}}: warning: expression result unused
  // CHECK: :4:{{[0-9]+}}: note: instantiated from:
  // CHECK: :3:{{[0-9]+}}: note: instantiated from:
}

#define A 1
#define B A
#define C B
void bar() {
  C;
  // CHECK: :17:3: warning: expression result unused
  // CHECK: :15:11: note: instantiated from:
  // CHECK: :14:11: note: instantiated from:
  // CHECK: :13:11: note: instantiated from:
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
  // CHECK: {{.*}}:36:36: note: instantiated from:
  // CHECK: {{.*}}:35:36: note: instantiated from:
  // CHECK: {{.*}}:34:24: note: instantiated from:

  macro_many_args3(
    1,
    2,
    3);
  // CHECK: {{.*}}:55:5: warning: expression result unused
  // CHECK: {{.*}}:40:55: note: instantiated from:
  // CHECK: {{.*}}:39:55: note: instantiated from:
  // CHECK: {{.*}}:38:35: note: instantiated from:

  macro_many_args3(
    1,
    M2,
    3);
  // CHECK: {{.*}}:64:5: warning: expression result unused
  // CHECK: {{.*}}:4:12: note: instantiated from:
  // CHECK: {{.*}}:40:55: note: instantiated from:
  // CHECK: {{.*}}:39:55: note: instantiated from:
  // CHECK: {{.*}}:38:35: note: instantiated from:

  macro_many_args3(
    1,
    macro_args2(2),
    3);
  // CHECK: {{.*}}:74:17: warning: expression result unused
  // This caret location needs to be printed *inside* a different macro's
  // arguments.
  // CHECK-NEXT: macro_args2(2),
  // CHECK-NEXT: ~~~~~~~~~~~~^~~
  // CHECK: {{.*}}:35:36: note: instantiated from:
  // CHECK: {{.*}}:34:24: note: instantiated from:
  // CHECK: {{.*}}:40:55: note: instantiated from:
  // CHECK: {{.*}}:39:55: note: instantiated from:
  // CHECK: {{.*}}:38:35: note: instantiated from:
}
