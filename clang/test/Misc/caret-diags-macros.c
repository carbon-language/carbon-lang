// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#define M1(x) x
#define M2 1;
void foo() {
  M1(
    M2);
  // CHECK: {{.*}}:6:{{[0-9]+}}: warning: expression result unused
  // CHECK: {{.*}}:7:{{[0-9]+}}: note: instantiated from:
  // CHECK: {{.*}}:4:{{[0-9]+}}: note: instantiated from:
}

#define A 1
#define B A
#define C B
void bar() {
  C;
  // CHECK: {{.*}}:17:{{[0-9]+}}: warning: expression result unused
  // CHECK: {{.*}}:15:{{[0-9]+}}: note: instantiated from:
  // CHECK: {{.*}}:14:{{[0-9]+}}: note: instantiated from:
  // CHECK: {{.*}}:13:{{[0-9]+}}: note: instantiated from:
}


// rdar://7597492
#define sprintf(str, A, B) \
__builtin___sprintf_chk (str, 0, 42, A, B)

void baz(char *Msg) {
  sprintf(Msg,  "  sizeof FoooLib            : =%3u\n",   12LL);
}

