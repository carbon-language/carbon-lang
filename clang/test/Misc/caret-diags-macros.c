// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#define M1(x) x
#define M2 1;
void foo() {
  M1(
    M2);
  // CHECK: :6:3: warning: expression result unused
  // CHECK: :7:5: note: instantiated from:
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



// PR9279 - Notes shouldn't print 'instantiated from' notes recursively.
#define N1(x) int arr[x]
#define N2(x) N1(x)
#define N3(x) N2(x)
N3(-1);

// CHECK: :39:1: error: 'arr' declared as an array with a negative size
// CHECK: :38:15: note: instantiated from:
// CHECK: :37:15: note: instantiated from:
// CHECK: :39:1: note: instantiated from:
