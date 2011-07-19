// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -x c++ %s 2> %t  || true
// RUN: FileCheck %s < %t
// PR5941
// END.

/* Test fixits for * and & mismatch in function arguments.
 * Since fixits are on the notes, they cannot be applied automatically. */

typedef int intTy;
typedef int intTy2;

void f0(int *a);
void f1(double *a);
void f1(intTy &a);

void f2(intTy2 *a) {
// CHECK: error: no matching function for call to 'f1
// CHECK: dereference the argument with *
// CHECK: void f1(intTy &a);
// CHECK: fix-it{{.*}}*(
// CHECK-NEXT: fix-it{{.*}})
// CHECK: void f1(double *a);
  f1(a + 1);

// This call cannot be fixed since without resulting in null pointer dereference.
// CHECK: error: no matching function for call to 'f1
// CHECK-NOT: take the address of the argument with &
// CHECK-NOT: fix-it
  f1((int *)0);
}

void f3(int &a) {
// CHECK: error: no matching function for call to 'f0
// CHECK: fix-it{{.*}}&
 f0(a);
}


void m(int *a, const int *b); // match 2
void m(double *a, int *b); // no match
void m(int *a, double *b); // no match
void m(intTy &a, int *b); // match 1

void mcaller(intTy2 a, int b) {
// CHECK: error: no matching function for call to 'm
// CHECK: take the address of the argument with &
// CHECK: fix-it{{.*}}&
// CHECK: take the address of the argument with &
// CHECK: fix-it{{.*}}&
// CHECK: fix-it{{.*}}&
  m(a, b);

// This call cannot be fixed because (a + 1) is not an l-value.
// CHECK: error: no matching function for call to 'm
// CHECK-NOT: fix-it
  m(a + 1, b);
}

// Test derived to base conversions.
struct A {
  int xx;
};

struct B : public A {
  double y;
};

bool br(A &a);
bool bp(A *a);
bool dv(B b);

void dbcaller(A *ptra, B *ptrb) {
  B b;

// CHECK: error: no matching function for call to 'br
// CHECK: fix-it{{.*}}*
  br(ptrb); // good
// CHECK: error: no matching function for call to 'bp
// CHECK: fix-it{{.*}}&
  bp(b); // good

// CHECK: error: no matching function for call to 'dv
// CHECK-NOT: fix-it
  dv(ptra); // bad: base to derived
}

// CHECK: errors generated
