// RUN: tblgen %s | FileCheck %s
// XFAIL: vg_leak
def X1;

class C1<int N> {
  dag d = (X1 N);
}

def VAL1 : C1<13>;

// CHECK: def VAL1 {
// CHECK-NEXT: dag d = (X1 13)


def X2;

class yclass;
def Y2 : yclass;

class C2<yclass N> {
  dag d = (X2 N);
  dag e = (N X2);
}

def VAL2 : C2<Y2>;

// CHECK: def VAL2 {
// CHECK-NEXT: dag d = (X2 Y2)
// CHECK-NEXT: dag e = (Y2 X2)

