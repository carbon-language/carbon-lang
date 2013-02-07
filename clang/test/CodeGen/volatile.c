// RUN: %clang_cc1 -emit-llvm < %s | FileCheck %s

// The number 28 comes from the current codegen for volatile loads;
// if this number changes, it's not necessarily something wrong, but
// something has changed to affect volatile load/store codegen

int S;
volatile int vS;

int* pS;
volatile int* pvS;

int A[10];
volatile int vA[10];

struct { int x; } F;
struct { volatile int x; } vF;

struct { int x; } F2;
volatile struct { int x; } vF2;
volatile struct { int x; } *vpF2;

struct { struct { int y; } x; } F3;
volatile struct { struct { int y; } x; } vF3;

struct { int x:3; } BF;
struct { volatile int x:3; } vBF;

typedef int v4si __attribute__ ((vector_size (16)));
v4si V;
volatile v4si vV;

typedef __attribute__(( ext_vector_type(4) )) int extv4;
extv4 VE;
volatile extv4 vVE;

volatile struct {int x;} aggFct(void);

typedef volatile int volatile_int;
volatile_int vtS;

int main() {
  int i;

  // load
  i=S;
  i=vS;
// CHECK: load volatile
  i=*pS;
  i=*pvS;
// CHECK: load volatile
  i=A[2];
  i=vA[2];
// CHECK: load volatile
  i=F.x;
  i=vF.x;
// CHECK: load volatile
  i=F2.x;
  i=vF2.x;
// CHECK: load volatile
  i=vpF2->x;
// CHECK: load volatile
  i=F3.x.y;
  i=vF3.x.y;
// CHECK: load volatile
  i=BF.x;
  i=vBF.x;
// CHECK: load volatile
  i=V[3];
  i=vV[3];
// CHECK: load volatile
  i=VE.yx[1];
  i=vVE.zy[1];
// CHECK: load volatile
  i = aggFct().x; // Note: not volatile
  i=vtS;
// CHECK: load volatile


  // store
  S=i;
  vS=i;
// CHECK: store volatile
  *pS=i;
  *pvS=i;
// CHECK: store volatile
  A[2]=i;
  vA[2]=i;
// CHECK: store volatile
  F.x=i;
  vF.x=i;
// CHECK: store volatile
  F2.x=i;
  vF2.x=i;
// CHECK: store volatile
  vpF2->x=i;
// CHECK: store volatile
  vF3.x.y=i;
// CHECK: store volatile
  BF.x=i;
  vBF.x=i;
// CHECK: store volatile
  V[3]=i;
  vV[3]=i;
// CHECK: store volatile
  vtS=i;
// CHECK: store volatile

  // other ops:
  ++S;
  ++vS;
// CHECK: load volatile
// CHECK: store volatile
  i+=S;
  i+=vS;
// CHECK: load volatile
  ++vtS;
// CHECK: load volatile
// CHECK: store volatile
  (void)vF2;
  // From vF2 to a temporary
// CHECK: call void @llvm.memcpy{{.*}} i1 true
  vF2 = vF2;
  // vF2 to itself
// CHECK: call void @llvm.memcpy{{.*}} i1 true
  vF2 = vF2 = vF2;
  // vF2 to itself twice
// CHECK: call void @llvm.memcpy{{.*}} i1 true
// CHECK: call void @llvm.memcpy{{.*}} i1 true
  vF2 = (vF2, vF2);
  // vF2 to a temporary, then vF2 to itself
// CHECK: call void @llvm.memcpy{{.*}} i1 true
// CHECK: call void @llvm.memcpy{{.*}} i1 true
}
