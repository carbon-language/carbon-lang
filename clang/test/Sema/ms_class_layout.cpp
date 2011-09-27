// RUN: %clang_cc1 -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s

#pragma pack(push, 8)

class B {
public:
  virtual void b(){}
  int b_field;
protected:
private:
};

class A : public B {
public:
  int a_field;
  virtual void a(){}
  char one;
protected:
private:
};

class D {
public:
  virtual void b(){}
  double a;
};

class C : public virtual A, 
          public D, public B {
public:
  double c1_field;
  int c2_field;
  double c3_field;
  int c4_field;
  virtual void foo(){}
  virtual void bar(){}
protected:
private:
};

struct BaseStruct
{
    BaseStruct(){}
    double v0;
    float v1;
    C fg;
};

struct DerivedStruct : public BaseStruct {
  int x;
};

struct G
{
    virtual ~G(){}
    int a;
    double b;
};

#pragma pack(pop)

// This needs only for building layouts. 
// Without this clang doesn`t dump record layouts.
int main() {
  // This avoid "Can't yet mangle constructors!" for MS ABI.
  C* c;
  c->foo();
  DerivedStruct* v;
  G* g;
  BaseStruct* u;
  return 0;
}

// CHECK:       0 | class D
// CHECK-NEXT:  0 |   (D vtable pointer)
// CHECK-NEXT:  8 |   double a

// CHECK-NEXT: sizeof=16, dsize=16, align=8
// CHECK-NEXT: nvsize=16, nvalign=8

// CHECK:       0 | class B
// CHECK-NEXT:  0 |   (B vtable pointer)
// CHECK-NEXT:  4 |   int b_field

// CHECK-NEXT: sizeof=8, dsize=8, align=4
// CHECK-NEXT: nvsize=8, nvalign=4

// CHECK:       0 | class A
// CHECK-NEXT:  0 |   class B (primary base)
// CHECK-NEXT:  0 |     (B vtable pointer)
// CHECK-NEXT:  4 |     int b_field
// CHECK-NEXT:  8 |   int a_field
// CHECK-NEXT: 12 |   char one

// CHECK-NEXT: sizeof=16, dsize=16, align=4
// CHECK-NEXT: nvsize=16, nvalign=4

// CHECK:       0 | class C
// CHECK-NEXT:  0 |   class D (primary base)
// CHECK-NEXT:  0 |     (D vtable pointer)
// CHECK-NEXT:  8 |     double a
// CHECK-NEXT: 16 |   class B (base)
// CHECK-NEXT: 16 |     (B vtable pointer)
// CHECK-NEXT: 20 |     int b_field
// CHECK-NEXT: 24 |   (C vbtable pointer)
// CHECK-NEXT: 32 |   double c1_field
// CHECK-NEXT: 40 |   int c2_field
// CHECK-NEXT: 48 |   double c3_field
// CHECK-NEXT: 56 |   int c4_field
// CHECK-NEXT: 64 |   class A (virtual base)
// CHECK-NEXT: 64 |     class B (primary base)
// CHECK-NEXT: 64 |       (B vtable pointer)
// CHECK-NEXT: 68 |       int b_field
// CHECK-NEXT: 72 |     int a_field
// CHECK-NEXT: 76 |     char one

// CHECK-NEXT: sizeof=80, dsize=80, align=8
// CHECK-NEXT: nvsize=80, nvalign=8
 
// CHECK:       0 | struct BaseStruct
// CHECK-NEXT:  0 |   double v0
// CHECK-NEXT:  8 |   float v1
// CHECK-NEXT: 16 |   class C fg
// CHECK-NEXT: 16 |     class D (primary base)
// CHECK-NEXT: 16 |       (D vtable pointer)
// CHECK-NEXT: 24 |       double a
// CHECK-NEXT: 32 |     class B (base)
// CHECK-NEXT: 32 |       (B vtable pointer)
// CHECK-NEXT: 36 |       int b_field
// CHECK-NEXT: 40 |     (C vbtable pointer)
// CHECK-NEXT: 48 |     double c1_field
// CHECK-NEXT: 56 |     int c2_field
// CHECK-NEXT: 64 |     double c3_field
// CHECK-NEXT: 72 |     int c4_field
// CHECK-NEXT: 80 |     class A (virtual base)
// CHECK-NEXT: 80 |       class B (primary base)
// CHECK-NEXT: 80 |         (B vtable pointer)
// CHECK-NEXT: 84 |         int b_field
// CHECK-NEXT: 88 |       int a_field
// CHECK-NEXT: 92 |       char one

// CHECK-NEXT: sizeof=80, dsize=80, align=8
// CHECK-NEXT: nvsize=80, nvalign=8

// CHECK: sizeof=96, dsize=96, align=8
// CHECK-NEXT: nvsize=96, nvalign=8

// CHECK:       0 | struct DerivedStruct
// CHECK-NEXT:  0 |   struct BaseStruct (base)
// CHECK-NEXT:  0 |     double v0
// CHECK-NEXT:  8 |     float v1
// CHECK-NEXT: 16 |     class C fg
// CHECK-NEXT: 16 |       class D (primary base)
// CHECK-NEXT: 16 |         (D vtable pointer)
// CHECK-NEXT: 24 |         double a
// CHECK-NEXT: 32 |       class B (base)
// CHECK-NEXT: 32 |         (B vtable pointer)
// CHECK-NEXT: 36 |         int b_field
// CHECK-NEXT: 40 |       (C vbtable pointer)
// CHECK-NEXT: 48 |       double c1_field
// CHECK-NEXT: 56 |       int c2_field
// CHECK-NEXT: 64 |       double c3_field
// CHECK-NEXT: 72 |       int c4_field
// CHECK-NEXT: 80 |       class A (virtual base)
// CHECK-NEXT: 80 |         class B (primary base)
// CHECK-NEXT: 80 |           (B vtable pointer)
// CHECK-NEXT: 84 |           int b_field
// CHECK-NEXT: 88 |         int a_field
// CHECK-NEXT: 92 |         char one
// CHECK-NEXT: sizeof=80, dsize=80, align=8
// CHECK-NEXT: nvsize=80, nvalign=8

// CHECK: 96 |   int x
// CHECK-NEXT: sizeof=104, dsize=104, align=8
// CHECK-NEXT: nvsize=104, nvalign=8
