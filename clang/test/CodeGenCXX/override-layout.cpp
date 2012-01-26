// RUN: %clang_cc1 -fdump-record-layouts-simple %s 2> %t.layouts
// RUN: %clang_cc1 -fdump-record-layouts-simple %s > %t.before 2>&1
// RUN: %clang_cc1 -DPACKED= -DALIGNED16= -fdump-record-layouts-simple -foverride-record-layout=%t.layouts %s > %t.after 2>&1
// RUN: diff %t.before %t.after
// RUN: FileCheck %s < %t.after

// If not explicitly disabled, set PACKED to the packed attribute.
#ifndef PACKED
#  define PACKED __attribute__((packed))
#endif

struct Empty1 { };
struct Empty2 { };

// CHECK: Type: struct X0
struct X0 : public Empty1 { 
  int x[6] PACKED; 
};

// CHECK: Type: struct X1
struct X1 : public X0, public Empty2 { 
  char x[13]; 
  struct X0 y; 
} PACKED;

// CHECK: Type: struct X2
struct PACKED X2 :  public X1, public X0, public Empty1 {
  short x;
  int y;
};

// CHECK: Type: struct X3
struct PACKED X3 : virtual public X1, public X0 {
  short x;
  int y;
};

void use_structs() {
  struct X0 x0;
  x0.x[5] = sizeof(struct X0);

  struct X1 x1;
  x1.x[5] = sizeof(struct X1);

  struct X2 x2;
  x2.y = sizeof(struct X2);

  struct X3 x3;
  x3.y = sizeof(struct X3);
}
