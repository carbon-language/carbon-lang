// Test is line- and column-sensitive. Run lines are below.

struct X {
  X();
  X(int);
  X(int, int);
  X(const X&);
};

X getX(int value) { 
  switch (value) {
  case 1: return X(value);
  case 2: return X(value, value);
  case 3: return (X)value;
  default: break;
  }
  return X();
}

struct Y {
  int member;

  X getX();
};

X Y::getX() {
  return member;
}

struct YDerived : Y {
  X getAnotherX() { return member; }
};

void test() {
  X foo;
}

// RUN: c-index-test -cursor-at=%s:12:20 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// RUN: c-index-test -cursor-at=%s:13:21 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// RUN: c-index-test -cursor-at=%s:13:28 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// RUN: c-index-test -cursor-at=%s:14:23 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// CHECK-VALUE-REF: DeclRefExpr=value:10:12

// RUN: c-index-test -cursor-at=%s:12:18 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR1 %s
// RUN: c-index-test -cursor-at=%s:13:18 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR2 %s
// RUN: c-index-test -cursor-at=%s:14:19 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR1 %s
// RUN: c-index-test -cursor-at=%s:17:10 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR3 %s
// CHECK-TYPE-REF: TypeRef=struct X:3:8
// CHECK-CONSTRUCTOR1: CallExpr=X:5:3
// CHECK-CONSTRUCTOR2: CallExpr=X:6:3
// CHECK-CONSTRUCTOR3: CallExpr=X:4:3

// RUN: c-index-test -cursor-at=%s:23:3 %s | FileCheck -check-prefix=CHECK-RETTYPE %s
// RUN: c-index-test -cursor-at=%s:26:1 %s | FileCheck -check-prefix=CHECK-RETTYPE %s
// CHECK-RETTYPE: TypeRef=struct X:3:8

// RUN: c-index-test -cursor-at=%s:23:7 %s | FileCheck -check-prefix=CHECK-MEMFUNC-DECL %s
// CHECK-MEMFUNC-DECL: CXXMethod=getX:23:5
// RUN: c-index-test -cursor-at=%s:26:7 %s | FileCheck -check-prefix=CHECK-MEMFUNC-DEF %s
// CHECK-MEMFUNC-DEF: CXXMethod=getX:26:6

// RUN: c-index-test -cursor-at=%s:26:3 %s | FileCheck -check-prefix=CHECK-TYPEREF-Y %s
// CHECK-TYPEREF-Y: TypeRef=struct Y:20:8

// RUN: c-index-test -cursor-at=%s:27:10 %s | FileCheck -check-prefix=CHECK-IMPLICIT-MEMREF %s
// RUN: c-index-test -cursor-at=%s:31:28 %s | FileCheck -check-prefix=CHECK-IMPLICIT-MEMREF %s
// CHECK-IMPLICIT-MEMREF: MemberRefExpr=member:21:7

// RUN: c-index-test -cursor-at=%s:35:5 %s | FileCheck -check-prefix=CHECK-DECL %s
// CHECK-DECL: VarDecl=foo:35:5
