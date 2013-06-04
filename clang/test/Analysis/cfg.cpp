// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCFG -std=c++11 %s 2>&1 | FileCheck %s

// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK: Succs (21): B2 B3 B4 B5 B6 B7 B8 B9
// CHECK: B10 B11 B12 B13 B14 B15 B16 B17 B18 B19
// CHECK: B20 B21 B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT: Preds (21): B2 B3 B4 B5 B6 B7 B8 B9
// CHECK-NEXT: B10 B11 B12 B13 B14 B15 B16 B17 B18 B19
// CHECK-NEXT: B20 B21 B1
void checkWrap(int i) {
  switch(i) {
    case 0: break;
    case 1: break;
    case 2: break;
    case 3: break;
    case 4: break;
    case 5: break;
    case 6: break;
    case 7: break;
    case 8: break;
    case 9: break;
    case 10: break;
    case 11: break;
    case 12: break;
    case 13: break;
    case 14: break;
    case 15: break;
    case 16: break;
    case 17: break;
    case 18: break;
    case 19: break;
  }
}

// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: int i;
// CHECK-NEXT:   2: int j;
// CHECK-NEXT:   3: 1
// CHECK-NEXT:   4: int k = 1;
// CHECK-NEXT:   5: int l;
// CHECK-NEXT:   6: 2
// CHECK-NEXT:   7: int m = 2;
// CHECK-NEXT: CXXConstructExpr
// CHECK-NEXT:   9: struct standalone myStandalone;
// CHECK-NEXT: CXXConstructExpr
// CHECK-NEXT:  11: struct <anonymous struct at {{.*}}> myAnon;
// CHECK-NEXT: CXXConstructExpr
// CHECK-NEXT:  13: struct named myNamed;
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
void checkDeclStmts() {
  int i, j;
  int k = 1, l, m = 2;

  struct standalone { int x, y; };
  struct standalone myStandalone;

  struct { int x, y; } myAnon;

  struct named { int x, y; } myNamed;

  static_assert(1, "abc");
}

// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: e
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, enum EmptyE)
// CHECK-NEXT:   3: [B1.2] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:   T: switch [B1.3]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
enum EmptyE {};
void F(EmptyE e) {
  switch (e) {}
}
