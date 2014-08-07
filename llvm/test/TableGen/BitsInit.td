
// RUN: not llvm-tblgen %s 2>&1 > %t
// RUN: FileCheck %s < %t

def a {
  bits<2> opc = { 0, 1 };
  bits<2> opc2 = { 1, 0 };
  bits<1> opc3 = { 1 };
  bits<2> a = { opc, opc2 }; // error!
  bits<2> b = { opc{0}, opc2{0} };
  bits<2> c = { opc{1}, opc2{1} };
  bits<2> c = { opc3{0}, opc3 };
}

// CHECK: def a {
// CHECK:   bits<2> opc = { 0, 1 };
// CHECK:   bits<2> opc2 = { 1, 0 };
// CHECK:   bits<1> opc3 = { 1 };
// CHECK:   bits<2> a = { ?, ? };
// CHECK:   bits<2> b = { 1, 0 };
// CHECK:   bits<2> c = { 1, 1 };
// CHECK: }

def {
  bits<2> B1 = 0b011;  // bitfield is too small, reject
  bits<3> B2 = 0b011;  // ok

  bits<2> C1 = 0b111;  // bitfield is too small, reject
  bits<3> C2 = 0b111;  // ok

  bits<2> D1 = { 0, 0 }; // ok
  bits<2> D2 = { 0b00 }; // ok
  bits<3> D3 = { 0, 0 }; // type mismatch.  RHS doesn't have enough bits
  bits<3> D4 = { 0b00 }; // type mismatch.  RHS doesn't have enough bits
  bits<1> D5 = { 0 };    // ok
  bits<1> D6 = { 1 };    // ok
  bits<1> D7 = { 3 };    // type mismatch.  LHS doesn't have enough bits
  bits<2> D8 = { 0 };    // type mismatch.  RHS doesn't have enough bits

  bits<8> E;
  let E{7-0} = {0,0,1,?,?,?,?,?};
  let E{3-0} = 0b0010;
}

// CHECK: def {{.*}} {
// CHECK: bits<2> B1;
// CHECK: bits<3> B2 = { 0, 1, 1 };
// CHECK: bits<2> C1;
// CHECK: bits<3> C2 = { 1, 1, 1 };
// CHECK: bits<2> D1 = { 0, 0 };
// CHECK: bits<2> D2 = { 0, 0 };
// CHECK: bits<3> D3;
// CHECK: bits<3> D4;
// CHECK: bits<1> D5 = { 0 };
// CHECK: bits<1> D6 = { 1 };
// CHECK: bits<1> D7 = { ? };
// CHECK: bits<2> D8;
// CHECK: bits<8> E = { 0, 0, 1, ?, 0, 0, 1, 0 };
// CHECK: }
