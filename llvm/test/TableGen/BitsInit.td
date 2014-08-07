
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
