// RUN: llvm-tblgen %s | FileCheck %s

// Check that a large integer is not truncated to a small bit sequence.
//
// CHECK: bits<2> X = { !cast<bits<2>>(5){1}, !cast<bits<2>>(5){0} };

def {
  bits<2> X = 5;  // bitfield is too small, reject
}
