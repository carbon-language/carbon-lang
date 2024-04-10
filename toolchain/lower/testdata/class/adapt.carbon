// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class PairOfInts {
  var a: i32;
  var b: i32;
}

class PairAdapter {
  adapt PairOfInts;
}

fn MakePair() -> PairAdapter {
  returned var v: PairAdapter;
  return var;
}

// CHECK:STDOUT: ; ModuleID = 'adapt.carbon'
// CHECK:STDOUT: source_filename = "adapt.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @MakePair(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
