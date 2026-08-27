// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/member_access.carbon

// --- test.carbon

fn F() {
  var a: auto = s . is_prime [ n ] ;
  var b: auto = Core . Range ( 1000 ) ;
  var c: auto = Sieve . Make ( ) ;
  var d: auto = m [ i ] [ j ] ;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var a: auto = s.is_prime[n];
// CHECK:STDOUT:   var b: auto = Core.Range(1000);
// CHECK:STDOUT:   var c: auto = Sieve.Make();
// CHECK:STDOUT:   var d: auto = m[i][j];
// CHECK:STDOUT: }
