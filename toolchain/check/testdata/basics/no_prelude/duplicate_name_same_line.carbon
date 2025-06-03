// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/duplicate_name_same_line.carbon

fn A() {
  if (true) {
    // This declares `n` in two different scopes, it's just showing the IR
    // behavior of having both on the same line.
    //@dump-sem-ir-begin
    var n: (); } else { var n: ();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc16_9: %pattern_type = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc16_5: %pattern_type = var_pattern %n.patt.loc16_9 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc16_5: ref %empty_tuple.type = var %n.var_patt.loc16_5
// CHECK:STDOUT:   %.loc16_13.1: type = splice_block %.loc16_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_13.3: type = converted %.loc16_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc16_9: ref %empty_tuple.type = bind_name n, %n.var.loc16_5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc16_29: %pattern_type = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc16_25: %pattern_type = var_pattern %n.patt.loc16_29 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc16_25: ref %empty_tuple.type = var %n.var_patt.loc16_25
// CHECK:STDOUT:   %.loc16_33.1: type = splice_block %.loc16_33.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16_33.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_33.3: type = converted %.loc16_33.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc16_29: ref %empty_tuple.type = bind_name n, %n.var.loc16_25
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
