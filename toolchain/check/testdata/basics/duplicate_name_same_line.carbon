// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/duplicate_name_same_line.carbon

fn A() {
  if (true) {
    // This declares `n` in two different scopes, it's just showing the IR
    // behavior of having both on the same line.
    //@dump-sem-ir-begin
    var unused n: (); } else { var unused n: ();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_16: %pattern_type.cb1 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc18_5: %pattern_type.cb1 = var_pattern %n.patt.loc18_16 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc18_5: ref %empty_tuple.type = var %n.var_patt.loc18_5
// CHECK:STDOUT:   %.loc18_20.1: type = splice_block %.loc18_20.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_20.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_20.3: type = converted %.loc18_20.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_16: ref %empty_tuple.type = ref_binding n, %n.var.loc18_5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_43: %pattern_type.cb1 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc18_32: %pattern_type.cb1 = var_pattern %n.patt.loc18_43 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc18_32: ref %empty_tuple.type = var %n.var_patt.loc18_32
// CHECK:STDOUT:   %.loc18_47.1: type = splice_block %.loc18_47.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_47.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_47.3: type = converted %.loc18_47.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_43: ref %empty_tuple.type = ref_binding n, %n.var.loc18_32
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc18_32: <bound method> = bound_method %n.var.loc18_32, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc18_32: init %empty_tuple.type = call %Destroy.Op.bound.loc18_32(%n.var.loc18_32)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18_5: <bound method> = bound_method %n.var.loc18_5, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc18_5: init %empty_tuple.type = call %Destroy.Op.bound.loc18_5(%n.var.loc18_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
