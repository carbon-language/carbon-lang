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
    var n: (); } else { var n: ();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.a63: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ea3: %T.as.Destroy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_9: %pattern_type.cb1 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc18_5: %pattern_type.cb1 = var_pattern %n.patt.loc18_9 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc18_5: ref %empty_tuple.type = var %n.var_patt.loc18_5
// CHECK:STDOUT:   %.loc18_13.1: type = splice_block %.loc18_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_13.3: type = converted %.loc18_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_9: ref %empty_tuple.type = bind_name n, %n.var.loc18_5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_29: %pattern_type.cb1 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc18_25: %pattern_type.cb1 = var_pattern %n.patt.loc18_29 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc18_25: ref %empty_tuple.type = var %n.var_patt.loc18_25
// CHECK:STDOUT:   %.loc18_33.1: type = splice_block %.loc18_33.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_33.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_33.3: type = converted %.loc18_33.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_29: ref %empty_tuple.type = bind_name n, %n.var.loc18_25
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18_25: <bound method> = bound_method %n.var.loc18_25, constants.%T.as.Destroy.impl.Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_25: <bound method> = bound_method %n.var.loc18_25, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc18_25: %ptr.843 = addr_of %n.var.loc18_25
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18_25: init %empty_tuple.type = call %bound_method.loc18_25(%addr.loc18_25)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18_5: <bound method> = bound_method %n.var.loc18_5, constants.%T.as.Destroy.impl.Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_5: <bound method> = bound_method %n.var.loc18_5, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc18_5: %ptr.843 = addr_of %n.var.loc18_5
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18_5: init %empty_tuple.type = call %bound_method.loc18_5(%addr.loc18_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
