// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/raw_identifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/raw_identifier.carbon

fn A(n: ()) -> () {
  return r#n;
}

fn B(r#n: ()) -> () {
  return n;
}

fn C(r#if: ()) -> () {
  return r#if;
}

// CHECK:STDOUT: --- raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %n.patt: %empty_tuple.type = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %empty_tuple.type = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_17.2: type = converted %.loc11_17.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc11_10.1: type = splice_block %.loc11_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_10.3: type = converted %.loc11_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %empty_tuple.type = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %n.patt: %empty_tuple.type = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %empty_tuple.type = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_19.2: type = converted %.loc15_19.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc15_12.1: type = splice_block %.loc15_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_12.3: type = converted %.loc15_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %empty_tuple.type = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %if.patt: %empty_tuple.type = binding_pattern r#if
// CHECK:STDOUT:     %if.param_patt: %empty_tuple.type = value_param_pattern %if.patt, call_param0
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_20.2: type = converted %.loc19_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %if.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc19_13.1: type = splice_block %.loc19_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc19_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc19_13.3: type = converted %.loc19_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %if: %empty_tuple.type = bind_name r#if, %if.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%if.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %if.ref: %empty_tuple.type = name_ref r#if, %if
// CHECK:STDOUT:   return %if.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
