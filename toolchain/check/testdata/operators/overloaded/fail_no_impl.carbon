// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl.carbon

package User;

class C {};

fn TestUnary(a: C) -> C {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Negate` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return -a;
  // CHECK:STDERR:          ^~
  // CHECK:STDERR:
  return -a;
}

fn TestBinary(a: C, b: C) -> C {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Add` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestRef(b: C) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssign` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a += b;
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.Inc` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   ++a;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  ++a;
}

// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TestUnary.type: type = fn_type @TestUnary [concrete]
// CHECK:STDOUT:   %TestUnary: %TestUnary.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestBinary.type: type = fn_type @TestBinary [concrete]
// CHECK:STDOUT:   %TestBinary: %TestBinary.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestRef.type: type = fn_type @TestRef [concrete]
// CHECK:STDOUT:   %TestRef: %TestRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .AddAssign = %Core.AddAssign
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestUnary = %TestUnary.decl
// CHECK:STDOUT:     .TestBinary = %TestBinary.decl
// CHECK:STDOUT:     .TestRef = %TestRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %TestUnary.decl: %TestUnary.type = fn_decl @TestUnary [concrete = constants.%TestUnary] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc15_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestBinary.decl: %TestBinary.type = fn_decl @TestBinary [concrete = constants.%TestBinary] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc23_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc23_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc23_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRef.decl: %TestRef.type = fn_decl @TestRef [concrete = constants.%TestRef] {
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnary(%a.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBinary(%a.param_patt: %C, %b.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRef(%b.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %.loc32_3.1: %C = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %.loc32_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc32_15.2: init %C = class_init (), %a.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc32_3.2: init %C = converted %.loc32_15.1, %.loc32_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %a.var, %.loc32_3.2
// CHECK:STDOUT:   %C.ref.loc32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref.loc37: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc42: ref %C = name_ref a, %a
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
