// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.AddWith(C)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestRef(b: C) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(C)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
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
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %TestUnary.type: type = fn_type @TestUnary [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestUnary: %TestUnary.type = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %AddWith.type.e05: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.e05 = struct_value () [concrete]
// CHECK:STDOUT:   %TestBinary.type: type = fn_type @TestBinary [concrete]
// CHECK:STDOUT:   %TestBinary: %TestBinary.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestRef.type: type = fn_type @TestRef [concrete]
// CHECK:STDOUT:   %TestRef: %TestRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.fc6: type = generic_interface_type @AddAssignWith [concrete]
// CHECK:STDOUT:   %AddAssignWith.generic: %AddAssignWith.type.fc6 = struct_value () [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.10, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.906: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.10, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.impl_witness.906) [concrete]
// CHECK:STDOUT:   %.b80: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.5d7, @Op.10(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .AddWith = %Core.AddWith
// CHECK:STDOUT:     .AddAssignWith = %Core.AddAssignWith
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/operators/arithmetic, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.e05 = import_ref Core//prelude/operators/arithmetic, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.AddAssignWith: %AddAssignWith.type.fc6 = import_ref Core//prelude/operators/arithmetic, AddAssignWith, loaded [concrete = constants.%AddAssignWith.generic]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/destroy, loc15_29, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc18_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestBinary.decl: %TestBinary.type = fn_decl @TestBinary [concrete = constants.%TestBinary] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc26_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc26_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc26_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRef.decl: %TestRef.type = fn_decl @TestRef [concrete = constants.%TestRef] {
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc34: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnary(%a.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBinary(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRef(%b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c48 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C = var %a.var_patt
// CHECK:STDOUT:   %.loc35_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc35_15.2: init %C = class_init (), %a.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc35_3: init %C = converted %.loc35_15.1, %.loc35_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %a.var, %.loc35_3
// CHECK:STDOUT:   %C.ref.loc35: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref.loc40: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc45: ref %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0: %.b80 = impl_witness_access constants.%Destroy.impl_witness.906, element0 [concrete = constants.%Op.5d7]
// CHECK:STDOUT:   %bound_method.loc35_3.1: <bound method> = bound_method %a.var, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.10(constants.%C) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_3.2: <bound method> = bound_method %a.var, %specific_fn
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %a.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc35_3.2(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
