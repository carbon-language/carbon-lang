// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/eq.carbon

// --- user.carbon

package User;

class C {};

impl C as Core.EqWith(C) {
  fn Equal[self: C](other: C) -> bool;
  fn NotEqual[self: C](other: C) -> bool;
}

fn TestEqual(a: C, b: C) -> bool {
  return a == b;
}

fn TestNotEqual(a: C, b: C) -> bool {
  return a != b;
}

// --- fail_no_impl.carbon

package FailNoImpl;

class D {};

fn TestEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.EqWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestNotEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.EqWith(D)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a != b;
}

// --- fail_no_impl_for_args.carbon

package FailNoImplForArgs;

class C {};
class D {};

impl C as Core.EqWith(C) {
  fn Equal[self: C](other: C) -> bool;
  fn NotEqual[self: C](other: C) -> bool;
}

fn TestRhsBad(a: C, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.EqWith(D)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestLhsBad(a: D, b: C) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.EqWith(C)` in type `D` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a != b;
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %EqWith.type.c2e: type = generic_interface_type @EqWith [concrete]
// CHECK:STDOUT:   %EqWith.generic: %EqWith.type.c2e = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EqWith.type.04e: type = facet_type <@EqWith, @EqWith(%C)> [concrete]
// CHECK:STDOUT:   %Equal.type.170: type = fn_type @Equal.1, @EqWith(%C) [concrete]
// CHECK:STDOUT:   %NotEqual.type.694: type = fn_type @NotEqual.1, @EqWith(%C) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness file.%EqWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.type.86d: type = fn_type @Equal.loc7 [concrete]
// CHECK:STDOUT:   %Equal.fdb: %Equal.type.86d = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.0f4: type = fn_type @NotEqual.loc8 [concrete]
// CHECK:STDOUT:   %NotEqual.133: %NotEqual.type.0f4 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.04e = facet_value %C, (%EqWith.impl_witness) [concrete]
// CHECK:STDOUT:   %TestEqual.type: type = fn_type @TestEqual [concrete]
// CHECK:STDOUT:   %TestEqual: %TestEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.6b4: type = fn_type_with_self_type %Equal.type.170, %EqWith.facet [concrete]
// CHECK:STDOUT:   %TestNotEqual.type: type = fn_type @TestNotEqual [concrete]
// CHECK:STDOUT:   %TestNotEqual: %TestNotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.3b7: type = fn_type_with_self_type %NotEqual.type.694, %EqWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .EqWith = %Core.EqWith
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.EqWith: %EqWith.type.c2e = import_ref Core//prelude/operators/comparison, EqWith, loaded [concrete = constants.%EqWith.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual.decl
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @EqWith.impl.cac [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc6_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %EqWith.ref: %EqWith.type.c2e = name_ref EqWith, imports.%Core.EqWith [concrete = constants.%EqWith.generic]
// CHECK:STDOUT:     %C.ref.loc6_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %EqWith.type: type = facet_type <@EqWith, @EqWith(constants.%C)> [concrete = constants.%EqWith.type.04e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EqWith.impl_witness_table = impl_witness_table (@EqWith.impl.cac.%Equal.decl, @EqWith.impl.cac.%NotEqual.decl), @EqWith.impl.cac [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness %EqWith.impl_witness_table [concrete = constants.%EqWith.impl_witness]
// CHECK:STDOUT:   %TestEqual.decl: %TestEqual.type = fn_decl @TestEqual [concrete = constants.%TestEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc11_29.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc11_29.2: type = converted %bool.make_type, %.loc11_29.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc11_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc11_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestNotEqual.decl: %TestNotEqual.type = fn_decl @TestNotEqual [concrete = constants.%TestNotEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %bool.make_type, %.loc15_32.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc15_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc15_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @EqWith.impl.cac: %C.ref.loc6_6 as %EqWith.type {
// CHECK:STDOUT:   %Equal.decl: %Equal.type.86d = fn_decl @Equal.loc7 [concrete = constants.%Equal.fdb] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc7_34.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc7_34.2: type = converted %bool.make_type, %.loc7_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc7_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc7_28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: %NotEqual.type.0f4 = fn_decl @NotEqual.loc8 [concrete = constants.%NotEqual.133] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc8_37.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc8_37.2: type = converted %bool.make_type, %.loc8_37.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc8_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
// CHECK:STDOUT:   witness = file.%EqWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.loc7(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.loc8(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a.param: %C, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.6b4 = impl_witness_access constants.%EqWith.impl_witness, element0 [concrete = constants.%Equal.fdb]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Equal.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc12_16.1: bool = value_of_initializer %Equal.call
// CHECK:STDOUT:   %.loc12_16.2: bool = converted %Equal.call, %.loc12_16.1
// CHECK:STDOUT:   return %.loc12_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a.param: %C, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.3b7 = impl_witness_access constants.%EqWith.impl_witness, element1 [concrete = constants.%NotEqual.133]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %NotEqual.call: init bool = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_16.1: bool = value_of_initializer %NotEqual.call
// CHECK:STDOUT:   %.loc16_16.2: bool = converted %NotEqual.call, %.loc16_16.1
// CHECK:STDOUT:   return %.loc16_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %TestEqual.type: type = fn_type @TestEqual [concrete]
// CHECK:STDOUT:   %TestEqual: %TestEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.type.c2e: type = generic_interface_type @EqWith [concrete]
// CHECK:STDOUT:   %EqWith.generic: %EqWith.type.c2e = struct_value () [concrete]
// CHECK:STDOUT:   %TestNotEqual.type: type = fn_type @TestNotEqual [concrete]
// CHECK:STDOUT:   %TestNotEqual: %TestNotEqual.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .EqWith = %Core.EqWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.EqWith: %EqWith.type.c2e = import_ref Core//prelude/operators/comparison, EqWith, loaded [concrete = constants.%EqWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual.decl
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %TestEqual.decl: %TestEqual.type = fn_decl @TestEqual [concrete = constants.%TestEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %bool.make_type, %.loc6_29.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc6_17: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc6_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestNotEqual.decl: %TestNotEqual.type = fn_decl @TestNotEqual [concrete = constants.%TestNotEqual] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_32.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc14_32.2: type = converted %bool.make_type, %.loc14_32.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc14_20: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc14_26: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a.param: %D, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a.param: %D, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl_for_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %EqWith.type.c2e: type = generic_interface_type @EqWith [concrete]
// CHECK:STDOUT:   %EqWith.generic: %EqWith.type.c2e = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EqWith.type.04e: type = facet_type <@EqWith, @EqWith(%C)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness file.%EqWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.type.86d: type = fn_type @Equal.loc8 [concrete]
// CHECK:STDOUT:   %Equal.fdb: %Equal.type.86d = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.0f4: type = fn_type @NotEqual.loc9 [concrete]
// CHECK:STDOUT:   %NotEqual.133: %NotEqual.type.0f4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %TestRhsBad.type: type = fn_type @TestRhsBad [concrete]
// CHECK:STDOUT:   %TestRhsBad: %TestRhsBad.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestLhsBad.type: type = fn_type @TestLhsBad [concrete]
// CHECK:STDOUT:   %TestLhsBad: %TestLhsBad.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .EqWith = %Core.EqWith
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.EqWith: %EqWith.type.c2e = import_ref Core//prelude/operators/comparison, EqWith, loaded [concrete = constants.%EqWith.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestRhsBad = %TestRhsBad.decl
// CHECK:STDOUT:     .TestLhsBad = %TestLhsBad.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @EqWith.impl.cac [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc7_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %EqWith.ref: %EqWith.type.c2e = name_ref EqWith, imports.%Core.EqWith [concrete = constants.%EqWith.generic]
// CHECK:STDOUT:     %C.ref.loc7_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %EqWith.type: type = facet_type <@EqWith, @EqWith(constants.%C)> [concrete = constants.%EqWith.type.04e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EqWith.impl_witness_table = impl_witness_table (@EqWith.impl.cac.%Equal.decl, @EqWith.impl.cac.%NotEqual.decl), @EqWith.impl.cac [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness %EqWith.impl_witness_table [concrete = constants.%EqWith.impl_witness]
// CHECK:STDOUT:   %TestRhsBad.decl: %TestRhsBad.type = fn_decl @TestRhsBad [concrete = constants.%TestRhsBad] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc12_30.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc12_30.2: type = converted %bool.make_type, %.loc12_30.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLhsBad.decl: %TestLhsBad.type = fn_decl @TestLhsBad [concrete = constants.%TestLhsBad] {
// CHECK:STDOUT:     %a.patt: %pattern_type.510 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.510 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc20_30.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc20_30.2: type = converted %bool.make_type, %.loc20_30.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @EqWith.impl.cac: %C.ref.loc7_6 as %EqWith.type {
// CHECK:STDOUT:   %Equal.decl: %Equal.type.86d = fn_decl @Equal.loc8 [concrete = constants.%Equal.fdb] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc8_34.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %bool.make_type, %.loc8_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc8_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: %NotEqual.type.0f4 = fn_decl @NotEqual.loc9 [concrete = constants.%NotEqual.133] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc9_37.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc9_37.2: type = converted %bool.make_type, %.loc9_37.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc9_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
// CHECK:STDOUT:   witness = file.%EqWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.loc8(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.loc9(%self.param: %C, %other.param: %C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRhsBad(%a.param: %C, %b.param: %D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLhsBad(%a.param: %D, %b.param: %C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
