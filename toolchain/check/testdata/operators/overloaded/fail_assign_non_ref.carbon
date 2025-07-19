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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_assign_non_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_assign_non_ref.carbon

package User;

class C {};

impl C as Core.Inc {
  fn Op[addr self: C*]();
}
impl C as Core.AddAssignWith(C) {
  fn Op[addr self: C*](other: C);
}

fn TestIncNonRef(a: C) {
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+7]]:5: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   ++a;
  // CHECK:STDERR:     ^
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-10]]:9: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Op[addr self: C*]();
  // CHECK:STDERR:         ^~~~~~~~~~~~~
  // CHECK:STDERR:
  ++a;
}

fn TestAddAssignNonRef(a: C, b: C) {
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-18]]:9: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Op[addr self: C*](other: C);
  // CHECK:STDERR:         ^~~~~~~~~~~~~
  // CHECK:STDERR:
  a += b;
}

// CHECK:STDOUT: --- fail_assign_non_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Op.type.e3a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness file.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %Op.type.73a: type = fn_type @Op.loc20 [concrete]
// CHECK:STDOUT:   %Op.0c9: %Op.type.73a = struct_value () [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %C, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.fc6: type = generic_interface_type @AddAssignWith [concrete]
// CHECK:STDOUT:   %AddAssignWith.generic: %AddAssignWith.type.fc6 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.393: type = facet_type <@AddAssignWith, @AddAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.5af: type = fn_type @Op.2, @AddAssignWith(%C) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness file.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Op.type.0d7: type = fn_type @Op.loc23 [concrete]
// CHECK:STDOUT:   %Op.572: %Op.type.0d7 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.393 = facet_value %C, (%AddAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %TestIncNonRef.type: type = fn_type @TestIncNonRef [concrete]
// CHECK:STDOUT:   %TestIncNonRef: %TestIncNonRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %.bc7: type = fn_type_with_self_type %Op.type.e3a, %Inc.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.9, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.5d7, @Op.9(%C) [concrete]
// CHECK:STDOUT:   %TestAddAssignNonRef.type: type = fn_type @TestAddAssignNonRef [concrete]
// CHECK:STDOUT:   %TestAddAssignNonRef: %TestAddAssignNonRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %.e2d: type = fn_type_with_self_type %Op.type.5af, %AddAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .AddAssignWith = %Core.AddAssignWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.AddAssignWith: %AddAssignWith.type.fc6 = import_ref Core//prelude/operators/arithmetic, AddAssignWith, loaded [concrete = constants.%AddAssignWith.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestIncNonRef = %TestIncNonRef.decl
// CHECK:STDOUT:     .TestAddAssignNonRef = %TestAddAssignNonRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @Inc.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, imports.%Core.Inc [concrete = constants.%Inc.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (@Inc.impl.%Op.decl), @Inc.impl [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness %Inc.impl_witness_table [concrete = constants.%Inc.impl_witness]
// CHECK:STDOUT:   impl_decl @AddAssignWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc22_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AddAssignWith.ref: %AddAssignWith.type.fc6 = name_ref AddAssignWith, imports.%Core.AddAssignWith [concrete = constants.%AddAssignWith.generic]
// CHECK:STDOUT:     %C.ref.loc22_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddAssignWith.type: type = facet_type <@AddAssignWith, @AddAssignWith(constants.%C)> [concrete = constants.%AddAssignWith.type.393]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (@AddAssignWith.impl.%Op.decl), @AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness %AddAssignWith.impl_witness_table [concrete = constants.%AddAssignWith.impl_witness]
// CHECK:STDOUT:   %TestIncNonRef.decl: %TestIncNonRef.type = fn_decl @TestIncNonRef [concrete = constants.%TestIncNonRef] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAddAssignNonRef.decl: %TestAddAssignNonRef.type = fn_decl @TestAddAssignNonRef [concrete = constants.%TestAddAssignNonRef] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c48 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc37_27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc37_33: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Inc.impl: %C.ref as %Inc.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.73a = fn_decl @Op.loc20 [concrete = constants.%Op.0c9] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc20_9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc20_21: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%Inc.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AddAssignWith.impl: %C.ref.loc22_6 as %AddAssignWith.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.0d7 = fn_decl @Op.loc23 [concrete = constants.%Op.572] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc23_9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc23_21: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc23_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc23_20 [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc23_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%AddAssignWith.impl_witness
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
// CHECK:STDOUT: fn @Op.loc20(%self.param: %ptr.019);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.loc23(%self.param: %ptr.019, %other.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIncNonRef(%a.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0: %.bc7 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Op.0c9]
// CHECK:STDOUT:   %bound_method.loc34_3: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %.loc34: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc34_5.1: %ptr.019 = addr_of %.loc34
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method.loc34_3(%addr.loc34_5.1)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc34, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.5d7, @Op.9(constants.%C) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_5: <bound method> = bound_method %.loc34, %Op.specific_fn
// CHECK:STDOUT:   %addr.loc34_5.2: %ptr.019 = addr_of %.loc34
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc34_5(%addr.loc34_5.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAddAssignNonRef(%a.param: %C, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.e2d = impl_witness_access constants.%AddAssignWith.impl_witness, element0 [concrete = constants.%Op.572]
// CHECK:STDOUT:   %bound_method.loc45_5: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %.loc45: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc45_3.1: %ptr.019 = addr_of %.loc45
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method.loc45_5(%addr.loc45_3.1, %b.ref)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc45, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.5d7, @Op.9(constants.%C) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_3: <bound method> = bound_method %.loc45, %Op.specific_fn
// CHECK:STDOUT:   %addr.loc45_3.2: %ptr.019 = addr_of %.loc45
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc45_3(%addr.loc45_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
