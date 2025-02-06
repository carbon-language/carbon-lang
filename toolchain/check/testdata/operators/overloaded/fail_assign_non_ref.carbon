// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
impl C as Core.AddAssign {
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.type.e3a: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %impl_witness.ec3: <witness> = impl_witness (@impl.1.%Op.decl) [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.73a: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.0c9: %Op.type.73a = struct_value () [template]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %C, %impl_witness.ec3 [template]
// CHECK:STDOUT:   %AddAssign.type: type = facet_type <@AddAssign> [template]
// CHECK:STDOUT:   %Op.type.421: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %impl_witness.95d: <witness> = impl_witness (@impl.2.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.0b8: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.d8e: %Op.type.0b8 = struct_value () [template]
// CHECK:STDOUT:   %AddAssign.facet: %AddAssign.type = facet_value %C, %impl_witness.95d [template]
// CHECK:STDOUT:   %TestIncNonRef.type: type = fn_type @TestIncNonRef [template]
// CHECK:STDOUT:   %TestIncNonRef: %TestIncNonRef.type = struct_value () [template]
// CHECK:STDOUT:   %.e6d: type = fn_type_with_self_type %Op.type.e3a, %Inc.facet [template]
// CHECK:STDOUT:   %TestAddAssignNonRef.type: type = fn_type @TestAddAssignNonRef [template]
// CHECK:STDOUT:   %TestAddAssignNonRef: %TestAddAssignNonRef.type = struct_value () [template]
// CHECK:STDOUT:   %.499: type = fn_type_with_self_type %Op.type.421, %AddAssign.facet [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .AddAssign = %Core.AddAssign
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [template = constants.%Inc.type]
// CHECK:STDOUT:   %Core.AddAssign: type = import_ref Core//prelude/operators/arithmetic, AddAssign, loaded [template = constants.%AddAssign.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestIncNonRef = %TestIncNonRef.decl
// CHECK:STDOUT:     .TestAddAssignNonRef = %TestAddAssignNonRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, imports.%Core.Inc [template = constants.%Inc.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (@impl.1.%Op.decl) [template = constants.%impl_witness.ec3]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, imports.%Core.AddAssign [template = constants.%AddAssign.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.2.%Op.decl) [template = constants.%impl_witness.95d]
// CHECK:STDOUT:   %TestIncNonRef.decl: %TestIncNonRef.type = fn_decl @TestIncNonRef [template = constants.%TestIncNonRef] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAddAssignNonRef.decl: %TestAddAssignNonRef.type = fn_decl @TestAddAssignNonRef [template = constants.%TestAddAssignNonRef] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref.loc33_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %C.ref.loc33_33: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %Inc.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.73a = fn_decl @Op.2 [template = constants.%Op.0c9] {
// CHECK:STDOUT:     %self.patt: %ptr.019 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.019 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc16_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param runtime_param0
// CHECK:STDOUT:     %.loc16_21: type = splice_block %ptr [template = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C.ref as %AddAssign.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.0b8 = fn_decl @Op.4 [template = constants.%Op.d8e] {
// CHECK:STDOUT:     %self.patt: %ptr.019 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.019 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc19_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param runtime_param0
// CHECK:STDOUT:     %.loc19_21: type = splice_block %ptr [template = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc19_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %C.ref.loc19_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr %self.param_patt: %ptr.019]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr %self.param_patt: %ptr.019](%other.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIncNonRef(%a.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0: %.e6d = impl_witness_access constants.%impl_witness.ec3, element0 [template = constants.%Op.0c9]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAddAssignNonRef(%a.param_patt: %C, %b.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.499 = impl_witness_access constants.%impl_witness.95d, element0 [template = constants.%Op.d8e]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(<error>, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
