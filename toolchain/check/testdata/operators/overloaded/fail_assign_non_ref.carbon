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
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+6]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-18]]:9: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Op[addr self: C*](other: C);
  // CHECK:STDERR:         ^~~~~~~~~~~~~
  a += b;
}

// CHECK:STDOUT: --- fail_assign_non_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %AddAssign.type: type = facet_type <@AddAssign> [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT:   %TestIncNonRef.type: type = fn_type @TestIncNonRef [template]
// CHECK:STDOUT:   %TestIncNonRef: %TestIncNonRef.type = struct_value () [template]
// CHECK:STDOUT:   %TestAddAssignNonRef.type: type = fn_type @TestAddAssignNonRef [template]
// CHECK:STDOUT:   %TestAddAssignNonRef: %TestAddAssignNonRef.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Inc = %import_ref.1
// CHECK:STDOUT:     .AddAssign = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+67, loaded [template = constants.%Inc.type]
// CHECK:STDOUT:   %import_ref.5: type = import_ref Core//prelude/operators/arithmetic, inst+36, loaded [template = constants.%AddAssign.type]
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
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, imports.%import_ref.1 [template = constants.%Inc.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, imports.%import_ref.5 [template = constants.%AddAssign.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestIncNonRef.decl: %TestIncNonRef.type = fn_decl @TestIncNonRef [template = constants.%TestIncNonRef] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAddAssignNonRef.decl: %TestAddAssignNonRef.type = fn_decl @TestAddAssignNonRef [template = constants.%TestAddAssignNonRef] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc33_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc33_33: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %Inc.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc16: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [template = constants.%ptr.1]
// CHECK:STDOUT:     %self.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %ptr.1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C.ref as %AddAssign.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.3 = fn_decl @Op.3 [template = constants.%Op.3] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc19: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc19_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [template = constants.%ptr.1]
// CHECK:STDOUT:     %C.ref.loc19_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %ptr.1 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.decl) [template = constants.%interface.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[addr %self.param_patt: %ptr.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr %self.param_patt: %ptr.1](%other.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIncNonRef(%a.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0: %Op.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %Op.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAddAssignNonRef(%a.param_patt: %C, %b.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %Op.type.4 = interface_witness_access constants.%interface.2, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %Op.bound(<error>, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
