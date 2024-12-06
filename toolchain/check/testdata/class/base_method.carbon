// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_method.carbon

base class Base {
  var a: i32;

  fn F[addr self: Self*]();
}

fn Base.F[addr self: Self*]() {
  (*self).a = 1;
}

class Derived {
  extend base: Base;
}

fn Call(p: Derived*) {
  (*p).F();
}

// CHECK:STDOUT: --- base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.a [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc17: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %Base [template = constants.%ptr.1]
// CHECK:STDOUT:     %self.param.loc17: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc17: %ptr.1 = bind_name self, %self.param.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %p.patt: %ptr.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %ptr: type = ptr_type %Derived [template = constants.%ptr.3]
// CHECK:STDOUT:     %p.param: %ptr.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.3 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %Base.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc17: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc14: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %ptr.loc14: type = ptr_type %Base [template = constants.%ptr.1]
// CHECK:STDOUT:     %self.param.loc14: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc14: %ptr.1 = bind_name self, %self.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc22: %Derived.elem = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt: %ptr.1]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.1 = name_ref self, %self.loc17
// CHECK:STDOUT:   %.loc18_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc18_10: ref %i32 = class_element_access %.loc18_4, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_13: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %.loc18_10, %.loc18_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param_patt: %ptr.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_4.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Base.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc26_4.1, %F.ref
// CHECK:STDOUT:   %addr.loc26_4.1: %ptr.3 = addr_of %.loc26_4.1
// CHECK:STDOUT:   %.loc26_4.2: ref %Derived = deref %addr.loc26_4.1
// CHECK:STDOUT:   %.loc26_4.3: ref %Base = class_element_access %.loc26_4.2, element0
// CHECK:STDOUT:   %addr.loc26_4.2: %ptr.1 = addr_of %.loc26_4.3
// CHECK:STDOUT:   %.loc26_4.4: %ptr.1 = converted %addr.loc26_4.1, %addr.loc26_4.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc26_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
