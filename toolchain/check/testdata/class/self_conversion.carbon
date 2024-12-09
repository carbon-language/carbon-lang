// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self_conversion.carbon

base class Base {
  var a: i32;
}

class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> i32;
  fn AddrSelfBase[addr self: Base*]();
}

fn Derived.SelfBase[self: Base]() -> i32 {
  return self.a;
}

fn Derived.AddrSelfBase[addr self: Base*]() {
  (*self).a = 1;
}

fn Call(p: Derived*) -> i32 {
  (*p).AddrSelfBase();
  return (*p).SelfBase();
}

// CHECK:STDOUT: --- self_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.a [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %SelfBase.type: type = fn_type @SelfBase [template]
// CHECK:STDOUT:   %SelfBase: %SelfBase.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %Base [template]
// CHECK:STDOUT:   %AddrSelfBase.type: type = fn_type @AddrSelfBase [template]
// CHECK:STDOUT:   %AddrSelfBase: %AddrSelfBase.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
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
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [template = constants.%SelfBase] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc22: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc22: init type = call constants.%Int(%int_32.loc22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_38.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_38.2: type = converted %int.make_type_signed.loc22, %.loc22_38.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc22: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self.loc22: %Base = bind_name self, %self.param.loc22
// CHECK:STDOUT:     %return.param.loc22: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc22: ref %i32 = return_slot %return.param.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [template = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %self.patt: %ptr.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc26: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc26: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %ptr.loc26: type = ptr_type %Base [template = constants.%ptr.2]
// CHECK:STDOUT:     %self.param.loc26: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc26: %ptr.2 = bind_name self, %self.param.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %p.patt: %ptr.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %ptr: type = ptr_type %Derived [template = constants.%ptr.3]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_25.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_25.2: type = converted %int.make_type_signed, %.loc30_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %Base.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc16: %Derived.elem = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [template = constants.%SelfBase] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc18: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18: init type = call constants.%Int(%int_32.loc18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_32.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_32.2: type = converted %int.make_type_signed.loc18, %.loc18_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc18: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self.loc18: %Base = bind_name self, %self.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc18: ref %i32 = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [template = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %self.patt: %ptr.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc26: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc19: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %Base [template = constants.%ptr.2]
// CHECK:STDOUT:     %self.param.loc19: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc19: %ptr.2 = bind_name self, %self.param.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .SelfBase = %SelfBase.decl
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase[%self.param_patt: %Base]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Base = name_ref self, %self.loc22
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc23_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc23_14.2: %i32 = bind_value %.loc23_14.1
// CHECK:STDOUT:   return %.loc23_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase[addr %self.param_patt: %ptr.2]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.2 = name_ref self, %self.loc26
// CHECK:STDOUT:   %.loc27_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc27_10: ref %i32 = class_element_access %.loc27_4, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc27_13: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %.loc27_10, %.loc27_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param_patt: %ptr.3) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc31: %ptr.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_4.1: ref %Derived = deref %p.ref.loc31
// CHECK:STDOUT:   %AddrSelfBase.ref: %AddrSelfBase.type = name_ref AddrSelfBase, @Derived.%AddrSelfBase.decl [template = constants.%AddrSelfBase]
// CHECK:STDOUT:   %AddrSelfBase.bound: <bound method> = bound_method %.loc31_4.1, %AddrSelfBase.ref
// CHECK:STDOUT:   %addr.loc31_4.1: %ptr.3 = addr_of %.loc31_4.1
// CHECK:STDOUT:   %.loc31_4.2: ref %Derived = deref %addr.loc31_4.1
// CHECK:STDOUT:   %.loc31_4.3: ref %Base = class_element_access %.loc31_4.2, element0
// CHECK:STDOUT:   %addr.loc31_4.2: %ptr.2 = addr_of %.loc31_4.3
// CHECK:STDOUT:   %.loc31_4.4: %ptr.2 = converted %addr.loc31_4.1, %addr.loc31_4.2
// CHECK:STDOUT:   %AddrSelfBase.call: init %empty_tuple.type = call %AddrSelfBase.bound(%.loc31_4.4)
// CHECK:STDOUT:   %p.ref.loc32: %ptr.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc32_11.1: ref %Derived = deref %p.ref.loc32
// CHECK:STDOUT:   %SelfBase.ref: %SelfBase.type = name_ref SelfBase, @Derived.%SelfBase.decl [template = constants.%SelfBase]
// CHECK:STDOUT:   %SelfBase.bound: <bound method> = bound_method %.loc32_11.1, %SelfBase.ref
// CHECK:STDOUT:   %.loc32_11.2: ref %Base = class_element_access %.loc32_11.1, element0
// CHECK:STDOUT:   %.loc32_11.3: ref %Base = converted %.loc32_11.1, %.loc32_11.2
// CHECK:STDOUT:   %.loc32_11.4: %Base = bind_value %.loc32_11.3
// CHECK:STDOUT:   %SelfBase.call: init %i32 = call %SelfBase.bound(%.loc32_11.4)
// CHECK:STDOUT:   %.loc32_25.1: %i32 = value_of_initializer %SelfBase.call
// CHECK:STDOUT:   %.loc32_25.2: %i32 = converted %SelfBase.call, %.loc32_25.1
// CHECK:STDOUT:   return %.loc32_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
