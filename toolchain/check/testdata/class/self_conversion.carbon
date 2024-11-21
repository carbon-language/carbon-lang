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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %SelfBase.type: type = fn_type @SelfBase [template]
// CHECK:STDOUT:   %SelfBase: %SelfBase.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %Base [template]
// CHECK:STDOUT:   %AddrSelfBase.type: type = fn_type @AddrSelfBase [template]
// CHECK:STDOUT:   %AddrSelfBase: %AddrSelfBase.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %.10: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.10, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.38: type = ptr_type %Derived [template]
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
// CHECK:STDOUT:     %.loc22_38.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc22: init type = call constants.%Int(%.loc22_38.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_38.2: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_38.3: type = converted %int.make_type_signed.loc22, %.loc22_38.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc22: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self.loc22: %Base = bind_name self, %self.param.loc22
// CHECK:STDOUT:     %return.param.loc22: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [template = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %self.patt: %.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc26_25: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc26: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc26_40: type = ptr_type %Base [template = constants.%.7]
// CHECK:STDOUT:     %self.param.loc26: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc26: %.7 = bind_name self, %self.param.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %p.patt: %.38 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.38 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc30_19: type = ptr_type %Derived [template = constants.%.38]
// CHECK:STDOUT:     %.loc30_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc30_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_25.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_25.3: type = converted %int.make_type_signed, %.loc30_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %.38 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.38 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc16: %.6 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [template = constants.%SelfBase] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc18: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc18_32.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_32.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_32.3: type = converted %int.make_type_signed.loc18, %.loc18_32.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc18: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self.loc18: %Base = bind_name self, %self.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %.loc18_29: ref %i32 = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [template = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %self.patt: %.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc26_25: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref.loc19: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc19: type = ptr_type %Base [template = constants.%.7]
// CHECK:STDOUT:     %self.param.loc19: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc19: %.7 = bind_name self, %self.param.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .SelfBase = %SelfBase.decl
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase[%self.param_patt: %Base]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Base = name_ref self, %self.loc22
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc23_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc23_14.2: %i32 = bind_value %.loc23_14.1
// CHECK:STDOUT:   return %.loc23_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase[addr %self.param_patt: %.7]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.7 = name_ref self, %self.loc26
// CHECK:STDOUT:   %.loc27_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc27_10: ref %i32 = class_element_access %.loc27_4, element0
// CHECK:STDOUT:   %.loc27_15: Core.IntLiteral = int_value 1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc27_13.1: %Convert.type.2 = interface_witness_access constants.%.34, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc27_13.2: <bound method> = bound_method %.loc27_15, %.loc27_13.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc27_13.3: <specific function> = specific_function %.loc27_13.2, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc27_13.3(%.loc27_15) [template = constants.%.37]
// CHECK:STDOUT:   %.loc27_13.4: init %i32 = converted %.loc27_15, %int.convert_checked [template = constants.%.37]
// CHECK:STDOUT:   assign %.loc27_10, %.loc27_13.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param_patt: %.38) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc31: %.38 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_4.1: ref %Derived = deref %p.ref.loc31
// CHECK:STDOUT:   %AddrSelfBase.ref: %AddrSelfBase.type = name_ref AddrSelfBase, @Derived.%AddrSelfBase.decl [template = constants.%AddrSelfBase]
// CHECK:STDOUT:   %.loc31_7: <bound method> = bound_method %.loc31_4.1, %AddrSelfBase.ref
// CHECK:STDOUT:   %.loc31_4.2: %.38 = addr_of %.loc31_4.1
// CHECK:STDOUT:   %.loc31_4.3: ref %Derived = deref %.loc31_4.2
// CHECK:STDOUT:   %.loc31_4.4: ref %Base = class_element_access %.loc31_4.3, element0
// CHECK:STDOUT:   %.loc31_4.5: %.7 = addr_of %.loc31_4.4
// CHECK:STDOUT:   %.loc31_4.6: %.7 = converted %.loc31_4.2, %.loc31_4.5
// CHECK:STDOUT:   %AddrSelfBase.call: init %empty_tuple.type = call %.loc31_7(%.loc31_4.6)
// CHECK:STDOUT:   %p.ref.loc32: %.38 = name_ref p, %p
// CHECK:STDOUT:   %.loc32_11.1: ref %Derived = deref %p.ref.loc32
// CHECK:STDOUT:   %SelfBase.ref: %SelfBase.type = name_ref SelfBase, @Derived.%SelfBase.decl [template = constants.%SelfBase]
// CHECK:STDOUT:   %.loc32_14: <bound method> = bound_method %.loc32_11.1, %SelfBase.ref
// CHECK:STDOUT:   %.loc32_11.2: ref %Base = class_element_access %.loc32_11.1, element0
// CHECK:STDOUT:   %.loc32_11.3: ref %Base = converted %.loc32_11.1, %.loc32_11.2
// CHECK:STDOUT:   %.loc32_11.4: %Base = bind_value %.loc32_11.3
// CHECK:STDOUT:   %SelfBase.call: init %i32 = call %.loc32_14(%.loc32_11.4)
// CHECK:STDOUT:   %.loc32_25.1: %i32 = value_of_initializer %SelfBase.call
// CHECK:STDOUT:   %.loc32_25.2: %i32 = converted %SelfBase.call, %.loc32_25.1
// CHECK:STDOUT:   return %.loc32_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
