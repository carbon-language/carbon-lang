// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/overloaded.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/overloaded.carbon

class X {
  var n: i32;
}

impl i32 as Core.As(X) {
  fn Convert[self: i32]() -> X { return {.n = self}; }
}

impl X as Core.As(i32) {
  fn Convert[self: X]() -> i32 { return self.n; }
}

let n: i32 = ((4 as i32) as X) as i32;

// CHECK:STDOUT: --- overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As.generic: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%X)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @As(%X) [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.9, @impl.7(%int_32) [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %interface.11: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_4.1, %Convert.17 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound.1, @Convert.9(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_4.2, %Convert.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %As.type.1 = import_ref Core//prelude/operators/as, inst+13, loaded [template = constants.%As.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_signed, %.loc15_6.1 [template = constants.%i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %As.ref: %As.type.1 = name_ref As, imports.%import_ref.2 [template = constants.%As.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%X)> [template = constants.%As.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %As.ref: %As.type.1 = name_ref As, imports.%import_ref.2 [template = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_22.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_22.2: type = converted %int.make_type_signed, %.loc19_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32)> [template = constants.%As.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_signed, %.loc23_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %self.patt: %i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_20.2: type = converted %int.make_type_signed, %.loc16_20.1 [template = constants.%i32]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %self.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %X.ref as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4 = fn_decl @Convert.3 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %self.patt: %X = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %X = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_28.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_28.2: type = converted %int.make_type_signed, %.loc20_28.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %X = value_param runtime_param0
// CHECK:STDOUT:     %self: %X = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Convert.decl) [template = constants.%interface.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %X.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: %i32]() -> %return.param_patt: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %i32 = name_ref self, %self
// CHECK:STDOUT:   %.loc16_51.1: %struct_type.n = struct_literal (%self.ref)
// CHECK:STDOUT:   %.loc16_51.2: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc16_51.3: init %i32 = initialize_from %self.ref to %.loc16_51.2
// CHECK:STDOUT:   %.loc16_51.4: init %X = class_init (%.loc16_51.3), %return
// CHECK:STDOUT:   %.loc16_52: init %X = converted %.loc16_51.1, %.loc16_51.4
// CHECK:STDOUT:   return %.loc16_52 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %X]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %X.elem = name_ref n, @X.%.loc12_8 [template = @X.%.loc12_8]
// CHECK:STDOUT:   %.loc20_45.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc20_45.2: %i32 = bind_value %.loc20_45.1
// CHECK:STDOUT:   return %.loc20_45.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %int_32.loc23_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc23_21: init type = call constants.%Int(%int_32.loc23_21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_21.1: type = value_of_initializer %int.make_type_signed.loc23_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_21.2: type = converted %int.make_type_signed.loc23_21, %.loc23_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc23_18: %Convert.type.5 = interface_witness_access constants.%interface.11, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %Convert.bound.loc23_18: <bound method> = bound_method %int_4, %impl.elem0.loc23_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound.loc23_18, @Convert.9(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_4) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc23_18.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc23_18.2: %i32 = converted %int_4, %.loc23_18.1 [template = constants.%int_4.2]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %impl.elem0.loc23_26: %Convert.type.3 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.2]
// CHECK:STDOUT:   %Convert.bound.loc23_26: <bound method> = bound_method %.loc23_18.2, %impl.elem0.loc23_26 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %.loc23_26.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Convert.call.loc23_26: init %X = call %Convert.bound.loc23_26(%.loc23_18.2) to %.loc23_26.1
// CHECK:STDOUT:   %.loc23_26.2: init %X = converted %.loc23_18.2, %Convert.call.loc23_26
// CHECK:STDOUT:   %int_32.loc23_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc23_35: init type = call constants.%Int(%int_32.loc23_35) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.1: type = value_of_initializer %int.make_type_signed.loc23_35 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.2: type = converted %int.make_type_signed.loc23_35, %.loc23_35.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc23_32: %Convert.type.5 = interface_witness_access constants.%interface.2, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:   %Convert.bound.loc23_32: <bound method> = bound_method %.loc23_26.2, %impl.elem0.loc23_32
// CHECK:STDOUT:   %.loc23_26.3: ref %X = temporary %.loc23_26.1, %.loc23_26.2
// CHECK:STDOUT:   %.loc23_26.4: %X = bind_value %.loc23_26.3
// CHECK:STDOUT:   %Convert.call.loc23_32: init %i32 = call %Convert.bound.loc23_32(%.loc23_26.4)
// CHECK:STDOUT:   %.loc23_32.1: %i32 = value_of_initializer %Convert.call.loc23_32
// CHECK:STDOUT:   %.loc23_32.2: %i32 = converted %.loc23_26.2, %.loc23_32.1
// CHECK:STDOUT:   %n: %i32 = bind_name n, %.loc23_32.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
