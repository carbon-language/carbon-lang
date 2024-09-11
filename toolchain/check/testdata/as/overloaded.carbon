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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %X, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %As.type: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @As.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @As, @As(%X) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @As(%X) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.7, %Convert.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.12: type = interface_type @As, @As(i32) [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @As(i32) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %.12, %Convert.type.5 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.15: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.17: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %.16, %Convert.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %As.type = import_ref Core//prelude/operators/as, inst+4, loaded [template = constants.%As]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.4: @As.%.2 (%.5) = import_ref Core//prelude/operators/as, inst+27, loaded [symbolic = @As.%.3 (constants.%.17)]
// CHECK:STDOUT:   %import_ref.5: @As.%Convert.type (%Convert.type.1) = import_ref Core//prelude/operators/as, inst+20, loaded [symbolic = @As.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+20, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+20, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   %.loc15_22.1: type = value_of_initializer %.loc15_20 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_22.2: type = converted %.loc15_20, %.loc15_22.1 [template = constants.%.7]
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32.loc15, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %As.ref.loc15: %As.type = name_ref As, imports.%import_ref.2 [template = constants.%As]
// CHECK:STDOUT:     %X.ref.loc15: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc15_20: init type = call %As.ref.loc15(%X.ref.loc15) [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_22.1: type = value_of_initializer %.loc19_18.3 [template = constants.%.12]
// CHECK:STDOUT:   %.loc19_22.2: type = converted %.loc19_18.3, %.loc19_22.1 [template = constants.%.12]
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %X.ref.loc19: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %As.ref.loc19: %As.type = name_ref As, imports.%import_ref.2 [template = constants.%As]
// CHECK:STDOUT:     %int.make_type_32.loc19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_18.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:     %.loc19_18.2: type = converted %int.make_type_32.loc19, %.loc19_18.1 [template = i32]
// CHECK:STDOUT:     %.loc19_18.3: init type = call %As.ref.loc19(%.loc19_18.2) [template = constants.%.12]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_32.loc23, %.loc23_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @As, @As(%Dest) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @As.%.1 (%.4), @As.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @As.%.2 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: i32 as %.7 {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc16_20.2: type = converted %int.make_type_32, %.loc16_20.1 [template = i32]
// CHECK:STDOUT:     %self.loc16_14.1: i32 = param self
// CHECK:STDOUT:     %self.loc16_14.2: i32 = bind_name self, %self.loc16_14.1
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.var: ref %X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = interface_witness (%Convert.decl) [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %X as %.12 {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4 = fn_decl @Convert.3 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %self.loc20_14.1: %X = param self
// CHECK:STDOUT:     %self.loc20_14.2: %X = bind_name self, %self.loc20_14.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc20_28.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc20_28.2: type = converted %int.make_type_32, %.loc20_28.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = interface_witness (%Convert.decl) [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @As.%.1 (%.4)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @As, @As(%Dest) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[@impl.1.%self.loc16_14.2: i32]() -> @impl.1.%return.var: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: i32 = name_ref self, @impl.1.%self.loc16_14.2
// CHECK:STDOUT:   %.loc16_51.1: %.3 = struct_literal (%self.ref)
// CHECK:STDOUT:   %.loc16_51.2: ref i32 = class_element_access @impl.1.%return.var, element0
// CHECK:STDOUT:   %.loc16_51.3: init i32 = initialize_from %self.ref to %.loc16_51.2
// CHECK:STDOUT:   %.loc16_51.4: init %X = class_init (%.loc16_51.3), @impl.1.%return.var
// CHECK:STDOUT:   %.loc16_52: init %X = converted %.loc16_51.1, %.loc16_51.4
// CHECK:STDOUT:   return %.loc16_52 to @impl.1.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[@impl.2.%self.loc20_14.2: %X]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, @impl.2.%self.loc20_14.2
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, @X.%.loc12_8 [template = @X.%.loc12_8]
// CHECK:STDOUT:   %.loc20_45.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc20_45.2: i32 = bind_value %.loc20_45.1
// CHECK:STDOUT:   return %.loc20_45.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_16: i32 = int_literal 4 [template = constants.%.16]
// CHECK:STDOUT:   %int.make_type_32.loc23_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_21.1: type = value_of_initializer %int.make_type_32.loc23_21 [template = i32]
// CHECK:STDOUT:   %.loc23_21.2: type = converted %int.make_type_32.loc23_21, %.loc23_21.1 [template = i32]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc23_26.1: init type = call constants.%As(constants.%X) [template = constants.%.7]
// CHECK:STDOUT:   %.loc23_26.2: %.8 = specific_constant imports.%import_ref.4, @As(constants.%X) [template = constants.%.9]
// CHECK:STDOUT:   %Convert.ref.loc23_26: %.8 = name_ref Convert, %.loc23_26.2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc23_26.3: %Convert.type.3 = interface_witness_access @impl.1.%.loc15, element0 [template = constants.%Convert.2]
// CHECK:STDOUT:   %.loc23_26.4: <bound method> = bound_method %.loc23_16, %.loc23_26.3 [template = constants.%.18]
// CHECK:STDOUT:   %.loc23_26.5: ref %X = temporary_storage
// CHECK:STDOUT:   %Convert.call.loc23_26: init %X = call %.loc23_26.4(%.loc23_16) to %.loc23_26.5
// CHECK:STDOUT:   %.loc23_26.6: init %X = converted %.loc23_16, %Convert.call.loc23_26
// CHECK:STDOUT:   %int.make_type_32.loc23_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_35.1: type = value_of_initializer %int.make_type_32.loc23_35 [template = i32]
// CHECK:STDOUT:   %.loc23_35.2: type = converted %int.make_type_32.loc23_35, %.loc23_35.1 [template = i32]
// CHECK:STDOUT:   %.loc23_32.1: init type = call constants.%As(i32) [template = constants.%.12]
// CHECK:STDOUT:   %.loc23_32.2: %.13 = specific_constant imports.%import_ref.4, @As(i32) [template = constants.%.14]
// CHECK:STDOUT:   %Convert.ref.loc23_32: %.13 = name_ref Convert, %.loc23_32.2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc23_26.7: ref %X = temporary %.loc23_26.5, %.loc23_26.6
// CHECK:STDOUT:   %.loc23_32.3: %Convert.type.5 = interface_witness_access @impl.2.%.loc19, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:   %.loc23_32.4: <bound method> = bound_method %.loc23_26.7, %.loc23_32.3
// CHECK:STDOUT:   %.loc23_26.8: %X = bind_value %.loc23_26.7
// CHECK:STDOUT:   %Convert.call.loc23_32: init i32 = call %.loc23_32.4(%.loc23_26.8)
// CHECK:STDOUT:   %.loc23_32.5: init i32 = converted %.loc23_26.6, %Convert.call.loc23_32
// CHECK:STDOUT:   %.loc23_38.1: i32 = value_of_initializer %.loc23_32.5
// CHECK:STDOUT:   %.loc23_38.2: i32 = converted %.loc23_32.5, %.loc23_38.1
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc23_38.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%X) {
// CHECK:STDOUT:   %Dest => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT:   %.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%X, i32) {
// CHECK:STDOUT:   %Dest => constants.%X
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.2 => constants.%.13
// CHECK:STDOUT:   %.3 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(i32, constants.%X) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %Self => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
