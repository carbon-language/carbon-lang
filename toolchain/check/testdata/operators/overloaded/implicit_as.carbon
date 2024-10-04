// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/implicit_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/implicit_as.carbon

class X {
  var n: i32;
}

impl i32 as Core.ImplicitAs(X) {
  fn Convert[self: i32]() -> X { return {.n = self}; }
}

impl X as Core.ImplicitAs(i32) {
  fn Convert[self: X]() -> i32 { return self.n; }
}

// TODO: fn Sink(T:! type, x: T);
// ... once we stop deducing `T` from the type of the second argument in this case.
fn Sink_i32(n: i32);
fn Sink_X(x: X);
fn Source(T:! type) -> T;

fn Test() {
  Sink_i32(Source(X));
  Sink_X(Source(i32));
}

// CHECK:STDOUT: --- implicit_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %X, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%X) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(%X) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.3 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = interface_type @ImplicitAs, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.5 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %Sink_i32.type: type = fn_type @Sink_i32 [template]
// CHECK:STDOUT:   %Sink_i32: %Sink_i32.type = struct_value () [template]
// CHECK:STDOUT:   %Sink_X.type: type = fn_type @Sink_X [template]
// CHECK:STDOUT:   %Sink_X: %Sink_X.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Source.type: type = fn_type @Source [template]
// CHECK:STDOUT:   %Source: %Source.type = struct_value () [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %.14: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.5) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%Convert.type (%Convert.type.1) = import_ref Core//prelude/operators/as, inst+56, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Sink_i32 = %Sink_i32.decl
// CHECK:STDOUT:     .Sink_X = %Sink_X.decl
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%X) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_26.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc19_26.2: type = converted %int.make_type_32, %.loc19_26.1 [template = i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(i32) [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_i32.decl: %Sink_i32.type = fn_decl @Sink_i32 [template = constants.%Sink_i32] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc25_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc25_16.2: type = converted %int.make_type_32, %.loc25_16.1 [template = i32]
// CHECK:STDOUT:     %n.param: i32 = param n, runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_X.decl: %Sink_X.type = fn_decl @Sink_X [template = constants.%Sink_X] {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %x.param: %X = param x, runtime_param0
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: %Source.type = fn_decl @Source [template = constants.%Source] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc27: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc27 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %return: ref @Source.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc16_20.2: type = converted %int.make_type_32, %.loc16_20.1 [template = i32]
// CHECK:STDOUT:     %self.param: i32 = param self, runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return: ref %X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_32: <witness> = interface_witness (%Convert.decl) [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc15_32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %X.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4 = fn_decl @Convert.3 [template = constants.%Convert.4] {
// CHECK:STDOUT:     %self.patt: %X = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %self.param: %X = param self, runtime_param0
// CHECK:STDOUT:     %self: %X = bind_name self, %self.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc20_28.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc20_28.2: type = converted %int.make_type_32, %.loc20_28.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_32: <witness> = interface_witness (%Convert.decl) [template = constants.%.13]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc19_32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self: i32]() -> %return: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: i32 = name_ref self, %self
// CHECK:STDOUT:   %.loc16_51.1: %.3 = struct_literal (%self.ref)
// CHECK:STDOUT:   %.loc16_51.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc16_51.3: init i32 = initialize_from %self.ref to %.loc16_51.2
// CHECK:STDOUT:   %.loc16_51.4: init %X = class_init (%.loc16_51.3), %return
// CHECK:STDOUT:   %.loc16_52: init %X = converted %.loc16_51.1, %.loc16_51.4
// CHECK:STDOUT:   return %.loc16_52 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self: %X]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, @X.%.loc12_8 [template = @X.%.loc12_8]
// CHECK:STDOUT:   %.loc20_45.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc20_45.2: i32 = bind_value %.loc20_45.1
// CHECK:STDOUT:   return %.loc20_45.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_i32(%n: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_X(%x: %X);
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Source(%T.loc27: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc27: type) -> @Source.%T.1 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sink_i32.ref: %Sink_i32.type = name_ref Sink_i32, file.%Sink_i32.decl [template = constants.%Sink_i32]
// CHECK:STDOUT:   %Source.ref.loc30: %Source.type = name_ref Source, file.%Source.decl [template = constants.%Source]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc30_18.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Source.call.loc30: init %X = call %Source.ref.loc30() to %.loc30_18.1
// CHECK:STDOUT:   %ImplicitAs.type.loc30: type = interface_type @ImplicitAs, @ImplicitAs(i32) [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %.loc30_11.1: %.11 = specific_constant imports.%import_ref.4, @ImplicitAs(i32) [template = constants.%.12]
// CHECK:STDOUT:   %Convert.ref.loc30: %.11 = name_ref Convert, %.loc30_11.1 [template = constants.%.12]
// CHECK:STDOUT:   %.loc30_18.2: ref %X = temporary %.loc30_18.1, %Source.call.loc30
// CHECK:STDOUT:   %.loc30_11.2: %Convert.type.5 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:   %.loc30_11.3: <bound method> = bound_method %.loc30_18.2, %.loc30_11.2
// CHECK:STDOUT:   %.loc30_18.3: %X = bind_value %.loc30_18.2
// CHECK:STDOUT:   %Convert.call.loc30: init i32 = call %.loc30_11.3(%.loc30_18.3)
// CHECK:STDOUT:   %.loc30_11.4: init i32 = converted %Source.call.loc30, %Convert.call.loc30
// CHECK:STDOUT:   %.loc30_11.5: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc30_11.6: ref i32 = temporary %.loc30_11.5, %.loc30_11.4
// CHECK:STDOUT:   %.loc30_11.7: %X = bind_value %.loc30_11.6
// CHECK:STDOUT:   %Sink_i32.call: init %.1 = call %Sink_i32.ref(%.loc30_11.7)
// CHECK:STDOUT:   %Sink_X.ref: %Sink_X.type = name_ref Sink_X, file.%Sink_X.decl [template = constants.%Sink_X]
// CHECK:STDOUT:   %Source.ref.loc31: %Source.type = name_ref Source, file.%Source.decl [template = constants.%Source]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc31_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc31_16.2: type = converted %int.make_type_32, %.loc31_16.1 [template = i32]
// CHECK:STDOUT:   %Source.call.loc31: init i32 = call %Source.ref.loc31()
// CHECK:STDOUT:   %ImplicitAs.type.loc31: type = interface_type @ImplicitAs, @ImplicitAs(constants.%X) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc31_9.1: %.7 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%X) [template = constants.%.8]
// CHECK:STDOUT:   %Convert.ref.loc31: %.7 = name_ref Convert, %.loc31_9.1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc31_16.3: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc31_16.4: ref i32 = temporary %.loc31_16.3, %Source.call.loc31
// CHECK:STDOUT:   %.loc31_9.2: %Convert.type.3 = interface_witness_access constants.%.9, element0 [template = constants.%Convert.2]
// CHECK:STDOUT:   %.loc31_9.3: <bound method> = bound_method %.loc31_16.4, %.loc31_9.2
// CHECK:STDOUT:   %.loc31_9.4: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc31_16.5: i32 = bind_value %.loc31_16.4
// CHECK:STDOUT:   %Convert.call.loc31: init %X = call %.loc31_9.3(%.loc31_16.5) to %.loc31_9.4
// CHECK:STDOUT:   %.loc31_9.5: init %X = converted %Source.call.loc31, %Convert.call.loc31
// CHECK:STDOUT:   %.loc31_9.6: ref %X = temporary %.loc31_9.4, %.loc31_9.5
// CHECK:STDOUT:   %.loc31_9.7: i32 = bind_value %.loc31_9.6
// CHECK:STDOUT:   %Sink_X.call: init %.1 = call %Sink_X.ref(%.loc31_9.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%X) {
// CHECK:STDOUT:   %Dest => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%X, i32) {
// CHECK:STDOUT:   %Dest => constants.%X
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(i32, constants.%X) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%X) {
// CHECK:STDOUT:   %T.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(i32) {
// CHECK:STDOUT:   %T.1 => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
