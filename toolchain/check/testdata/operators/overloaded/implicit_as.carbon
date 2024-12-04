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
fn Source(T:! type) -> T { return Source(T); }

fn Test() {
  Sink_i32(Source(X));
  Sink_X(Source(i32));
}

// CHECK:STDOUT: --- implicit_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%X)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(%X) [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%Convert.4) [template]
// CHECK:STDOUT:   %Sink_i32.type: type = fn_type @Sink_i32 [template]
// CHECK:STDOUT:   %Sink_i32: %Sink_i32.type = struct_value () [template]
// CHECK:STDOUT:   %Sink_X.type: type = fn_type @Sink_X [template]
// CHECK:STDOUT:   %Sink_X: %Sink_X.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Source.type: type = fn_type @Source [template]
// CHECK:STDOUT:   %Source: %Source.type = struct_value () [template]
// CHECK:STDOUT:   %Source.specific_fn.1: <specific function> = specific_function %Source, @Source(%T) [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %Source.specific_fn.2: <specific function> = specific_function %Source, @Source(%X) [template]
// CHECK:STDOUT:   %Source.specific_fn.3: <specific function> = specific_function %Source, @Source(%i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+56, loaded [template = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_signed, %.loc15_6.1 [template = constants.%i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%X)> [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_30.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_30.2: type = converted %int.make_type_signed, %.loc19_30.1 [template = constants.%i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_i32.decl: %Sink_i32.type = fn_decl @Sink_i32 [template = constants.%Sink_i32] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc25_16.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc25_16.2: type = converted %int.make_type_signed, %.loc25_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_X.decl: %Sink_X.type = fn_decl @Sink_X [template = constants.%Sink_X] {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %X = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %x.param: %X = value_param runtime_param0
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: %Source.type = fn_decl @Source [template = constants.%Source] {
// CHECK:STDOUT:     %T.patt.loc27_11.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_11.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc27_11.1, runtime_param<invalid> [symbolic = %T.patt.loc27_11.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @Source.%T.loc27_11.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Source.%T.loc27_11.2 (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc27_24: type = name_ref T, %T.loc27_11.1 [symbolic = %T.loc27_11.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc27_11.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc27_11.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @Source.%T.loc27_11.2 (%T) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Source.%T.loc27_11.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %ImplicitAs.type {
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
// CHECK:STDOUT: impl @impl.2: %X.ref as %ImplicitAs.type {
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
// CHECK:STDOUT: fn @Sink_i32(%n.param_patt: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_X(%x.param_patt: %X);
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Source(%T.loc27_11.1: type) {
// CHECK:STDOUT:   %T.loc27_11.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc27_11.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc27_11.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_11.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Source.specific_fn.loc27_35.2: <specific function> = specific_function constants.%Source, @Source(%T.loc27_11.2) [symbolic = %Source.specific_fn.loc27_35.2 (constants.%Source.specific_fn.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @Source.%T.loc27_11.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Source.ref: %Source.type = name_ref Source, file.%Source.decl [template = constants.%Source]
// CHECK:STDOUT:     %T.ref.loc27_42: type = name_ref T, %T.loc27_11.1 [symbolic = %T.loc27_11.2 (constants.%T)]
// CHECK:STDOUT:     %Source.specific_fn.loc27_35.1: <specific function> = specific_function %Source.ref, @Source(constants.%T) [symbolic = %Source.specific_fn.loc27_35.2 (constants.%Source.specific_fn.1)]
// CHECK:STDOUT:     %Source.call: init @Source.%T.loc27_11.2 (%T) = call %Source.specific_fn.loc27_35.1()
// CHECK:STDOUT:     %.loc27_44.1: @Source.%T.loc27_11.2 (%T) = value_of_initializer %Source.call
// CHECK:STDOUT:     %.loc27_44.2: @Source.%T.loc27_11.2 (%T) = converted %Source.call, %.loc27_44.1
// CHECK:STDOUT:     return %.loc27_44.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sink_i32.ref: %Sink_i32.type = name_ref Sink_i32, file.%Sink_i32.decl [template = constants.%Sink_i32]
// CHECK:STDOUT:   %Source.ref.loc30: %Source.type = name_ref Source, file.%Source.decl [template = constants.%Source]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %Source.specific_fn.loc30: <specific function> = specific_function %Source.ref.loc30, @Source(constants.%X) [template = constants.%Source.specific_fn.2]
// CHECK:STDOUT:   %.loc30_20.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Source.call.loc30: init %X = call %Source.specific_fn.loc30() to %.loc30_20.1
// CHECK:STDOUT:   %impl.elem0.loc30: %Convert.type.5 = interface_witness_access constants.%interface.2, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:   %Convert.bound.loc30: <bound method> = bound_method %Source.call.loc30, %impl.elem0.loc30
// CHECK:STDOUT:   %.loc30_20.2: ref %X = temporary %.loc30_20.1, %Source.call.loc30
// CHECK:STDOUT:   %.loc30_20.3: %X = bind_value %.loc30_20.2
// CHECK:STDOUT:   %Convert.call.loc30: init %i32 = call %Convert.bound.loc30(%.loc30_20.3)
// CHECK:STDOUT:   %.loc30_20.4: %i32 = value_of_initializer %Convert.call.loc30
// CHECK:STDOUT:   %.loc30_20.5: %i32 = converted %Source.call.loc30, %.loc30_20.4
// CHECK:STDOUT:   %Sink_i32.call: init %empty_tuple.type = call %Sink_i32.ref(%.loc30_20.5)
// CHECK:STDOUT:   %Sink_X.ref: %Sink_X.type = name_ref Sink_X, file.%Sink_X.decl [template = constants.%Sink_X]
// CHECK:STDOUT:   %Source.ref.loc31: %Source.type = name_ref Source, file.%Source.decl [template = constants.%Source]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc31_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc31_20.2: type = converted %int.make_type_signed, %.loc31_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %Source.specific_fn.loc31: <specific function> = specific_function %Source.ref.loc31, @Source(constants.%i32) [template = constants.%Source.specific_fn.3]
// CHECK:STDOUT:   %Source.call.loc31: init %i32 = call %Source.specific_fn.loc31()
// CHECK:STDOUT:   %impl.elem0.loc31: %Convert.type.3 = interface_witness_access constants.%interface.1, element0 [template = constants.%Convert.2]
// CHECK:STDOUT:   %Convert.bound.loc31: <bound method> = bound_method %Source.call.loc31, %impl.elem0.loc31
// CHECK:STDOUT:   %.loc31_20.3: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc31_20.4: %i32 = value_of_initializer %Source.call.loc31
// CHECK:STDOUT:   %.loc31_20.5: %i32 = converted %Source.call.loc31, %.loc31_20.4
// CHECK:STDOUT:   %Convert.call.loc31: init %X = call %Convert.bound.loc31(%.loc31_20.5) to %.loc31_20.3
// CHECK:STDOUT:   %.loc31_20.6: init %X = converted %Source.call.loc31, %Convert.call.loc31
// CHECK:STDOUT:   %.loc31_20.7: ref %X = temporary %.loc31_20.3, %.loc31_20.6
// CHECK:STDOUT:   %.loc31_20.8: %X = bind_value %.loc31_20.7
// CHECK:STDOUT:   %Sink_X.call: init %empty_tuple.type = call %Sink_X.ref(%.loc31_20.8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%T) {
// CHECK:STDOUT:   %T.loc27_11.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc27_11.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Source.specific_fn.loc27_35.2 => constants.%Source.specific_fn.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(%T.loc27_11.2) {
// CHECK:STDOUT:   %T.loc27_11.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc27_11.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%X) {
// CHECK:STDOUT:   %T.loc27_11.2 => constants.%X
// CHECK:STDOUT:   %T.patt.loc27_11.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Source.specific_fn.loc27_35.2 => constants.%Source.specific_fn.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%i32) {
// CHECK:STDOUT:   %T.loc27_11.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc27_11.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Source.specific_fn.loc27_35.2 => constants.%Source.specific_fn.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
