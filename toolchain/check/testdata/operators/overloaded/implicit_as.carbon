// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.059: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e51: type = facet_type <@ImplicitAs, @ImplicitAs(%X)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.952: <witness> = impl_witness @i32.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %.8fb: Core.Form = init_form %X [concrete]
// CHECK:STDOUT:   %pattern_type.a05: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b33: %ImplicitAs.type.e51 = facet_value %i32, (%ImplicitAs.impl_witness.952) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cb0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%X, %ImplicitAs.facet.b33) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.043: <witness> = impl_witness @X.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.type: type = fn_type @X.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert: %X.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f17: %ImplicitAs.type.544 = facet_value %X, (%ImplicitAs.impl_witness.043) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.caa: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.f17) [concrete]
// CHECK:STDOUT:   %Sink_i32.type: type = fn_type @Sink_i32 [concrete]
// CHECK:STDOUT:   %Sink_i32: %Sink_i32.type = struct_value () [concrete]
// CHECK:STDOUT:   %Sink_X.type: type = fn_type @Sink_X [concrete]
// CHECK:STDOUT:   %Sink_X: %Sink_X.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.184347.2: Core.Form = init_form %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %Source.type: type = fn_type @Source [concrete]
// CHECK:STDOUT:   %Source: %Source.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Source.specific_fn.e48: <specific function> = specific_function %Source, @Source(%T.67d) [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %Source.specific_fn.6df: <specific function> = specific_function %Source, @Source(%X) [concrete]
// CHECK:STDOUT:   %.91c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.caa, %ImplicitAs.facet.f17 [concrete]
// CHECK:STDOUT:   %Source.specific_fn.5a8: <specific function> = specific_function %Source, @Source(%i32) [concrete]
// CHECK:STDOUT:   %.b0e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cb0, %ImplicitAs.facet.b33 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc35_20.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Sink_i32 = %Sink_i32.decl
// CHECK:STDOUT:     .Sink_X = %Sink_X.decl
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @i32.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.649 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%X)> [concrete = constants.%ImplicitAs.type.e51]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.649 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [concrete = constants.%ImplicitAs.type.544]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_i32.decl: %Sink_i32.type = fn_decl @Sink_i32 [concrete = constants.%Sink_i32] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.c6b = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_X.decl: %Sink_X.type = fn_decl @Sink_X [concrete = constants.%Sink_X] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a05 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.a05 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: %Source.type = fn_decl @Source [concrete = constants.%Source] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @Source.%pattern_type (%pattern_type.51d1c4.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Source.%pattern_type (%pattern_type.51d1c4.2) = return_slot_pattern %return.param_patt, %T.ref.loc31_24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc31_24: type = name_ref T, %T.loc31_12.2 [symbolic = %T.loc31_12.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc31_24.3: Core.Form = init_form %T.ref.loc31_24 [symbolic = %.loc31_24.2 (constants.%.184347.2)]
// CHECK:STDOUT:     %.loc31_15.1: type = splice_block %.loc31_15.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc31_15.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_12.2: type = symbolic_binding T, 0 [symbolic = %T.loc31_12.1 (constants.%T.67d)]
// CHECK:STDOUT:     %return.param: ref @Source.%T.loc31_12.1 (%T.67d) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Source.%T.loc31_12.1 (%T.67d) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.ImplicitAs.impl: %i32 as %ImplicitAs.type {
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.decl: %i32.as.ImplicitAs.impl.Convert.type = fn_decl @i32.as.ImplicitAs.impl.Convert [concrete = constants.%i32.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.c6b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a05 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a05 = return_slot_pattern %return.param_patt, %X.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc20_30: Core.Form = init_form %X.ref [concrete = constants.%.8fb]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self: %i32 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param1
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%i32.as.ImplicitAs.impl.Convert.decl), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.952]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Convert = %i32.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.ImplicitAs.impl: %X.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.decl: %X.as.ImplicitAs.impl.Convert.type = fn_decl @X.as.ImplicitAs.impl.Convert [concrete = constants.%X.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a05 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a05 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc24_28: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%X.as.ImplicitAs.impl.Convert.decl), @X.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.043]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Convert = %X.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc16: %X.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.059]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.ImplicitAs.impl.Convert(%self.param: %i32) -> out %return.param: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %i32 = name_ref self, %self
// CHECK:STDOUT:   %.loc20_51.1: %struct_type.n = struct_literal (%self.ref)
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc20_47.1: <bound method> = bound_method %self.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_47.2: <bound method> = bound_method %self.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc20_47.2(%self.ref)
// CHECK:STDOUT:   %.loc20_51.2: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc20_51.3: init %i32 to %.loc20_51.2 = in_place_init %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc20_51.4: init %X to %return.param = class_init (%.loc20_51.3)
// CHECK:STDOUT:   %.loc20_52: init %X = converted %.loc20_51.1, %.loc20_51.4
// CHECK:STDOUT:   return %.loc20_52 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.ImplicitAs.impl.Convert(%self.param: %X) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %X.elem = name_ref n, @X.%.loc16 [concrete = @X.%.loc16]
// CHECK:STDOUT:   %.loc24_45.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc24_45.2: %i32 = acquire_value %.loc24_45.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc24_45.1: <bound method> = bound_method %.loc24_45.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_45.2: <bound method> = bound_method %.loc24_45.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc24_45.2(%.loc24_45.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_i32(%n.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_X(%x.param: %X);
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Source(%T.loc31_12.2: type) {
// CHECK:STDOUT:   %T.loc31_12.1: type = symbolic_binding T, 0 [symbolic = %T.loc31_12.1 (constants.%T.67d)]
// CHECK:STDOUT:   %.loc31_24.2: Core.Form = init_form %T.loc31_12.1 [symbolic = %.loc31_24.2 (constants.%.184347.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc31_12.1 [symbolic = %pattern_type (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc31_12.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2: <specific function> = specific_function constants.%Source, @Source(%T.loc31_12.1) [symbolic = %Source.specific_fn.loc31_35.2 (constants.%Source.specific_fn.e48)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Source.%T.loc31_12.1 (%T.67d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Source.ref: %Source.type = name_ref Source, file.%Source.decl [concrete = constants.%Source]
// CHECK:STDOUT:     %T.ref.loc31_42: type = name_ref T, %T.loc31_12.2 [symbolic = %T.loc31_12.1 (constants.%T.67d)]
// CHECK:STDOUT:     %Source.specific_fn.loc31_35.1: <specific function> = specific_function %Source.ref, @Source(constants.%T.67d) [symbolic = %Source.specific_fn.loc31_35.2 (constants.%Source.specific_fn.e48)]
// CHECK:STDOUT:     %.loc31_24.1: ref @Source.%T.loc31_12.1 (%T.67d) = splice_block %return.param {}
// CHECK:STDOUT:     %Source.call: init @Source.%T.loc31_12.1 (%T.67d) to %.loc31_24.1 = call %Source.specific_fn.loc31_35.1()
// CHECK:STDOUT:     return %Source.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sink_i32.ref: %Sink_i32.type = name_ref Sink_i32, file.%Sink_i32.decl [concrete = constants.%Sink_i32]
// CHECK:STDOUT:   %Source.ref.loc34: %Source.type = name_ref Source, file.%Source.decl [concrete = constants.%Source]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %Source.specific_fn.loc34: <specific function> = specific_function %Source.ref.loc34, @Source(constants.%X) [concrete = constants.%Source.specific_fn.6df]
// CHECK:STDOUT:   %.loc34_20.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Source.call.loc34: init %X to %.loc34_20.1 = call %Source.specific_fn.loc34()
// CHECK:STDOUT:   %impl.elem0.loc34: %.91c = impl_witness_access constants.%ImplicitAs.impl_witness.043, element0 [concrete = constants.%X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %Source.call.loc34, %impl.elem0.loc34
// CHECK:STDOUT:   %.loc34_20.2: ref %X = temporary %.loc34_20.1, %Source.call.loc34
// CHECK:STDOUT:   %.loc34_20.3: %X = acquire_value %.loc34_20.2
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc34(%.loc34_20.3)
// CHECK:STDOUT:   %.loc34_20.4: %i32 = value_of_initializer %X.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc34_20.5: %i32 = converted %Source.call.loc34, %.loc34_20.4
// CHECK:STDOUT:   %Sink_i32.call: init %empty_tuple.type = call %Sink_i32.ref(%.loc34_20.5)
// CHECK:STDOUT:   %Sink_X.ref: %Sink_X.type = name_ref Sink_X, file.%Sink_X.decl [concrete = constants.%Sink_X]
// CHECK:STDOUT:   %Source.ref.loc35: %Source.type = name_ref Source, file.%Source.decl [concrete = constants.%Source]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Source.specific_fn.loc35: <specific function> = specific_function %Source.ref.loc35, @Source(constants.%i32) [concrete = constants.%Source.specific_fn.5a8]
// CHECK:STDOUT:   %Source.call.loc35: init %i32 = call %Source.specific_fn.loc35()
// CHECK:STDOUT:   %impl.elem0.loc35: %.b0e = impl_witness_access constants.%ImplicitAs.impl_witness.952, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc35: <bound method> = bound_method %Source.call.loc35, %impl.elem0.loc35
// CHECK:STDOUT:   %.loc35_20.1: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc35_20.2: %i32 = value_of_initializer %Source.call.loc35
// CHECK:STDOUT:   %.loc35_20.3: %i32 = converted %Source.call.loc35, %.loc35_20.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call: init %X to %.loc35_20.1 = call %bound_method.loc35(%.loc35_20.3)
// CHECK:STDOUT:   %.loc35_20.4: init %X = converted %Source.call.loc35, %i32.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc35_20.5: ref %X = temporary %.loc35_20.1, %.loc35_20.4
// CHECK:STDOUT:   %.loc35_20.6: %X = acquire_value %.loc35_20.5
// CHECK:STDOUT:   %Sink_X.call: init %empty_tuple.type = call %Sink_X.ref(%.loc35_20.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc35: <bound method> = bound_method %.loc35_20.5, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc35: init %empty_tuple.type = call %Destroy.Op.bound.loc35(%.loc35_20.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc34: <bound method> = bound_method %.loc34_20.2, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc34: init %empty_tuple.type = call %Destroy.Op.bound.loc34(%.loc34_20.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_20.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_20.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_20.3(%self.param: ref %struct_type.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_20.4(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc31_12.1 => constants.%T.67d
// CHECK:STDOUT:   %.loc31_24.2 => constants.%.184347.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2 => constants.%Source.specific_fn.e48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%X) {
// CHECK:STDOUT:   %T.loc31_12.1 => constants.%X
// CHECK:STDOUT:   %.loc31_24.2 => constants.%.8fb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a05
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.059
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2 => constants.%Source.specific_fn.6df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%i32) {
// CHECK:STDOUT:   %T.loc31_12.1 => constants.%i32
// CHECK:STDOUT:   %.loc31_24.2 => constants.%.70c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c6b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2 => constants.%Source.specific_fn.5a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
