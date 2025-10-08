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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.179: type = facet_type <@ImplicitAs, @ImplicitAs(%X)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.665: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%X) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8b6: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc19 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d14: %ImplicitAs.type.179 = facet_value %i32, (%ImplicitAs.impl_witness.8b6) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cbe: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc23 [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.type: type = fn_type @X.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert: %X.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.81e: %ImplicitAs.type.d14 = facet_value %X, (%ImplicitAs.impl_witness.cbe) [concrete]
// CHECK:STDOUT:   %Sink_i32.type: type = fn_type @Sink_i32 [concrete]
// CHECK:STDOUT:   %Sink_i32: %Sink_i32.type = struct_value () [concrete]
// CHECK:STDOUT:   %Sink_X.type: type = fn_type @Sink_X [concrete]
// CHECK:STDOUT:   %Sink_X: %Sink_X.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Source.type: type = fn_type @Source [concrete]
// CHECK:STDOUT:   %Source: %Source.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Source.specific_fn.ff0: <specific function> = specific_function %Source, @Source(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %Source.specific_fn.363: <specific function> = specific_function %Source, @Source(%X) [concrete]
// CHECK:STDOUT:   %.6e6: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.81e [concrete]
// CHECK:STDOUT:   %Source.specific_fn.cc7: <specific function> = specific_function %Source, @Source(%i32) [concrete]
// CHECK:STDOUT:   %.c6c: type = fn_type_with_self_type %ImplicitAs.Convert.type.665, %ImplicitAs.facet.d14 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b2e: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.756: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b2e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.756, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%X)> [concrete = constants.%ImplicitAs.type.179]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc19 = impl_witness_table (@i32.as.ImplicitAs.impl.%i32.as.ImplicitAs.impl.Convert.decl), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc19: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc19 [concrete = constants.%ImplicitAs.impl_witness.8b6]
// CHECK:STDOUT:   impl_decl @X.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [concrete = constants.%ImplicitAs.type.d14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc23 = impl_witness_table (@X.as.ImplicitAs.impl.%X.as.ImplicitAs.impl.Convert.decl), @X.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc23: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc23 [concrete = constants.%ImplicitAs.impl_witness.cbe]
// CHECK:STDOUT:   %Sink_i32.decl: %Sink_i32.type = fn_decl @Sink_i32 [concrete = constants.%Sink_i32] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc29: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sink_X.decl: %Sink_X.type = fn_decl @Sink_X [concrete = constants.%Sink_X] {
// CHECK:STDOUT:     %x.patt: %pattern_type.019 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.019 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: %Source.type = fn_decl @Source [concrete = constants.%Source] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: @Source.%pattern_type (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Source.%pattern_type (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc31_24: type = name_ref T, %T.loc31_11.2 [symbolic = %T.loc31_11.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc31_11.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc31_11.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %return.param: ref @Source.%T.loc31_11.1 (%T.8b3) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Source.%T.loc31_11.1 (%T.8b3) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.ImplicitAs.impl: %i32 as %ImplicitAs.type {
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.decl: %i32.as.ImplicitAs.impl.Convert.type = fn_decl @i32.as.ImplicitAs.impl.Convert [concrete = constants.%i32.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7ce = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ce = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.019 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.019 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc20_20: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param1
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Convert = %i32.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.ImplicitAs.impl: %X.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.decl: %X.as.ImplicitAs.impl.Convert.type = fn_decl @X.as.ImplicitAs.impl.Convert [concrete = constants.%X.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.019 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.019 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Convert = %X.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %X.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.ImplicitAs.impl.Convert(%self.param: %i32) -> %return.param: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %i32 = name_ref self, %self
// CHECK:STDOUT:   %.loc20_51.1: %struct_type.n = struct_literal (%self.ref)
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc20_47.1: <bound method> = bound_method %self.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_47.2: <bound method> = bound_method %self.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc20_47.2(%self.ref)
// CHECK:STDOUT:   %.loc20_51.2: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc20_51.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call to %.loc20_51.2
// CHECK:STDOUT:   %.loc20_51.4: init %X = class_init (%.loc20_51.3), %return
// CHECK:STDOUT:   %.loc20_52: init %X = converted %.loc20_51.1, %.loc20_51.4
// CHECK:STDOUT:   return %.loc20_52 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.ImplicitAs.impl.Convert(%self.param: %X) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %X.elem = name_ref n, @X.%.loc16 [concrete = @X.%.loc16]
// CHECK:STDOUT:   %.loc24_45.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc24_45.2: %i32 = bind_value %.loc24_45.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc24_45.1: <bound method> = bound_method %.loc24_45.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_45.2: <bound method> = bound_method %.loc24_45.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc24_45.2(%.loc24_45.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_i32(%n.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sink_X(%x.param: %X);
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Source(%T.loc31_11.2: type) {
// CHECK:STDOUT:   %T.loc31_11.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc31_11.1 (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc31_11.1 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc31_11.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2: <specific function> = specific_function constants.%Source, @Source(%T.loc31_11.1) [symbolic = %Source.specific_fn.loc31_35.2 (constants.%Source.specific_fn.ff0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Source.%T.loc31_11.1 (%T.8b3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Source.ref: %Source.type = name_ref Source, file.%Source.decl [concrete = constants.%Source]
// CHECK:STDOUT:     %T.ref.loc31_42: type = name_ref T, %T.loc31_11.2 [symbolic = %T.loc31_11.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %Source.specific_fn.loc31_35.1: <specific function> = specific_function %Source.ref, @Source(constants.%T.8b3) [symbolic = %Source.specific_fn.loc31_35.2 (constants.%Source.specific_fn.ff0)]
// CHECK:STDOUT:     %.loc31: ref @Source.%T.loc31_11.1 (%T.8b3) = splice_block %return {}
// CHECK:STDOUT:     %Source.call: init @Source.%T.loc31_11.1 (%T.8b3) = call %Source.specific_fn.loc31_35.1() to %.loc31
// CHECK:STDOUT:     return %Source.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sink_i32.ref: %Sink_i32.type = name_ref Sink_i32, file.%Sink_i32.decl [concrete = constants.%Sink_i32]
// CHECK:STDOUT:   %Source.ref.loc34: %Source.type = name_ref Source, file.%Source.decl [concrete = constants.%Source]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %Source.specific_fn.loc34: <specific function> = specific_function %Source.ref.loc34, @Source(constants.%X) [concrete = constants.%Source.specific_fn.363]
// CHECK:STDOUT:   %.loc34_20.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Source.call.loc34: init %X = call %Source.specific_fn.loc34() to %.loc34_20.1
// CHECK:STDOUT:   %impl.elem0.loc34: %.6e6 = impl_witness_access constants.%ImplicitAs.impl_witness.cbe, element0 [concrete = constants.%X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc34_20.1: <bound method> = bound_method %Source.call.loc34, %impl.elem0.loc34
// CHECK:STDOUT:   %.loc34_20.2: ref %X = temporary %.loc34_20.1, %Source.call.loc34
// CHECK:STDOUT:   %.loc34_20.3: %X = bind_value %.loc34_20.2
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc34_20.1(%.loc34_20.3)
// CHECK:STDOUT:   %.loc34_20.4: %i32 = value_of_initializer %X.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc34_20.5: %i32 = converted %Source.call.loc34, %.loc34_20.4
// CHECK:STDOUT:   %Sink_i32.call: init %empty_tuple.type = call %Sink_i32.ref(%.loc34_20.5)
// CHECK:STDOUT:   %Sink_X.ref: %Sink_X.type = name_ref Sink_X, file.%Sink_X.decl [concrete = constants.%Sink_X]
// CHECK:STDOUT:   %Source.ref.loc35: %Source.type = name_ref Source, file.%Source.decl [concrete = constants.%Source]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Source.specific_fn.loc35: <specific function> = specific_function %Source.ref.loc35, @Source(constants.%i32) [concrete = constants.%Source.specific_fn.cc7]
// CHECK:STDOUT:   %Source.call.loc35: init %i32 = call %Source.specific_fn.loc35()
// CHECK:STDOUT:   %impl.elem0.loc35: %.c6c = impl_witness_access constants.%ImplicitAs.impl_witness.8b6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc35_20.1: <bound method> = bound_method %Source.call.loc35, %impl.elem0.loc35
// CHECK:STDOUT:   %.loc35_20.1: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc35_20.2: %i32 = value_of_initializer %Source.call.loc35
// CHECK:STDOUT:   %.loc35_20.3: %i32 = converted %Source.call.loc35, %.loc35_20.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call: init %X = call %bound_method.loc35_20.1(%.loc35_20.3) to %.loc35_20.1
// CHECK:STDOUT:   %.loc35_20.4: init %X = converted %Source.call.loc35, %i32.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc35_20.5: ref %X = temporary %.loc35_20.1, %.loc35_20.4
// CHECK:STDOUT:   %.loc35_20.6: %X = bind_value %.loc35_20.5
// CHECK:STDOUT:   %Sink_X.call: init %empty_tuple.type = call %Sink_X.ref(%.loc35_20.6)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc35: <bound method> = bound_method %.loc35_20.5, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.756
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.756, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_20.2: <bound method> = bound_method %.loc35_20.5, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc35: %ptr.d17 = addr_of %.loc35_20.5
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_20.2(%addr.loc35)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc34: <bound method> = bound_method %.loc34_20.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.756
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.756, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_20.2: <bound method> = bound_method %.loc34_20.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc34: %ptr.d17 = addr_of %.loc34_20.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_20.2(%addr.loc34)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc31_11.1 => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2 => constants.%Source.specific_fn.ff0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%X) {
// CHECK:STDOUT:   %T.loc31_11.1 => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.019
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.54b
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2 => constants.%Source.specific_fn.363
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Source(constants.%i32) {
// CHECK:STDOUT:   %T.loc31_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Source.specific_fn.loc31_35.2 => constants.%Source.specific_fn.cc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
