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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/raw_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/raw_self.carbon

class Class {
  fn F[ref self: Self](r#self: i32);
  fn G[self: Self](r#self: i32) -> (i32, i32);
  var n: i32;
}

fn Class.F[ref self: Self](r#self: i32) {
  self.n = r#self;
}

fn Class.G[self: Self](r#self: i32) -> (i32, i32) {
  return (self.n, r#self);
}

// CHECK:STDOUT: --- raw_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.904: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.95a: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %.f32: Core.Form = init_form %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt.loc50: %pattern_type.904 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc49: %pattern_type.904 = at_binding_pattern self, %self.param_patt.loc50 [concrete]
// CHECK:STDOUT:     %self.param_patt.loc55: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc54: %pattern_type.7ce = at_binding_pattern r#self, %self.param_patt.loc55 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc50: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc51: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc49: ref %Class = ref_binding self, %self.param.loc50
// CHECK:STDOUT:     %self.param.loc55: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc56: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self.loc54: %i32 = value_binding r#self, %self.param.loc55
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt.loc72: %pattern_type.904 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc71: %pattern_type.904 = at_binding_pattern self, %self.param_patt.loc72 [concrete]
// CHECK:STDOUT:     %self.param_patt.loc77: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc76: %pattern_type.7ce = at_binding_pattern r#self, %self.param_patt.loc77 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.511 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.511 = return_slot_pattern %return.param_patt, %.loc85.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc82: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc84: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc85.1: %tuple.type.24b = tuple_literal (%i32.loc82, %i32.loc84) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:     %.loc85.2: type = converted %.loc85.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %.loc85.3: Core.Form = init_form %.loc85.2 [concrete = constants.%.f32]
// CHECK:STDOUT:     %self.param.loc72: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc73: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc71: %Class = value_binding self, %self.param.loc72
// CHECK:STDOUT:     %self.param.loc77: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc78: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self.loc76: %i32 = value_binding r#self, %self.param.loc77
// CHECK:STDOUT:     %return.param.loc85: ref %tuple.type.d07 = out_param call_param2
// CHECK:STDOUT:     %return.loc85: ref %tuple.type.d07 = return_slot %return.param.loc85
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt.loc50: %pattern_type.904 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc49: %pattern_type.904 = at_binding_pattern self, %self.param_patt.loc50 [concrete]
// CHECK:STDOUT:     %self.param_patt.loc55: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc54: %pattern_type.7ce = at_binding_pattern r#self, %self.param_patt.loc55 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc9: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc10: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc8: ref %Class = ref_binding self, %self.param.loc9
// CHECK:STDOUT:     %self.param.loc14: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self.loc13: %i32 = value_binding r#self, %self.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt.loc72: %pattern_type.904 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc71: %pattern_type.904 = at_binding_pattern self, %self.param_patt.loc72 [concrete]
// CHECK:STDOUT:     %self.param_patt.loc77: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt.loc76: %pattern_type.7ce = at_binding_pattern r#self, %self.param_patt.loc77 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.511 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.511 = return_slot_pattern %return.param_patt, %.loc85.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc35.1: %tuple.type.24b = tuple_literal (%i32.loc32, %i32.loc34) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:     %.loc35.2: type = converted %.loc35.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %.loc35.3: Core.Form = init_form %.loc35.2 [concrete = constants.%.f32]
// CHECK:STDOUT:     %self.param.loc22: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc23: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc21: %Class = value_binding self, %self.param.loc22
// CHECK:STDOUT:     %self.param.loc27: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self.loc26: %i32 = value_binding r#self, %self.param.loc27
// CHECK:STDOUT:     %return.param.loc35: ref %tuple.type.d07 = out_param call_param2
// CHECK:STDOUT:     %return.loc35: ref %tuple.type.d07 = return_slot %return.param.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc39: %Class.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT:   .n = %.loc39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param.loc50: ref %Class, %self.param.loc55: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc59: ref %Class = name_ref self, %self.loc49
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc39 [concrete = @Class.%.loc39]
// CHECK:STDOUT:   %.loc60: ref %i32 = class_element_access %self.ref.loc59, element0
// CHECK:STDOUT:   %self.ref.loc63: %i32 = name_ref r#self, %self.loc54
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc63.1: <bound method> = bound_method %self.ref.loc63, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63.2: <bound method> = bound_method %self.ref.loc63, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc63.2(%self.ref.loc63)
// CHECK:STDOUT:   assign %.loc60, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param.loc72: %Class, %self.param.loc77: %i32) -> out %return.param.loc85: %tuple.type.d07 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc89: %Class = name_ref self, %self.loc71
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc39 [concrete = @Class.%.loc39]
// CHECK:STDOUT:   %.loc90.1: ref %i32 = class_element_access %self.ref.loc89, element0
// CHECK:STDOUT:   %.loc90.2: %i32 = acquire_value %.loc90.1
// CHECK:STDOUT:   %self.ref.loc93: %i32 = name_ref r#self, %self.loc76
// CHECK:STDOUT:   %.loc94.1: %tuple.type.d07 = tuple_literal (%.loc90.2, %self.ref.loc93)
// CHECK:STDOUT:   %impl.elem0.loc90: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc90.1: <bound method> = bound_method %.loc90.2, %impl.elem0.loc90
// CHECK:STDOUT:   %specific_fn.loc90: <specific function> = specific_function %impl.elem0.loc90, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc90.2: <bound method> = bound_method %.loc90.2, %specific_fn.loc90
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc90: init %i32 = call %bound_method.loc90.2(%.loc90.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param.loc85, element0
// CHECK:STDOUT:   %.loc94.2: init %i32 to %tuple.elem0 = in_place_init %Int.as.Copy.impl.Op.call.loc90
// CHECK:STDOUT:   %impl.elem0.loc93: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc93.1: <bound method> = bound_method %self.ref.loc93, %impl.elem0.loc93
// CHECK:STDOUT:   %specific_fn.loc93: <specific function> = specific_function %impl.elem0.loc93, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc93.2: <bound method> = bound_method %self.ref.loc93, %specific_fn.loc93
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc93: init %i32 = call %bound_method.loc93.2(%self.ref.loc93)
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param.loc85, element1
// CHECK:STDOUT:   %.loc94.3: init %i32 to %tuple.elem1 = in_place_init %Int.as.Copy.impl.Op.call.loc93
// CHECK:STDOUT:   %.loc94.4: init %tuple.type.d07 to %return.param.loc85 = tuple_init (%.loc94.2, %.loc94.3)
// CHECK:STDOUT:   %.loc95: init %tuple.type.d07 = converted %.loc94.1, %.loc94.4
// CHECK:STDOUT:   return %.loc95 to %return.param.loc85
// CHECK:STDOUT: }
// CHECK:STDOUT:
