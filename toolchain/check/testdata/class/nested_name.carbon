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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/nested_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/nested_name.carbon

class Outer {
  class Inner {
    var n: i32;
  }
}

fn F(oi: Outer.Inner) -> i32 {
  return oi.n;
}

fn G(o: Outer) {
  var unused i: o.Inner;
}

// CHECK:STDOUT: --- nested_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.dc4: type = pattern_type %Inner [concrete]
// CHECK:STDOUT:   %oi.param_patt: %pattern_type.dc4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %oi.patt: %pattern_type.dc4 = at_binding_pattern oi, %oi.param_patt [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d6c: type = pattern_type %Outer [concrete]
// CHECK:STDOUT:   %o.param_patt: %pattern_type.d6c = value_param_pattern [concrete]
// CHECK:STDOUT:   %o.patt: %pattern_type.d6c = at_binding_pattern o, %o.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.dc4 = ref_binding_pattern i [concrete]
// CHECK:STDOUT:   %i.var_patt: %pattern_type.dc4 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.b1a: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.374, @T.as.DefaultOrUnformed.impl(%Inner) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.dd7: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Inner) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.0e7: %T.as.DefaultOrUnformed.impl.Op.type.dd7 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Inner, (%DefaultOrUnformed.impl_witness.b1a) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.0e7, @T.as.DefaultOrUnformed.impl.Op(%Inner) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc26_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.374 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [concrete = constants.%Outer] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %oi.param_patt: %pattern_type.dc4 = value_param_pattern [concrete = constants.%oi.param_patt]
// CHECK:STDOUT:     %oi.patt: %pattern_type.dc4 = at_binding_pattern oi, %oi.param_patt [concrete = constants.%oi.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc21_26: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %oi.param: %Inner = value_param call_param0
// CHECK:STDOUT:     %.loc21_15: type = splice_block %Inner.ref [concrete = constants.%Inner] {
// CHECK:STDOUT:       %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %oi: %Inner = value_binding oi, %oi.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %o.param_patt: %pattern_type.d6c = value_param_pattern [concrete = constants.%o.param_patt]
// CHECK:STDOUT:     %o.patt: %pattern_type.d6c = at_binding_pattern o, %o.param_patt [concrete = constants.%o.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %o.param: %Outer = value_param call_param0
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %o: %Outer = value_binding o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc17: %Inner.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.cdf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .n = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%oi.param: %Inner) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %oi.ref: %Inner = name_ref oi, %oi
// CHECK:STDOUT:   %n.ref: %Inner.elem = name_ref n, @Inner.%.loc17 [concrete = @Inner.%.loc17]
// CHECK:STDOUT:   %.loc22_12.1: ref %i32 = class_element_access %oi.ref, element0
// CHECK:STDOUT:   %.loc22_12.2: %i32 = acquire_value %.loc22_12.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc22_12.1: <bound method> = bound_method %.loc22_12.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_12.2: <bound method> = bound_method %.loc22_12.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_12.2(%.loc22_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%o.param: %Outer) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.dc4 = ref_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.dc4 = var_pattern %i.patt [concrete = constants.%i.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.b1a) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc26_24.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc26_24.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc26_24.2: type = converted %.loc26_24.1, %as_type [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.0e7, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc26_3: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Inner to %.loc26_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc26_18: type = splice_block %Inner.ref [concrete = constants.%Inner] {
// CHECK:STDOUT:     %o.ref: %Outer = name_ref o, %o
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %i.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%i.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.3(%self.param: ref %struct_type.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.4(%self.param: ref %Inner) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
