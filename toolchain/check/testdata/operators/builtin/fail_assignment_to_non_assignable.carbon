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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_non_assignable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_non_assignable.carbon

fn F() -> i32;

fn Main() {
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-5: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-8: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-8: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-5: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-18: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-25: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+2]]:3-25: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:
  (if true then a else a) = 10;
}

// CHECK:STDOUT: --- fail_assignment_to_non_assignable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0b3: <witness> = impl_witness imports.%Copy.impl_witness_table.ee5 [concrete]
// CHECK:STDOUT:   %Copy.facet.d66: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.0b3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.85c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d66) [concrete]
// CHECK:STDOUT:   %.af5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.85c, %Copy.facet.d66 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.e21: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.64e: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.50f: <bound method> = bound_method %int_4, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.6b6 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.6b6 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.102: %tuple.type.e55 = tuple_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c99: <witness> = impl_witness imports.%Copy.impl_witness_table.7b6 [concrete]
// CHECK:STDOUT:   %Copy.facet.95f: %Copy.type = facet_value type, (%Copy.impl_witness.c99) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ac1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95f) [concrete]
// CHECK:STDOUT:   %.224: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ac1, %Copy.facet.95f [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %ptr.d08, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct.a0d: %struct_type.x.y = struct_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.6b6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.850: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.e40: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.81c: %T.as.DefaultOrUnformed.impl.Op.type.e40 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %i32, (%DefaultOrUnformed.impl_witness.850) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.81c, @T.as.DefaultOrUnformed.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcb: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.1b6: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.6bf: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc42_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ee5 = impl_witness_table (%Core.import_ref.9c3475.1), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.7b6 = impl_witness_table (%Core.import_ref.9c3475.2), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.856 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc20: %.af5 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %int_2.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.e21]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc20: init Core.IntLiteral = call %bound_method.loc20(%int_2.loc20) [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   assign %int_1.loc20, %Core.IntLiteral.as.Copy.impl.Op.call.loc20
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc23: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc23_7.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_7.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_7.2(%int_1.loc23) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc23: init %i32 = converted %int_1.loc23, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   assign %F.call, %.loc23
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc26: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc26_8.1: %tuple.type.f94 = tuple_literal (%int_1.loc26, %int_2.loc26) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %int_3.loc26: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc26: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc26_17.1: %tuple.type.f94 = tuple_literal (%int_3.loc26, %int_4.loc26) [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %impl.elem0.loc26_13: %.af5 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc26_13: <bound method> = bound_method %int_3.loc26, %impl.elem0.loc26_13 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.64e]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc26_13: init Core.IntLiteral = call %bound_method.loc26_13(%int_3.loc26) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %tuple.elem0.loc26: Core.IntLiteral = tuple_access %.loc26_8.1, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc26_17.2: init Core.IntLiteral to %tuple.elem0.loc26 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc26_13 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc26_16: %.af5 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc26_16: <bound method> = bound_method %int_4.loc26, %impl.elem0.loc26_16 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.50f]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc26_16: init Core.IntLiteral = call %bound_method.loc26_16(%int_4.loc26) [concrete = constants.%int_4]
// CHECK:STDOUT:   %tuple.elem1.loc26: Core.IntLiteral = tuple_access %.loc26_8.1, element1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc26_17.3: init Core.IntLiteral to %tuple.elem1.loc26 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc26_16 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc26_17.4: init %tuple.type.f94 to %.loc26_8.1 = tuple_init (%.loc26_17.2, %.loc26_17.3) [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %.loc26_10: init %tuple.type.f94 = converted %.loc26_17.1, %.loc26_17.4 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   assign %.loc26_8.1, %.loc26_10
// CHECK:STDOUT:   %tuple.loc26: %tuple.type.f94 = tuple_value (%int_1.loc26, %int_2.loc26) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc26_8.2: %tuple.type.f94 = converted %.loc26_8.1, %tuple.loc26 [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %n.var: ref %i32 = var_storage %n.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc27: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc27_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_3.2: <bound method> = bound_method %int_0, %specific_fn.loc27 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27: init %i32 = call %bound_method.loc27_3.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc27: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   assign %n.var, %.loc27
// CHECK:STDOUT:   %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: ref %i32 = wrapper_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = ref_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.6b6 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.ref.loc30_4: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc30_7: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc30_8.1: %tuple.type.e55 = tuple_literal (%n.ref.loc30_4, %n.ref.loc30_7)
// CHECK:STDOUT:   %int_1.loc30: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc30_17.1: %tuple.type.f94 = tuple_literal (%int_1.loc30, %int_2.loc30) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc30_17.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc30_17.1: <bound method> = bound_method %int_1.loc30, %impl.elem0.loc30_17.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc30_17.1: <specific function> = specific_function %impl.elem0.loc30_17.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_17.2: <bound method> = bound_method %int_1.loc30, %specific_fn.loc30_17.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30_17.1: init %i32 = call %bound_method.loc30_17.2(%int_1.loc30) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc30_17.2: init %i32 = converted %int_1.loc30, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30_17.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %tuple.elem0.loc30: %i32 = tuple_access %.loc30_8.1, element0
// CHECK:STDOUT:   %.loc30_17.3: init %i32 to %tuple.elem0.loc30 = in_place_init %.loc30_17.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc30_17.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc30_17.3: <bound method> = bound_method %int_2.loc30, %impl.elem0.loc30_17.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc30_17.2: <specific function> = specific_function %impl.elem0.loc30_17.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_17.4: <bound method> = bound_method %int_2.loc30, %specific_fn.loc30_17.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30_17.2: init %i32 = call %bound_method.loc30_17.4(%int_2.loc30) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc30_17.4: init %i32 = converted %int_2.loc30, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30_17.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %tuple.elem1.loc30: %i32 = tuple_access %.loc30_8.1, element1
// CHECK:STDOUT:   %.loc30_17.5: init %i32 to %tuple.elem1.loc30 = in_place_init %.loc30_17.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc30_17.6: init %tuple.type.e55 to %.loc30_8.1 = tuple_init (%.loc30_17.3, %.loc30_17.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:   %.loc30_10: init %tuple.type.e55 = converted %.loc30_17.1, %.loc30_17.6 [concrete = constants.%tuple.102]
// CHECK:STDOUT:   assign %.loc30_8.1, %.loc30_10
// CHECK:STDOUT:   %.loc30_4: %i32 = acquire_value %n.ref.loc30_4
// CHECK:STDOUT:   %.loc30_7: %i32 = acquire_value %n.ref.loc30_7
// CHECK:STDOUT:   %tuple.loc30: %tuple.type.e55 = tuple_value (%.loc30_4, %.loc30_7)
// CHECK:STDOUT:   %.loc30_8.2: %tuple.type.e55 = converted %.loc30_8.1, %tuple.loc30
// CHECK:STDOUT:   %i32.loc33_3: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc33_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32.loc33_9 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   %impl.elem0.loc33: %.224 = impl_witness_access constants.%Copy.impl_witness.c99, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc33: <bound method> = bound_method %ptr, %impl.elem0.loc33 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method.loc33(%ptr) [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   assign %i32.loc33_3, %type.as.Copy.impl.Op.call
// CHECK:STDOUT:   %int_1.loc36: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc36: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc36_18.1: %struct_type.x.y = struct_literal (%int_1.loc36, %int_2.loc36) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %int_3.loc36: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc36: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc36_37.1: %struct_type.x.y = struct_literal (%int_3.loc36, %int_4.loc36) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %impl.elem0.loc36_28: %.af5 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc36_28: <bound method> = bound_method %int_3.loc36, %impl.elem0.loc36_28 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.64e]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc36_28: init Core.IntLiteral = call %bound_method.loc36_28(%int_3.loc36) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc36_37.2: Core.IntLiteral = struct_access %.loc36_18.1, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc36_37.3: init Core.IntLiteral to %.loc36_37.2 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc36_28 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc36_36: %.af5 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc36_36: <bound method> = bound_method %int_4.loc36, %impl.elem0.loc36_36 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.50f]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc36_36: init Core.IntLiteral = call %bound_method.loc36_36(%int_4.loc36) [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc36_37.4: Core.IntLiteral = struct_access %.loc36_18.1, element1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc36_37.5: init Core.IntLiteral to %.loc36_37.4 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc36_36 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc36_37.6: init %struct_type.x.y to %.loc36_18.1 = struct_init (%.loc36_37.3, %.loc36_37.5) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %.loc36_20: init %struct_type.x.y = converted %.loc36_37.1, %.loc36_37.6 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   assign %.loc36_18.1, %.loc36_20
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%int_1.loc36, %int_2.loc36) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc36_18.2: %struct_type.x.y = converted %.loc36_18.1, %struct [concrete = constants.%struct.004]
// CHECK:STDOUT:   %true.loc39: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc39 br !if.expr.then.loc39 else br !if.expr.else.loc39
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc39:
// CHECK:STDOUT:   %int_1.loc39: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc39_12: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc39_12.1: <bound method> = bound_method %int_1.loc39, %impl.elem0.loc39_12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc39_12: <specific function> = specific_function %impl.elem0.loc39_12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_12.2: <bound method> = bound_method %int_1.loc39, %specific_fn.loc39_12 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39_12: init %i32 = call %bound_method.loc39_12.2(%int_1.loc39) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc39_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39_12 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc39_12.2: %i32 = converted %int_1.loc39, %.loc39_12.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   br !if.expr.result.loc39(%.loc39_12.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc39:
// CHECK:STDOUT:   %int_2.loc39: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc39_19: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc39_19.1: <bound method> = bound_method %int_2.loc39, %impl.elem0.loc39_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc39_19: <specific function> = specific_function %impl.elem0.loc39_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_19.2: <bound method> = bound_method %int_2.loc39, %specific_fn.loc39_19 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39_19: init %i32 = call %bound_method.loc39_19.2(%int_2.loc39) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc39_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39_19 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc39_19.2: %i32 = converted %int_2.loc39, %.loc39_19.1 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   br !if.expr.result.loc39(%.loc39_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc39:
// CHECK:STDOUT:   %.loc39_4: %i32 = block_arg !if.expr.result.loc39 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %int_3.loc39: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc39_27: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc39_27.1: <bound method> = bound_method %int_3.loc39, %impl.elem0.loc39_27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc39_27: <specific function> = specific_function %impl.elem0.loc39_27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_27.2: <bound method> = bound_method %int_3.loc39, %specific_fn.loc39_27 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39_27: init %i32 = call %bound_method.loc39_27.2(%int_3.loc39) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc39_27: init %i32 = converted %int_3.loc39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39_27 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   assign %.loc39_4, %.loc39_27
// CHECK:STDOUT:   %a.var: ref %i32 = var_storage %a.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%i32, (constants.%DefaultOrUnformed.impl_witness.850) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc42_13.1: %DefaultOrUnformed.type = converted constants.%i32, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc42_13.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc42_13.2: type = converted %.loc42_13.1, %as_type [concrete = constants.%i32]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.81c, @T.as.DefaultOrUnformed.impl.Op(constants.%i32) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %i32 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %a.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %i32.loc42: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6b6 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %true.loc45: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc45 br !if.expr.then.loc45 else br !if.expr.else.loc45
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc45:
// CHECK:STDOUT:   %a.ref.loc45_17: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc45_17: %i32 = acquire_value %a.ref.loc45_17
// CHECK:STDOUT:   br !if.expr.result.loc45(%.loc45_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc45:
// CHECK:STDOUT:   %a.ref.loc45_24: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc45_24: %i32 = acquire_value %a.ref.loc45_24
// CHECK:STDOUT:   br !if.expr.result.loc45(%.loc45_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc45:
// CHECK:STDOUT:   %.loc45_4: %i32 = block_arg !if.expr.result.loc45
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc45: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc45_27.1: <bound method> = bound_method %int_10, %impl.elem0.loc45 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcb]
// CHECK:STDOUT:   %specific_fn.loc45: <specific function> = specific_function %impl.elem0.loc45, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_27.2: <bound method> = bound_method %int_10, %specific_fn.loc45 [concrete = constants.%bound_method.1b6]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc45: init %i32 = call %bound_method.loc45_27.2(%int_10) [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   %.loc45_27: init %i32 = converted %int_10, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc45 [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   assign %.loc45_4, %.loc45_27
// CHECK:STDOUT:   %Destroy.Op.bound.loc42: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc42: init %empty_tuple.type = call %Destroy.Op.bound.loc42(%a.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27: <bound method> = bound_method %n.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc27: init %empty_tuple.type = call %Destroy.Op.bound.loc27(%n.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc42_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc42_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
