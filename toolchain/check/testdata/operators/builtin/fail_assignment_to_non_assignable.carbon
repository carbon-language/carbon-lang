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
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ac8: <witness> = impl_witness imports.%Copy.impl_witness_table.243 [concrete]
// CHECK:STDOUT:   %Copy.facet.001: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.ac8) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.06a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.001) [concrete]
// CHECK:STDOUT:   %.538: type = fn_type_with_self_type %Copy.WithSelf.Op.type.06a, %Copy.facet.001 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.1bc: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ac5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ac5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b91, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d43: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e79: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.e73: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.5a2: <bound method> = bound_method %int_4, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.12b: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %bound_method.c19: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %bound_method.419: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.037: <witness> = impl_witness imports.%Copy.impl_witness_table.794 [concrete]
// CHECK:STDOUT:   %Copy.facet.acb: %Copy.type = facet_value type, (%Copy.impl_witness.037) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.994: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.acb) [concrete]
// CHECK:STDOUT:   %.641: type = fn_type_with_self_type %Copy.WithSelf.Op.type.994, %Copy.facet.acb [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %ptr.235, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct.a0d: %struct_type.x.y = struct_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aca: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %bound_method.5b5: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.462: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.b80: %T.as.DefaultOrUnformed.impl.Op.type.462 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.89f: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.725, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.2b0: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.3bd: %T.as.DefaultOrUnformed.impl.Op.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %i32, (%DefaultOrUnformed.impl_witness.89f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.3bd, @T.as.DefaultOrUnformed.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3f7: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %bound_method.0fa: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.265: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc56_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.37f: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.243 = impl_witness_table (%Core.import_ref.37f), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.87a: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.794 = impl_witness_table (%Core.import_ref.87a), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.dba: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.462) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.b80)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.725 = impl_witness_table (%Core.import_ref.dba), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32 [concrete = constants.%.ff5]
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
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc22: %.538 = impl_witness_access constants.%Copy.impl_witness.ac8, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.1bc]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc22: init Core.IntLiteral = call %bound_method.loc22(%int_2.loc22) [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   assign %int_1.loc22, %Core.IntLiteral.as.Copy.impl.Op.call.loc22
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc27: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc27_7.1: <bound method> = bound_method %int_1.loc27, %impl.elem0.loc27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d43]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_7.2: <bound method> = bound_method %int_1.loc27, %specific_fn.loc27 [concrete = constants.%bound_method.e79]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27: init %i32 = call %bound_method.loc27_7.2(%int_1.loc27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc27: init %i32 = converted %int_1.loc27, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %F.call, %.loc27
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc32: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32_8.1: %tuple.type.f94 = tuple_literal (%int_1.loc32, %int_2.loc32) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %int_3.loc32: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc32: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc32_17.1: %tuple.type.f94 = tuple_literal (%int_3.loc32, %int_4.loc32) [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %impl.elem0.loc32_13: %.538 = impl_witness_access constants.%Copy.impl_witness.ac8, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc32_13: <bound method> = bound_method %int_3.loc32, %impl.elem0.loc32_13 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.e73]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc32_13: init Core.IntLiteral = call %bound_method.loc32_13(%int_3.loc32) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %tuple.elem0.loc32: Core.IntLiteral = tuple_access %.loc32_8.1, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc32_17.2: init Core.IntLiteral to %tuple.elem0.loc32 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc32_13 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc32_16: %.538 = impl_witness_access constants.%Copy.impl_witness.ac8, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc32_16: <bound method> = bound_method %int_4.loc32, %impl.elem0.loc32_16 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.5a2]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc32_16: init Core.IntLiteral = call %bound_method.loc32_16(%int_4.loc32) [concrete = constants.%int_4]
// CHECK:STDOUT:   %tuple.elem1.loc32: Core.IntLiteral = tuple_access %.loc32_8.1, element1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32_17.3: init Core.IntLiteral to %tuple.elem1.loc32 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc32_16 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc32_17.4: init %tuple.type.f94 to %.loc32_8.1 = tuple_init (%.loc32_17.2, %.loc32_17.3) [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %.loc32_10: init %tuple.type.f94 = converted %.loc32_17.1, %.loc32_17.4 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   assign %.loc32_8.1, %.loc32_10
// CHECK:STDOUT:   %tuple.loc32: %tuple.type.f94 = tuple_value (%int_1.loc32, %int_2.loc32) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc32_8.2: %tuple.type.f94 = converted %.loc32_8.1, %tuple.loc32 [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc33: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc33_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.12b]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_3.2: <bound method> = bound_method %int_0, %specific_fn.loc33 [concrete = constants.%bound_method.c19]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33: init %i32 = call %bound_method.loc33_3.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc33: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc33
// CHECK:STDOUT:   %i32.loc33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: ref %i32 = ref_binding n, %n.var
// CHECK:STDOUT:   %n.ref.loc38_4: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc38_7: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc38_8.1: %tuple.type.d07 = tuple_literal (%n.ref.loc38_4, %n.ref.loc38_7)
// CHECK:STDOUT:   %int_1.loc38: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc38: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc38_17.1: %tuple.type.f94 = tuple_literal (%int_1.loc38, %int_2.loc38) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc38_17.1: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc38_17.1: <bound method> = bound_method %int_1.loc38, %impl.elem0.loc38_17.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d43]
// CHECK:STDOUT:   %specific_fn.loc38_17.1: <specific function> = specific_function %impl.elem0.loc38_17.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_17.2: <bound method> = bound_method %int_1.loc38, %specific_fn.loc38_17.1 [concrete = constants.%bound_method.e79]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc38_17.1: init %i32 = call %bound_method.loc38_17.2(%int_1.loc38) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc38_17.2: init %i32 = converted %int_1.loc38, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc38_17.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc38: %i32 = tuple_access %.loc38_8.1, element0
// CHECK:STDOUT:   %.loc38_17.3: init %i32 to %tuple.elem0.loc38 = in_place_init %.loc38_17.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc38_17.2: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc38_17.3: <bound method> = bound_method %int_2.loc38, %impl.elem0.loc38_17.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9cb]
// CHECK:STDOUT:   %specific_fn.loc38_17.2: <specific function> = specific_function %impl.elem0.loc38_17.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_17.4: <bound method> = bound_method %int_2.loc38, %specific_fn.loc38_17.2 [concrete = constants.%bound_method.419]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc38_17.2: init %i32 = call %bound_method.loc38_17.4(%int_2.loc38) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_17.4: init %i32 = converted %int_2.loc38, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc38_17.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc38: %i32 = tuple_access %.loc38_8.1, element1
// CHECK:STDOUT:   %.loc38_17.5: init %i32 to %tuple.elem1.loc38 = in_place_init %.loc38_17.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_17.6: init %tuple.type.d07 to %.loc38_8.1 = tuple_init (%.loc38_17.3, %.loc38_17.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc38_10: init %tuple.type.d07 = converted %.loc38_17.1, %.loc38_17.6 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign %.loc38_8.1, %.loc38_10
// CHECK:STDOUT:   %.loc38_4: %i32 = acquire_value %n.ref.loc38_4
// CHECK:STDOUT:   %.loc38_7: %i32 = acquire_value %n.ref.loc38_7
// CHECK:STDOUT:   %tuple.loc38: %tuple.type.d07 = tuple_value (%.loc38_4, %.loc38_7)
// CHECK:STDOUT:   %.loc38_8.2: %tuple.type.d07 = converted %.loc38_8.1, %tuple.loc38
// CHECK:STDOUT:   %i32.loc43_3: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc43_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32.loc43_9 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   %impl.elem0.loc43: %.641 = impl_witness_access constants.%Copy.impl_witness.037, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc43: <bound method> = bound_method %ptr, %impl.elem0.loc43 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method.loc43(%ptr) [concrete = constants.%ptr.235]
// CHECK:STDOUT:   assign %i32.loc43_3, %type.as.Copy.impl.Op.call
// CHECK:STDOUT:   %int_1.loc48: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc48: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc48_18.1: %struct_type.x.y = struct_literal (%int_1.loc48, %int_2.loc48) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %int_3.loc48: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc48: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc48_37.1: %struct_type.x.y = struct_literal (%int_3.loc48, %int_4.loc48) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %impl.elem0.loc48_28: %.538 = impl_witness_access constants.%Copy.impl_witness.ac8, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc48_28: <bound method> = bound_method %int_3.loc48, %impl.elem0.loc48_28 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.e73]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc48_28: init Core.IntLiteral = call %bound_method.loc48_28(%int_3.loc48) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc48_37.2: Core.IntLiteral = struct_access %.loc48_18.1, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc48_37.3: init Core.IntLiteral to %.loc48_37.2 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc48_28 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc48_36: %.538 = impl_witness_access constants.%Copy.impl_witness.ac8, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc48_36: <bound method> = bound_method %int_4.loc48, %impl.elem0.loc48_36 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.5a2]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc48_36: init Core.IntLiteral = call %bound_method.loc48_36(%int_4.loc48) [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc48_37.4: Core.IntLiteral = struct_access %.loc48_18.1, element1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc48_37.5: init Core.IntLiteral to %.loc48_37.4 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc48_36 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc48_37.6: init %struct_type.x.y to %.loc48_18.1 = struct_init (%.loc48_37.3, %.loc48_37.5) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %.loc48_20: init %struct_type.x.y = converted %.loc48_37.1, %.loc48_37.6 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   assign %.loc48_18.1, %.loc48_20
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%int_1.loc48, %int_2.loc48) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc48_18.2: %struct_type.x.y = converted %.loc48_18.1, %struct [concrete = constants.%struct.004]
// CHECK:STDOUT:   %true.loc53: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc53 br !if.expr.then.loc53 else br !if.expr.else.loc53
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc53:
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc53_12: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc53_12.1: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53_12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d43]
// CHECK:STDOUT:   %specific_fn.loc53_12: <specific function> = specific_function %impl.elem0.loc53_12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_12.2: <bound method> = bound_method %int_1.loc53, %specific_fn.loc53_12 [concrete = constants.%bound_method.e79]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_12: init %i32 = call %bound_method.loc53_12.2(%int_1.loc53) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_12 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_12.2: %i32 = converted %int_1.loc53, %.loc53_12.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   br !if.expr.result.loc53(%.loc53_12.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc53:
// CHECK:STDOUT:   %int_2.loc53: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc53_19: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc53_19.1: <bound method> = bound_method %int_2.loc53, %impl.elem0.loc53_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9cb]
// CHECK:STDOUT:   %specific_fn.loc53_19: <specific function> = specific_function %impl.elem0.loc53_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_19.2: <bound method> = bound_method %int_2.loc53, %specific_fn.loc53_19 [concrete = constants.%bound_method.419]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_19: init %i32 = call %bound_method.loc53_19.2(%int_2.loc53) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc53_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_19 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc53_19.2: %i32 = converted %int_2.loc53, %.loc53_19.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   br !if.expr.result.loc53(%.loc53_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc53:
// CHECK:STDOUT:   %.loc53_4: %i32 = block_arg !if.expr.result.loc53 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_3.loc53: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc53_27: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc53_27.1: <bound method> = bound_method %int_3.loc53, %impl.elem0.loc53_27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aca]
// CHECK:STDOUT:   %specific_fn.loc53_27: <specific function> = specific_function %impl.elem0.loc53_27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_27.2: <bound method> = bound_method %int_3.loc53, %specific_fn.loc53_27 [concrete = constants.%bound_method.5b5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_27: init %i32 = call %bound_method.loc53_27.2(%int_3.loc53) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc53_27: init %i32 = converted %int_3.loc53, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_27 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc53_4, %.loc53_27
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%i32, (constants.%DefaultOrUnformed.impl_witness.89f) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc56_13.1: %DefaultOrUnformed.type = converted constants.%i32, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc56_13.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc56_13.2: type = converted %.loc56_13.1, %as_type [concrete = constants.%i32]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.3bd, @T.as.DefaultOrUnformed.impl.Op(constants.%i32) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %i32 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %a.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %i32.loc56: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   %true.loc61: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc61 br !if.expr.then.loc61 else br !if.expr.else.loc61
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc61:
// CHECK:STDOUT:   %a.ref.loc61_17: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc61_17: %i32 = acquire_value %a.ref.loc61_17
// CHECK:STDOUT:   br !if.expr.result.loc61(%.loc61_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc61:
// CHECK:STDOUT:   %a.ref.loc61_24: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc61_24: %i32 = acquire_value %a.ref.loc61_24
// CHECK:STDOUT:   br !if.expr.result.loc61(%.loc61_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc61:
// CHECK:STDOUT:   %.loc61_4: %i32 = block_arg !if.expr.result.loc61
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc61: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc61_27.1: <bound method> = bound_method %int_10, %impl.elem0.loc61 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3f7]
// CHECK:STDOUT:   %specific_fn.loc61: <specific function> = specific_function %impl.elem0.loc61, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc61_27.2: <bound method> = bound_method %int_10, %specific_fn.loc61 [concrete = constants.%bound_method.0fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc61: init %i32 = call %bound_method.loc61_27.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc61_27: init %i32 = converted %int_10, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc61 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc61_4, %.loc61_27
// CHECK:STDOUT:   %Destroy.Op.bound.loc56: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc56: init %empty_tuple.type = call %Destroy.Op.bound.loc56(%a.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc33: <bound method> = bound_method %n.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc33: init %empty_tuple.type = call %Destroy.Op.bound.loc33(%n.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc56_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc56_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
