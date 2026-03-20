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
// CHECK:STDOUT:   %Copy.impl_witness.98e: <witness> = impl_witness imports.%Copy.impl_witness_table.d8f [concrete]
// CHECK:STDOUT:   %Copy.facet.34f: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.98e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.42e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.34f) [concrete]
// CHECK:STDOUT:   %.5e6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.42e, %Copy.facet.34f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.74e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.04c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound.e8e: <bound method> = bound_method %int_4, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b47: <witness> = impl_witness imports.%Copy.impl_witness_table.b1c [concrete]
// CHECK:STDOUT:   %Copy.facet.4b0: %Copy.type = facet_value type, (%Copy.impl_witness.b47) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a4f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.4b0) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a4f, %Copy.facet.4b0 [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %ptr.235, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct.a0d: %struct_type.x.y = struct_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.8d3: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.c67: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.137: %T.as.DefaultOrUnformed.impl.Op.type.c67 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %i32, (%DefaultOrUnformed.impl_witness.8d3) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.137, @T.as.DefaultOrUnformed.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2bf: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.409: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.265: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.66a: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc190, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d8f = impl_witness_table (%Core.import_ref.66a), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1a7: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc237, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b1c = impl_witness_table (%Core.import_ref.1a7), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc34, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32 [concrete = constants.%.ff5]
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
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc15: %.5e6 = impl_witness_access constants.%Copy.impl_witness.98e, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_2.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.74e]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc15: init Core.IntLiteral = call %bound_method.loc15(%int_2.loc15) [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   assign %int_1.loc13, %Core.IntLiteral.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc20.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc20 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %i32 = call %bound_method.loc20.2(%int_1.loc21) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc20: init %i32 = converted %int_1.loc21, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %F.call, %.loc20
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc26: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc27.1: %tuple.type.f94 = tuple_literal (%int_1.loc24, %int_2.loc26) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %int_3.loc30: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc32: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc33.1: %tuple.type.f94 = tuple_literal (%int_3.loc30, %int_4.loc32) [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %impl.elem0.loc30: %.5e6 = impl_witness_access constants.%Copy.impl_witness.98e, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %int_3.loc30, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.04c]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc30: init Core.IntLiteral = call %bound_method.loc30(%int_3.loc30) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %tuple.elem0.loc33: Core.IntLiteral = tuple_access %.loc27.1, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc33.2: init Core.IntLiteral to %tuple.elem0.loc33 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc30 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc32: %.5e6 = impl_witness_access constants.%Copy.impl_witness.98e, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc32: <bound method> = bound_method %int_4.loc32, %impl.elem0.loc32 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.e8e]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc32: init Core.IntLiteral = call %bound_method.loc32(%int_4.loc32) [concrete = constants.%int_4]
// CHECK:STDOUT:   %tuple.elem1.loc33: Core.IntLiteral = tuple_access %.loc27.1, element1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc33.3: init Core.IntLiteral to %tuple.elem1.loc33 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc32 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc33.4: init %tuple.type.f94 to %.loc27.1 = tuple_init (%.loc33.2, %.loc33.3) [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %.loc28: init %tuple.type.f94 = converted %.loc33.1, %.loc33.4 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   assign %.loc27.1, %.loc28
// CHECK:STDOUT:   %tuple.loc27: %tuple.type.f94 = tuple_value (%int_1.loc24, %int_2.loc26) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc27.2: %tuple.type.f94 = converted %.loc27.1, %tuple.loc27 [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc35: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc35.1: <bound method> = bound_method %int_0, %impl.elem0.loc35 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35.2: <bound method> = bound_method %int_0, %specific_fn.loc35 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35: init %i32 = call %bound_method.loc35.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc35: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc35
// CHECK:STDOUT:   %i32.loc38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: ref %i32 = ref_binding n, %n.var
// CHECK:STDOUT:   %n.ref.loc43: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc45: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc46.1: %tuple.type.d07 = tuple_literal (%n.ref.loc43, %n.ref.loc45)
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc51: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc52.1: %tuple.type.f94 = tuple_literal (%int_1.loc49, %int_2.loc51) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc52.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc52.1: <bound method> = bound_method %int_1.loc49, %impl.elem0.loc52.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc52.1: <specific function> = specific_function %impl.elem0.loc52.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52.2: <bound method> = bound_method %int_1.loc49, %specific_fn.loc52.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52.1: init %i32 = call %bound_method.loc52.2(%int_1.loc49) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc52.2: init %i32 = converted %int_1.loc49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc52: %i32 = tuple_access %.loc46.1, element0
// CHECK:STDOUT:   %.loc52.3: init %i32 to %tuple.elem0.loc52 = in_place_init %.loc52.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc52.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc52.3: <bound method> = bound_method %int_2.loc51, %impl.elem0.loc52.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc52.2: <specific function> = specific_function %impl.elem0.loc52.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52.4: <bound method> = bound_method %int_2.loc51, %specific_fn.loc52.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52.2: init %i32 = call %bound_method.loc52.4(%int_2.loc51) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc52.4: init %i32 = converted %int_2.loc51, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc52: %i32 = tuple_access %.loc46.1, element1
// CHECK:STDOUT:   %.loc52.5: init %i32 to %tuple.elem1.loc52 = in_place_init %.loc52.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc52.6: init %tuple.type.d07 to %.loc46.1 = tuple_init (%.loc52.3, %.loc52.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc47: init %tuple.type.d07 = converted %.loc52.1, %.loc52.6 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign %.loc46.1, %.loc47
// CHECK:STDOUT:   %.loc43: %i32 = acquire_value %n.ref.loc43
// CHECK:STDOUT:   %.loc45: %i32 = acquire_value %n.ref.loc45
// CHECK:STDOUT:   %tuple.loc46: %tuple.type.d07 = tuple_value (%.loc43, %.loc45)
// CHECK:STDOUT:   %.loc46.2: %tuple.type.d07 = converted %.loc46.1, %tuple.loc46
// CHECK:STDOUT:   %i32.loc54: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc56: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32.loc56 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   %impl.elem0.loc57: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc57: <bound method> = bound_method %ptr, %impl.elem0.loc57 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method.loc57(%ptr) [concrete = constants.%ptr.235]
// CHECK:STDOUT:   assign %i32.loc54, %type.as.Copy.impl.Op.call
// CHECK:STDOUT:   %int_1.loc63: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc68: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc69.1: %struct_type.x.y = struct_literal (%int_1.loc63, %int_2.loc68) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %int_3.loc75: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc80: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc81.1: %struct_type.x.y = struct_literal (%int_3.loc75, %int_4.loc80) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %impl.elem0.loc75: %.5e6 = impl_witness_access constants.%Copy.impl_witness.98e, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc75: <bound method> = bound_method %int_3.loc75, %impl.elem0.loc75 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.04c]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc75: init Core.IntLiteral = call %bound_method.loc75(%int_3.loc75) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc81.2: Core.IntLiteral = struct_access %.loc69.1, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc81.3: init Core.IntLiteral to %.loc81.2 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc75 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc80: %.5e6 = impl_witness_access constants.%Copy.impl_witness.98e, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc80: <bound method> = bound_method %int_4.loc80, %impl.elem0.loc80 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound.e8e]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc80: init Core.IntLiteral = call %bound_method.loc80(%int_4.loc80) [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc81.4: Core.IntLiteral = struct_access %.loc69.1, element1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc81.5: init Core.IntLiteral to %.loc81.4 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc80 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc81.6: init %struct_type.x.y to %.loc69.1 = struct_init (%.loc81.3, %.loc81.5) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %.loc70: init %struct_type.x.y = converted %.loc81.1, %.loc81.6 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   assign %.loc69.1, %.loc70
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%int_1.loc63, %int_2.loc68) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc69.2: %struct_type.x.y = converted %.loc69.1, %struct [concrete = constants.%struct.004]
// CHECK:STDOUT:   %true.loc85: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc85 br !if.expr.then.loc84 else br !if.expr.else.loc84
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc84:
// CHECK:STDOUT:   %int_1.loc87: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc86: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc86.1: <bound method> = bound_method %int_1.loc87, %impl.elem0.loc86 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc86: <specific function> = specific_function %impl.elem0.loc86, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc86.2: <bound method> = bound_method %int_1.loc87, %specific_fn.loc86 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc86: init %i32 = call %bound_method.loc86.2(%int_1.loc87) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc86.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc86 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc86.2: %i32 = converted %int_1.loc87, %.loc86.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   br !if.expr.result.loc84(%.loc86.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc84:
// CHECK:STDOUT:   %int_2.loc89: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc88: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc88.1: <bound method> = bound_method %int_2.loc89, %impl.elem0.loc88 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc88: <specific function> = specific_function %impl.elem0.loc88, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc88.2: <bound method> = bound_method %int_2.loc89, %specific_fn.loc88 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc88: init %i32 = call %bound_method.loc88.2(%int_2.loc89) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc88.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc88 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc88.2: %i32 = converted %int_2.loc89, %.loc88.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   br !if.expr.result.loc84(%.loc88.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc84:
// CHECK:STDOUT:   %.loc84: %i32 = block_arg !if.expr.result.loc84 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_3.loc92: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc91: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc91.1: <bound method> = bound_method %int_3.loc92, %impl.elem0.loc91 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc91: <specific function> = specific_function %impl.elem0.loc91, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc91.2: <bound method> = bound_method %int_3.loc92, %specific_fn.loc91 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc91: init %i32 = call %bound_method.loc91.2(%int_3.loc92) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc91: init %i32 = converted %int_3.loc92, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc91 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc84, %.loc91
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%i32, (constants.%DefaultOrUnformed.impl_witness.8d3) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc98.1: %DefaultOrUnformed.type = converted constants.%i32, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc98.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc98.2: type = converted %.loc98.1, %as_type [concrete = constants.%i32]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.137, @T.as.DefaultOrUnformed.impl.Op(constants.%i32) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %i32 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %a.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %i32.loc97: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   %true.loc101: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc101 br !if.expr.then.loc100 else br !if.expr.else.loc100
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc100:
// CHECK:STDOUT:   %a.ref.loc103: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc103: %i32 = acquire_value %a.ref.loc103
// CHECK:STDOUT:   br !if.expr.result.loc100(%.loc103)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc100:
// CHECK:STDOUT:   %a.ref.loc105: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc105: %i32 = acquire_value %a.ref.loc105
// CHECK:STDOUT:   br !if.expr.result.loc100(%.loc105)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc100:
// CHECK:STDOUT:   %.loc100: %i32 = block_arg !if.expr.result.loc100
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc107: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc107.1: <bound method> = bound_method %int_10, %impl.elem0.loc107 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2bf]
// CHECK:STDOUT:   %specific_fn.loc107: <specific function> = specific_function %impl.elem0.loc107, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc107.2: <bound method> = bound_method %int_10, %specific_fn.loc107 [concrete = constants.%bound_method.409]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc107: init %i32 = call %bound_method.loc107.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc107: init %i32 = converted %int_10, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc107 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc100, %.loc107
// CHECK:STDOUT:   %Destroy.Op.bound.loc94: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc94: init %empty_tuple.type = call %Destroy.Op.bound.loc94(%a.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc35: <bound method> = bound_method %n.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc35: init %empty_tuple.type = call %Destroy.Op.bound.loc35(%n.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
