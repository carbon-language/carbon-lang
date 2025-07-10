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
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a0d: %struct_type.x.y = struct_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Convert.bound.491: <bound method> = bound_method %int_10.64f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.5ba: <bound method> = bound_method %int_10.64f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.265: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.3, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.e6a, @Op.3(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   assign %int_1.loc22, %int_2.loc22
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc27: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_7.1: <bound method> = bound_method %int_1.loc27, %impl.elem0.loc27 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_7.2: <bound method> = bound_method %int_1.loc27, %specific_fn.loc27 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %bound_method.loc27_7.2(%int_1.loc27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc27: init %i32 = converted %int_1.loc27, %int.convert_checked.loc27 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %F.call, %.loc27
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc32: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32_8.1: %tuple.type.f94 = tuple_literal (%int_1.loc32, %int_2.loc32)
// CHECK:STDOUT:   %int_3.loc32: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc32: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc32_17.1: %tuple.type.f94 = tuple_literal (%int_3.loc32, %int_4.loc32)
// CHECK:STDOUT:   %tuple.elem0.loc32: Core.IntLiteral = tuple_access %.loc32_8.1, element0
// CHECK:STDOUT:   %.loc32_17.2: init Core.IntLiteral = initialize_from %int_3.loc32 to %tuple.elem0.loc32 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %tuple.elem1.loc32: Core.IntLiteral = tuple_access %.loc32_8.1, element1
// CHECK:STDOUT:   %.loc32_17.3: init Core.IntLiteral = initialize_from %int_4.loc32 to %tuple.elem1.loc32 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc32_17.4: init %tuple.type.f94 = tuple_init (%.loc32_17.2, %.loc32_17.3) to %.loc32_8.1 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %.loc32_10: init %tuple.type.f94 = converted %.loc32_17.1, %.loc32_17.4 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   assign %.loc32_8.1, %.loc32_10
// CHECK:STDOUT:   %tuple.loc32: %tuple.type.f94 = tuple_value (%int_1.loc32, %int_2.loc32) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc32_8.2: %tuple.type.f94 = converted %.loc32_8.1, %tuple.loc32 [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc33: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc33_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc33 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_3.2: <bound method> = bound_method %int_0, %specific_fn.loc33 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc33: init %i32 = call %bound_method.loc33_3.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc33_3: init %i32 = converted %int_0, %int.convert_checked.loc33 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc33_3
// CHECK:STDOUT:   %.loc33_10: type = splice_block %i32.loc33 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   %n.ref.loc38_4: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc38_7: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc38_8.1: %tuple.type.d07 = tuple_literal (%n.ref.loc38_4, %n.ref.loc38_7)
// CHECK:STDOUT:   %int_1.loc38: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc38: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc38_17.1: %tuple.type.f94 = tuple_literal (%int_1.loc38, %int_2.loc38)
// CHECK:STDOUT:   %impl.elem0.loc38_17.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc38_17.1: <bound method> = bound_method %int_1.loc38, %impl.elem0.loc38_17.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc38_17.1: <specific function> = specific_function %impl.elem0.loc38_17.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_17.2: <bound method> = bound_method %int_1.loc38, %specific_fn.loc38_17.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc38_17.1: init %i32 = call %bound_method.loc38_17.2(%int_1.loc38) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc38_17.2: init %i32 = converted %int_1.loc38, %int.convert_checked.loc38_17.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc38: %i32 = tuple_access %.loc38_8.1, element0
// CHECK:STDOUT:   %.loc38_17.3: init %i32 = initialize_from %.loc38_17.2 to %tuple.elem0.loc38 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc38_17.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc38_17.3: <bound method> = bound_method %int_2.loc38, %impl.elem0.loc38_17.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc38_17.2: <specific function> = specific_function %impl.elem0.loc38_17.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_17.4: <bound method> = bound_method %int_2.loc38, %specific_fn.loc38_17.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc38_17.2: init %i32 = call %bound_method.loc38_17.4(%int_2.loc38) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_17.4: init %i32 = converted %int_2.loc38, %int.convert_checked.loc38_17.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc38: %i32 = tuple_access %.loc38_8.1, element1
// CHECK:STDOUT:   %.loc38_17.5: init %i32 = initialize_from %.loc38_17.4 to %tuple.elem1.loc38 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_17.6: init %tuple.type.d07 = tuple_init (%.loc38_17.3, %.loc38_17.5) to %.loc38_8.1 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc38_10: init %tuple.type.d07 = converted %.loc38_17.1, %.loc38_17.6 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign %.loc38_8.1, %.loc38_10
// CHECK:STDOUT:   %.loc38_4: %i32 = bind_value %n.ref.loc38_4
// CHECK:STDOUT:   %.loc38_7: %i32 = bind_value %n.ref.loc38_7
// CHECK:STDOUT:   %tuple.loc38: %tuple.type.d07 = tuple_value (%.loc38_4, %.loc38_7)
// CHECK:STDOUT:   %.loc38_8.2: %tuple.type.d07 = converted %.loc38_8.1, %tuple.loc38
// CHECK:STDOUT:   %int_32.loc43_3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc43_3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc43_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc43_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32.loc43_9 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   assign %i32.loc43_3, %ptr
// CHECK:STDOUT:   %int_1.loc48: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc48: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc48_18.1: %struct_type.x.y = struct_literal (%int_1.loc48, %int_2.loc48)
// CHECK:STDOUT:   %int_3.loc48: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc48: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc48_37.1: %struct_type.x.y = struct_literal (%int_3.loc48, %int_4.loc48)
// CHECK:STDOUT:   %.loc48_37.2: Core.IntLiteral = struct_access %.loc48_18.1, element0
// CHECK:STDOUT:   %.loc48_37.3: init Core.IntLiteral = initialize_from %int_3.loc48 to %.loc48_37.2 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc48_37.4: Core.IntLiteral = struct_access %.loc48_18.1, element1
// CHECK:STDOUT:   %.loc48_37.5: init Core.IntLiteral = initialize_from %int_4.loc48 to %.loc48_37.4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc48_37.6: init %struct_type.x.y = struct_init (%.loc48_37.3, %.loc48_37.5) to %.loc48_18.1 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %.loc48_20: init %struct_type.x.y = converted %.loc48_37.1, %.loc48_37.6 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   assign %.loc48_18.1, %.loc48_20
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%int_1.loc48, %int_2.loc48) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc48_18.2: %struct_type.x.y = converted %.loc48_18.1, %struct [concrete = constants.%struct.004]
// CHECK:STDOUT:   %true.loc53: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc53 br !if.expr.then.loc53 else br !if.expr.else.loc53
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc53:
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc53: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc53: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc53_12: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc53_12.1: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53_12 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc53_12: <specific function> = specific_function %impl.elem0.loc53_12, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_12.2: <bound method> = bound_method %int_1.loc53, %specific_fn.loc53_12 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc53_12: init %i32 = call %bound_method.loc53_12.2(%int_1.loc53) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_12.1: %i32 = value_of_initializer %int.convert_checked.loc53_12 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_12.2: %i32 = converted %int_1.loc53, %.loc53_12.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   br !if.expr.result.loc53(%.loc53_12.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc53:
// CHECK:STDOUT:   %int_2.loc53: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc53_19: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc53_19.1: <bound method> = bound_method %int_2.loc53, %impl.elem0.loc53_19 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc53_19: <specific function> = specific_function %impl.elem0.loc53_19, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_19.2: <bound method> = bound_method %int_2.loc53, %specific_fn.loc53_19 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc53_19: init %i32 = call %bound_method.loc53_19.2(%int_2.loc53) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc53_19.1: %i32 = value_of_initializer %int.convert_checked.loc53_19 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc53_19.2: %i32 = converted %int_2.loc53, %.loc53_19.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   br !if.expr.result.loc53(%.loc53_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc53:
// CHECK:STDOUT:   %.loc53_4: %i32 = block_arg !if.expr.result.loc53 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_3.loc53: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc53_27: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc53_27.1: <bound method> = bound_method %int_3.loc53, %impl.elem0.loc53_27 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc53_27: <specific function> = specific_function %impl.elem0.loc53_27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_27.2: <bound method> = bound_method %int_3.loc53, %specific_fn.loc53_27 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc53_27: init %i32 = call %bound_method.loc53_27.2(%int_3.loc53) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc53_27: init %i32 = converted %int_3.loc53, %int.convert_checked.loc53_27 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc53_4, %.loc53_27
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %.loc56: type = splice_block %i32.loc56 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc56: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc56: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %true.loc61: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc61 br !if.expr.then.loc61 else br !if.expr.else.loc61
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc61:
// CHECK:STDOUT:   %a.ref.loc61_17: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc61_17: %i32 = bind_value %a.ref.loc61_17
// CHECK:STDOUT:   br !if.expr.result.loc61(%.loc61_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc61:
// CHECK:STDOUT:   %a.ref.loc61_24: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc61_24: %i32 = bind_value %a.ref.loc61_24
// CHECK:STDOUT:   br !if.expr.result.loc61(%.loc61_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc61:
// CHECK:STDOUT:   %.loc61_4: %i32 = block_arg !if.expr.result.loc61
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc61: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc61_27.1: <bound method> = bound_method %int_10, %impl.elem0.loc61 [concrete = constants.%Convert.bound.491]
// CHECK:STDOUT:   %specific_fn.loc61: <specific function> = specific_function %impl.elem0.loc61, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc61_27.2: <bound method> = bound_method %int_10, %specific_fn.loc61 [concrete = constants.%bound_method.5ba]
// CHECK:STDOUT:   %int.convert_checked.loc61: init %i32 = call %bound_method.loc61_27.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc61_27: init %i32 = converted %int_10, %int.convert_checked.loc61 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc61_4, %.loc61_27
// CHECK:STDOUT:   %Op.bound.loc56: <bound method> = bound_method %a.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc56: <bound method> = bound_method %a.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc56: %ptr.235 = addr_of %a.var
// CHECK:STDOUT:   %no_op.loc56: init %empty_tuple.type = call %bound_method.loc56(%addr.loc56)
// CHECK:STDOUT:   %Op.bound.loc33: <bound method> = bound_method %n.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_3.3: <bound method> = bound_method %n.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc33: %ptr.235 = addr_of %n.var
// CHECK:STDOUT:   %no_op.loc33: init %empty_tuple.type = call %bound_method.loc33_3.3(%addr.loc33)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
