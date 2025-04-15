// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_init.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() {
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: error: cannot initialize class with 2 fields from struct with 1 field [StructInitElementCountMismatch]
  // CHECK:STDERR:   {.a = 1} as Class;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  {.a = 1} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: error: missing value for field `b` in struct initialization [StructInitMissingFieldInLiteral]
  // CHECK:STDERR:   {.a = 1, .c = 2} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = 1, .c = 2} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: error: cannot initialize class with 2 fields from struct with 3 fields [StructInitElementCountMismatch]
  // CHECK:STDERR:   {.a = 1, .b = 2, .c = 3} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = 1, .b = 2, .c = 3} as Class;
}

// CHECK:STDOUT: --- fail_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.fef: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%Class) [concrete]
// CHECK:STDOUT:   %Op.type.7de: type = fn_type @Op.2, @impl.49c(%Class) [concrete]
// CHECK:STDOUT:   %Op.d64: %Op.type.7de = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class, (%Destroy.impl_witness.fef) [concrete]
// CHECK:STDOUT:   %.b9f: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.c: type = struct_type {.a: Core.IntLiteral, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (imports.%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness %ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.d64, @Op.2(%Class) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc12: %Class.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13: %Class.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   .b = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc21_10.1: %struct_type.a = struct_literal (%int_1.loc21)
// CHECK:STDOUT:   %Class.ref.loc21: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc21: %.b9f = impl_witness_access constants.%Destroy.impl_witness.fef, element0 [concrete = constants.%Op.d64]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %.loc21_10.2, %impl.elem0.loc21
// CHECK:STDOUT:   %.loc21_10.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc21_10.3: ref %Class = temporary %.loc21_10.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_12: ref %Class = converted %.loc21_10.1, %.loc21_10.3 [concrete = <error>]
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc26: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc26_18.1: %struct_type.a.c = struct_literal (%int_1.loc26, %int_2.loc26)
// CHECK:STDOUT:   %Class.ref.loc26: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc26_18.1: %.b9f = impl_witness_access constants.%Destroy.impl_witness.fef, element0 [concrete = constants.%Op.d64]
// CHECK:STDOUT:   %bound_method.loc26_18.1: <bound method> = bound_method %.loc26_18.3, %impl.elem0.loc26_18.1
// CHECK:STDOUT:   %impl.elem0.loc26_18.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_18.2: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_18.2 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc26_18.1: <specific function> = specific_function %impl.elem0.loc26_18.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_18.3: <bound method> = bound_method %int_1.loc26, %specific_fn.loc26_18.1 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc26_18.3(%int_1.loc26) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc26_18.2: init %i32 = converted %int_1.loc26, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc26_18.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc26_18.4: ref %i32 = class_element_access %.loc26_18.3, element0
// CHECK:STDOUT:   %.loc26_18.5: init %i32 = initialize_from %.loc26_18.2 to %.loc26_18.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc26_18.6: ref %Class = temporary %.loc26_18.3, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc26_20: ref %Class = converted %.loc26_18.1, %.loc26_18.6 [concrete = <error>]
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc31: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc31_26.1: %struct_type.a.b.c = struct_literal (%int_1.loc31, %int_2.loc31, %int_3)
// CHECK:STDOUT:   %Class.ref.loc31: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc31: %.b9f = impl_witness_access constants.%Destroy.impl_witness.fef, element0 [concrete = constants.%Op.d64]
// CHECK:STDOUT:   %bound_method.loc31: <bound method> = bound_method %.loc31_26.2, %impl.elem0.loc31
// CHECK:STDOUT:   %.loc31_26.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc31_26.3: ref %Class = temporary %.loc31_26.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc31_28: ref %Class = converted %.loc31_26.1, %.loc31_26.3 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Op.2(constants.%Class) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc31_26.2, %specific_fn.loc31
// CHECK:STDOUT:   %.loc31_26.4: %Class = bind_value %.loc31_26.2
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc31_26.4)
// CHECK:STDOUT:   %specific_fn.loc26_18.2: <specific function> = specific_function %impl.elem0.loc26_18.1, @Op.2(constants.%Class) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.loc26_18.3, %specific_fn.loc26_18.2
// CHECK:STDOUT:   %.loc26_18.7: %Class = bind_value %.loc26_18.3
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc26_18.7)
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Op.2(constants.%Class) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %.loc21_10.2, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_10.4: %Class = bind_value %.loc21_10.2
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc21_10.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
