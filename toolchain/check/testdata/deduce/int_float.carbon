// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/int_float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/int_float.carbon

// --- int.carbon

library "[[@TEST_NAME]]";

fn F[N:! Core.BigInt()](n: Core.Int(N)) -> Core.BigInt() {
  return N;
}

fn G(a: i64) -> Core.BigInt() {
  return F(a);
}

// --- fail_todo_float.carbon

library "[[@TEST_NAME]]";

// TODO: This should pass once we accept symbolic widths for Core.Float.
// CHECK:STDERR: fail_todo_float.carbon:[[@LINE+3]]:28: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[N:! Core.BigInt()](n: Core.Float(N)) -> Core.BigInt() {
// CHECK:STDERR:                            ^~~~~~~~~~~~~
fn F[N:! Core.BigInt()](n: Core.Float(N)) -> Core.BigInt() {
  return N;
}

fn G(a: f64) -> Core.BigInt() {
  return F(a);
}

// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BigInt.type: type = fn_type @BigInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BigInt: %BigInt.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.BigInt = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.BigInt = int_literal 64 [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %F, @F(%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .BigInt = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %BigInt.type = import_ref Core//prelude/types, inst+7, loaded [template = constants.%BigInt]
// CHECK:STDOUT:   %import_ref.2: %Int.type = import_ref Core//prelude/types, inst+30, loaded [template = constants.%Int]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc4_6.1: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.BigInt = value_param_pattern %N.patt.loc4_6.1, runtime_param<invalid> [symbolic = %N.patt.loc4_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %n.patt: @F.%.loc4_36 (%.2) = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: @F.%.loc4_36 (%.2) = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.BigInt = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.BigInt = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref.loc4_14: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type.loc4_21: init type = call %BigInt.ref.loc4_14() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc4_22.1: type = value_of_initializer %big_int.make_type.loc4_21 [template = Core.BigInt]
// CHECK:STDOUT:     %.loc4_22.2: type = converted %big_int.make_type.loc4_21, %.loc4_22.1 [template = Core.BigInt]
// CHECK:STDOUT:     %Core.ref.loc4_28: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, imports.%import_ref.2 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc4: Core.BigInt = name_ref N, %N.loc4_6.1 [symbolic = %N.loc4_6.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%N.ref.loc4) [symbolic = %.loc4_36 (constants.%.2)]
// CHECK:STDOUT:     %.loc4_38.1: type = value_of_initializer %int.make_type_signed [symbolic = %.loc4_36 (constants.%.2)]
// CHECK:STDOUT:     %.loc4_38.2: type = converted %int.make_type_signed, %.loc4_38.1 [symbolic = %.loc4_36 (constants.%.2)]
// CHECK:STDOUT:     %Core.ref.loc4_44: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref.loc4_48: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type.loc4_55: init type = call %BigInt.ref.loc4_48() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc4_56.1: type = value_of_initializer %big_int.make_type.loc4_55 [template = Core.BigInt]
// CHECK:STDOUT:     %.loc4_56.2: type = converted %big_int.make_type.loc4_55, %.loc4_56.1 [template = Core.BigInt]
// CHECK:STDOUT:     %N.param: Core.BigInt = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_6.1: Core.BigInt = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_6.2 (constants.%N)]
// CHECK:STDOUT:     %n.param: @F.%.loc4_36 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %n: @F.%.loc4_36 (%.2) = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.BigInt = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.BigInt = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %.4 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.4 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.BigInt = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.BigInt = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_9.1: Core.BigInt = int_literal 64 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc8_9.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc8_9.2: type = value_of_initializer %int.make_type_signed [template = constants.%.4]
// CHECK:STDOUT:     %.loc8_9.3: type = converted %int.make_type_signed, %.loc8_9.2 [template = constants.%.4]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type: init type = call %BigInt.ref() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc8_29.1: type = value_of_initializer %big_int.make_type [template = Core.BigInt]
// CHECK:STDOUT:     %.loc8_29.2: type = converted %big_int.make_type, %.loc8_29.1 [template = Core.BigInt]
// CHECK:STDOUT:     %a.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.4 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.BigInt = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.BigInt = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BigInt() -> type = "big_int.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%size.param_patt: Core.BigInt) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc4_6.1: Core.BigInt) {
// CHECK:STDOUT:   %N.loc4_6.2: Core.BigInt = bind_symbolic_name N, 0 [symbolic = %N.loc4_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_6.2: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %.loc4_36: type = int_type signed, %N.loc4_6.2 [symbolic = %.loc4_36 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: Core.BigInt](%n.param_patt: @F.%.loc4_36 (%.2)) -> Core.BigInt {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc5: Core.BigInt = name_ref N, %N.loc4_6.1 [symbolic = %N.loc4_6.2 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %.4) -> Core.BigInt {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: %.4 = name_ref a, %a
// CHECK:STDOUT:   %.loc9_10: <specific function> = specific_function %F.ref, @F(constants.%.3) [template = constants.%.5]
// CHECK:STDOUT:   %F.call: init Core.BigInt = call %.loc9_10(%a.ref)
// CHECK:STDOUT:   %.loc9_14.1: Core.BigInt = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_14.2: Core.BigInt = converted %F.call, %.loc9_14.1
// CHECK:STDOUT:   return %.loc9_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc4_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc4_6.2 => constants.%N
// CHECK:STDOUT:   %.loc4_36 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.3) {
// CHECK:STDOUT:   %N.loc4_6.2 => constants.%.3
// CHECK:STDOUT:   %N.patt.loc4_6.2 => constants.%.3
// CHECK:STDOUT:   %.loc4_36 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BigInt.type: type = fn_type @BigInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BigInt: %BigInt.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.BigInt = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.BigInt = int_literal 64 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .BigInt = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %BigInt.type = import_ref Core//prelude/types, inst+7, loaded [template = constants.%BigInt]
// CHECK:STDOUT:   %import_ref.2: %Float.type = import_ref Core//prelude/types, inst+60, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc8_6.1: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc8_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.BigInt = value_param_pattern %N.patt.loc8_6.1, runtime_param<invalid> [symbolic = %N.patt.loc8_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %n.patt: <error> = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: <error> = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.BigInt = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.BigInt = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc8_10: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref.loc8_14: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type.loc8_21: init type = call %BigInt.ref.loc8_14() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc8_22.1: type = value_of_initializer %big_int.make_type.loc8_21 [template = Core.BigInt]
// CHECK:STDOUT:     %.loc8_22.2: type = converted %big_int.make_type.loc8_21, %.loc8_22.1 [template = Core.BigInt]
// CHECK:STDOUT:     %Core.ref.loc8_28: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Float.ref: %Float.type = name_ref Float, imports.%import_ref.2 [template = constants.%Float]
// CHECK:STDOUT:     %N.ref.loc8: Core.BigInt = name_ref N, %N.loc8_6.1 [symbolic = %N.loc8_6.2 (constants.%N)]
// CHECK:STDOUT:     %float.make_type: init type = call %Float.ref(%N.ref.loc8)
// CHECK:STDOUT:     %.loc8_40.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:     %.loc8_40.2: type = converted %float.make_type, %.loc8_40.1
// CHECK:STDOUT:     %Core.ref.loc8_46: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref.loc8_50: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type.loc8_57: init type = call %BigInt.ref.loc8_50() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc8_58.1: type = value_of_initializer %big_int.make_type.loc8_57 [template = Core.BigInt]
// CHECK:STDOUT:     %.loc8_58.2: type = converted %big_int.make_type.loc8_57, %.loc8_58.1 [template = Core.BigInt]
// CHECK:STDOUT:     %N.param: Core.BigInt = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc8_6.1: Core.BigInt = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc8_6.2 (constants.%N)]
// CHECK:STDOUT:     %n.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %n: <error> = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.BigInt = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.BigInt = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.BigInt = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.BigInt = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_9.1: Core.BigInt = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%.loc12_9.1) [template = f64]
// CHECK:STDOUT:     %.loc12_9.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:     %.loc12_9.3: type = converted %float.make_type, %.loc12_9.2 [template = f64]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %BigInt.ref: %BigInt.type = name_ref BigInt, imports.%import_ref.1 [template = constants.%BigInt]
// CHECK:STDOUT:     %big_int.make_type: init type = call %BigInt.ref() [template = Core.BigInt]
// CHECK:STDOUT:     %.loc12_29.1: type = value_of_initializer %big_int.make_type [template = Core.BigInt]
// CHECK:STDOUT:     %.loc12_29.2: type = converted %big_int.make_type, %.loc12_29.1 [template = Core.BigInt]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.BigInt = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.BigInt = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BigInt() -> type = "big_int.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: Core.BigInt) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc8_6.1: Core.BigInt) {
// CHECK:STDOUT:   %N.loc8_6.2: Core.BigInt = bind_symbolic_name N, 0 [symbolic = %N.loc8_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc8_6.2: Core.BigInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc8_6.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: Core.BigInt](%n.param_patt: <error>) -> Core.BigInt {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc9: Core.BigInt = name_ref N, %N.loc8_6.1 [symbolic = %N.loc8_6.2 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: f64) -> Core.BigInt {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc8_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc8_6.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
