// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_unimplemented_op.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_unimplemented_op.carbon

fn Main() -> i32 {
  // CHECK:STDERR: fail_unimplemented_op.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface Add in type i32 that does not implement that interface.
  // CHECK:STDERR:   return 12 + 34;
  // CHECK:STDERR:          ^~~~~~~
  return 12 + 34;
}

// CHECK:STDOUT: --- fail_unimplemented_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 34 [template]
// CHECK:STDOUT:   %.4: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op, @Add[%Self] [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %Op.type.2 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.3 [symbolic]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Add = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/arithmetic, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/arithmetic, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.5: @Add.%.1 (%.5) = import_ref Core//prelude/operators/arithmetic, inst+29, loaded [symbolic = @Add.%.2 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_32, %.loc11_14.1 [template = i32]
// CHECK:STDOUT:     @Main.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add[constants.%Self: %.4] {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Add[%Self] [symbolic = %Op.type (constants.%Op.type.2)]
// CHECK:STDOUT:   %Op: @Add.%Op.type (%Op.type.2) = struct_value () [symbolic = %Op (constants.%Op.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.4, @Add.%Op.type (%Op.type.2) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @Add.%.1 (%.5) = assoc_entity element0, imports.%import_ref.3 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .Op = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_10: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_15: i32 = int_literal 34 [template = constants.%.3]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%Self: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self](%other: %Self) -> %Self;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add[constants.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add[@Add.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
