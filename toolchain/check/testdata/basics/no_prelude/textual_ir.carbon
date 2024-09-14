// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/textual_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/textual_ir.carbon

fn Foo(n: ()) -> ((), ()) {
  return (n, ());
}

// CHECK:STDOUT: --- textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (%.1, %.1) [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple.1: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %tuple.2: %.2 = tuple_value (%tuple.1, %tuple.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %.loc15_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc15_8.1: %.1 = param n, runtime_param0
// CHECK:STDOUT:     @Foo.%n: %.1 = bind_name n, %n.loc15_8.1
// CHECK:STDOUT:     %.loc15_20: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_24: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_25.1: %.2 = tuple_literal (%.loc15_20, %.loc15_24)
// CHECK:STDOUT:     %.loc15_25.2: type = converted %.loc15_20, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc15_25.3: type = converted %.loc15_24, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc15_25.4: type = converted %.loc15_25.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     @Foo.%return: ref %.2 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: %.1) -> %return: %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.1 = name_ref n, %n
// CHECK:STDOUT:   %.loc16_15.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc16_16.1: %.2 = tuple_literal (%n.ref, %.loc16_15.1)
// CHECK:STDOUT:   %.loc16_16.2: ref %.1 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc16_11: init %.1 = tuple_init () to %.loc16_16.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc16_16.3: init %.1 = converted %n.ref, %.loc16_11 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc16_16.4: ref %.1 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc16_15.2: init %.1 = tuple_init () to %.loc16_16.4 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc16_16.5: init %.1 = converted %.loc16_15.1, %.loc16_15.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc16_16.6: init %.2 = tuple_init (%.loc16_16.3, %.loc16_16.5) to %return [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc16_17: init %.2 = converted %.loc16_16.1, %.loc16_16.6 [template = constants.%tuple.2]
// CHECK:STDOUT:   return %.loc16_17 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
