// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// TODO: Short-circuit operators should be permitted outside functions.
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+12]]:11: error: cannot evaluate type expression
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:           ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+8]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+4]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_: F(true and true);

// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+11]]:10: error: cannot evaluate type expression
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+7]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+3]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~~~~~~
var or_: F(true or true);

// CHECK:STDOUT: --- fail_and_or_not_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.3: bool = bool_literal true [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace <unexpected>.inst+1, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.3: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc42_17: bool = block_arg <unexpected instblockref block18> [template = constants.%.3]
// CHECK:STDOUT:   %F.call: init type = call <unexpected>.inst+60.loc42_10(%.loc42_17)
// CHECK:STDOUT:   %.loc42_24.1: type = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc42_24.2: type = converted %F.call, %.loc42_24.1
// CHECK:STDOUT:   %or_.var: ref <error> = var or_
// CHECK:STDOUT:   %or_: ref <error> = bind_name or_, %or_.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_20.2: type = converted %int.make_type_32, %.loc13_20.1 [template = i32]
// CHECK:STDOUT:   br !if.expr.result(%.loc13_20.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc13_29: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc13_29) [template = f64]
// CHECK:STDOUT:   %.loc13_24.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc13_24.2: type = converted %float.make_type, %.loc13_24.1 [template = f64]
// CHECK:STDOUT:   br !if.expr.result(%.loc13_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc13_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc13_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
