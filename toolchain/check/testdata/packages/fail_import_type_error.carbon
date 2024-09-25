// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_import_type_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_import_type_error.carbon

// --- fail_implicit.carbon

package Implicit;

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:12: error: name `x` not found
// CHECK:STDERR: var a_ref: x;
// CHECK:STDERR:            ^
// CHECK:STDERR:
var a_ref: x;
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:18: error: name `x` not found
// CHECK:STDERR: var b_ref: {.a = x};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
var b_ref: {.a = x};
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:13: error: name `x` not found
// CHECK:STDERR: var c_ref: (x,);
// CHECK:STDERR:             ^
// CHECK:STDERR:
var c_ref: (x,);
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+3]]:12: error: name `x` not found
// CHECK:STDERR: var d_ref: x*;
// CHECK:STDERR:            ^
var d_ref: x*;

// --- implicit.impl.carbon

impl package Implicit;

// In each of these cases, name lookup should succeed, but they should all have
// an error type.
var a: i32 = a_ref;
var b: i32 = b_ref;
var c: i32 = c_ref;
var d: i32 = d_ref;

// CHECK:STDOUT: --- fail_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .c_ref = %c_ref
// CHECK:STDOUT:     .d_ref = %d_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.ref.loc8: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %a_ref.var: ref <error> = var a_ref
// CHECK:STDOUT:   %a_ref: ref <error> = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %x.ref.loc13: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %.loc13: <error> = struct_literal (%x.ref.loc13)
// CHECK:STDOUT:   %b_ref.var: ref <error> = var b_ref
// CHECK:STDOUT:   %b_ref: ref <error> = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %x.ref.loc18: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %.loc18: <error> = tuple_literal (%x.ref.loc18)
// CHECK:STDOUT:   %c_ref.var: ref <error> = var c_ref
// CHECK:STDOUT:   %c_ref: ref <error> = bind_name c_ref, %c_ref.var
// CHECK:STDOUT:   %x.ref.loc22: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %.loc22: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:   %d_ref.var: ref <error> = var d_ref
// CHECK:STDOUT:   %d_ref: ref <error> = bind_name d_ref, %d_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref <error> = import_ref Implicit//default, inst+5, loaded
// CHECK:STDOUT:   %import_ref.2: ref <error> = import_ref Implicit//default, inst+10, loaded
// CHECK:STDOUT:   %import_ref.3: ref <error> = import_ref Implicit//default, inst+14, loaded
// CHECK:STDOUT:   %import_ref.4: ref <error> = import_ref Implicit//default, inst+18, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .c_ref = imports.%import_ref.3
// CHECK:STDOUT:     .d_ref = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_32.loc7, %.loc7_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_32.loc8, %.loc8_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_32.loc9, %.loc9_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, imports.%import_ref.1
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %b_ref.ref: ref <error> = name_ref b_ref, imports.%import_ref.2
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %c_ref.ref: ref <error> = name_ref c_ref, imports.%import_ref.3
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %d_ref.ref: ref <error> = name_ref d_ref, imports.%import_ref.4
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
