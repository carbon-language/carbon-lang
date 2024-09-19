// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn Float(size: i32) -> type = "float.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

var f: Float(64) = 0.0;

fn GetFloat(dyn_size: i32) -> type {
  return Float(dyn_size);
}

// --- fail_invalid_size.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:20: error: bit width must be 64
// CHECK:STDERR: var invalid_float: Float(32);
// CHECK:STDERR:                    ^~~~~~
// CHECK:STDERR:
var invalid_float: Float(32);

var dyn_size: i32 = 64;
// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+3]]:10: error: cannot evaluate type expression
// CHECK:STDERR: var dyn: Float(dyn_size);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
var dyn: Float(dyn_size);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Float = %Float.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Float.decl: %Float.type = fn_decl @Float [template = constants.%Float] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_16.2: type = converted %int.make_type_32, %.loc4_16.1 [template = i32]
// CHECK:STDOUT:     %size.loc4_10.1: i32 = param size, runtime_param0
// CHECK:STDOUT:     @Float.%size: i32 = bind_name size, %size.loc4_10.1
// CHECK:STDOUT:     @Float.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %GetFloat.type: type = fn_type @GetFloat [template]
// CHECK:STDOUT:   %GetFloat: %GetFloat.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Main//types, inst+15, loaded [template = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Float = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .GetFloat = %GetFloat.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Float.ref: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %.loc6_14: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%.loc6_14) [template = f64]
// CHECK:STDOUT:   %.loc6_16.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc6_16.2: type = converted %float.make_type, %.loc6_16.1 [template = f64]
// CHECK:STDOUT:   %f.var: ref f64 = var f
// CHECK:STDOUT:   %f: ref f64 = bind_name f, %f.var
// CHECK:STDOUT:   %GetFloat.decl: %GetFloat.type = fn_decl @GetFloat [template = constants.%GetFloat] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_23.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_23.2: type = converted %int.make_type_32, %.loc8_23.1 [template = i32]
// CHECK:STDOUT:     %dyn_size.loc8_13.1: i32 = param dyn_size, runtime_param0
// CHECK:STDOUT:     @GetFloat.%dyn_size: i32 = bind_name dyn_size, %dyn_size.loc8_13.1
// CHECK:STDOUT:     @GetFloat.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size: i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc9_25.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:   %.loc9_25.2: type = converted %float.make_type, %.loc9_25.1
// CHECK:STDOUT:   return %.loc9_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: f64 = float_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%f.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 64 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Main//types, inst+15, loaded [template = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Float = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .invalid_float = %invalid_float
// CHECK:STDOUT:     .dyn_size = %dyn_size
// CHECK:STDOUT:     .dyn = %dyn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Float.ref.loc10: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 32 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type.loc10: init type = call %Float.ref.loc10(%.loc10_26) [template = <error>]
// CHECK:STDOUT:   %.loc10_28.1: type = value_of_initializer %float.make_type.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_28.2: type = converted %float.make_type.loc10, %.loc10_28.1 [template = <error>]
// CHECK:STDOUT:   %invalid_float.var: ref <error> = var invalid_float
// CHECK:STDOUT:   %invalid_float: ref <error> = bind_name invalid_float, %invalid_float.var
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_15.2: type = converted %int.make_type_32, %.loc12_15.1 [template = i32]
// CHECK:STDOUT:   %dyn_size.var: ref i32 = var dyn_size
// CHECK:STDOUT:   %dyn_size: ref i32 = bind_name dyn_size, %dyn_size.var
// CHECK:STDOUT:   %Float.ref.loc16: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: ref i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %.loc16_16: i32 = bind_value %dyn_size.ref
// CHECK:STDOUT:   %float.make_type.loc16: init type = call %Float.ref.loc16(%.loc16_16)
// CHECK:STDOUT:   %.loc16_24.1: type = value_of_initializer %float.make_type.loc16
// CHECK:STDOUT:   %.loc16_24.2: type = converted %float.make_type.loc16, %.loc16_24.1
// CHECK:STDOUT:   %dyn.var: ref <error> = var dyn
// CHECK:STDOUT:   %dyn: ref <error> = bind_name dyn, %dyn.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 64 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%dyn_size.var, %.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
