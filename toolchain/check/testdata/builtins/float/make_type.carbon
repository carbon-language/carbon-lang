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

// CHECK:STDERR: use_types.carbon:[[@LINE+4]]:8: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var f: Float(64) = 0.0;
// CHECK:STDERR:        ^~~~~~~~~
// CHECK:STDERR:
var f: Float(64) = 0.0;

fn GetFloat(dyn_size: i32) -> type {
  return Float(dyn_size);
}

// --- fail_invalid_size.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:20: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var invalid_float: Float(32);
// CHECK:STDERR:                    ^~~~~~~~~
// CHECK:STDERR:
var invalid_float: Float(32);

var dyn_size: i32 = 64;
// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+3]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var dyn: Float(dyn_size);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
var dyn: Float(dyn_size);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Float = %Float.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Float.decl: %Float.type = fn_decl @Float [template = constants.%Float] {
// CHECK:STDOUT:     %size.patt: %i32 = binding_pattern size
// CHECK:STDOUT:     %size.param_patt: %i32 = value_param_pattern %size.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %size.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %size: %i32 = bind_name size, %size.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_64, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %Int = call %Convert.specific_fn(%int_64) [symbolic]
// CHECK:STDOUT:   %float: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %GetFloat.type: type = fn_type @GetFloat [template]
// CHECK:STDOUT:   %GetFloat: %GetFloat.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Main//types, Float, loaded [template = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     .Int = %import_ref.193
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_64, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %Int = call %Convert.specific_fn(%int_64) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc10_14.1: %i32 = value_of_initializer %int.convert_checked [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc10_14.2: %i32 = converted %int_64, %.loc10_14.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%.loc10_14.2)
// CHECK:STDOUT:   %.loc10_16.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:   %.loc10_16.2: type = converted %float.make_type, %.loc10_16.1
// CHECK:STDOUT:   %f.var: ref <error> = var f
// CHECK:STDOUT:   %f: ref <error> = bind_name f, %f.var
// CHECK:STDOUT:   %GetFloat.decl: %GetFloat.type = fn_decl @GetFloat [template = constants.%GetFloat] {
// CHECK:STDOUT:     %dyn_size.patt: %i32 = binding_pattern dyn_size
// CHECK:STDOUT:     %dyn_size.param_patt: %i32 = value_param_pattern %dyn_size.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %dyn_size.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %dyn_size: %i32 = bind_name dyn_size, %dyn_size.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size.param_patt: %i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: %i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc13_25.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:   %.loc13_25.2: type = converted %float.make_type, %.loc13_25.1
// CHECK:STDOUT:   return %.loc13_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: f64 = float_literal 0 [template = constants.%float]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_32, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_32) [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_64, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_64) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Main//types, Float, loaded [template = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     .Int = %import_ref.193
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32.loc10, %impl.elem0 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked: init %Int = call %Convert.specific_fn(%int_32.loc10) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc10_26.1: %i32 = value_of_initializer %int.convert_checked [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc10_26.2: %i32 = converted %int_32.loc10, %.loc10_26.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %float.make_type.loc10: init type = call %Float.ref.loc10(%.loc10_26.2)
// CHECK:STDOUT:   %.loc10_28.1: type = value_of_initializer %float.make_type.loc10
// CHECK:STDOUT:   %.loc10_28.2: type = converted %float.make_type.loc10, %.loc10_28.1
// CHECK:STDOUT:   %invalid_float.var: ref <error> = var invalid_float
// CHECK:STDOUT:   %invalid_float: ref <error> = bind_name invalid_float, %invalid_float.var
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %dyn_size.var: ref %i32 = var dyn_size
// CHECK:STDOUT:   %dyn_size: ref %i32 = bind_name dyn_size, %dyn_size.var
// CHECK:STDOUT:   %Float.ref.loc16: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: ref %i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %.loc16_16: %i32 = bind_value %dyn_size.ref
// CHECK:STDOUT:   %float.make_type.loc16: init type = call %Float.ref.loc16(%.loc16_16)
// CHECK:STDOUT:   %.loc16_24.1: type = value_of_initializer %float.make_type.loc16
// CHECK:STDOUT:   %.loc16_24.2: type = converted %float.make_type.loc16, %.loc16_24.1
// CHECK:STDOUT:   %dyn.var: ref <error> = var dyn
// CHECK:STDOUT:   %dyn: ref <error> = bind_name dyn, %dyn.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_64, %impl.elem0 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked: init %Int = call %Convert.specific_fn(%int_64) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12: init %i32 = converted %int_64, %int.convert_checked [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   assign file.%dyn_size.var, %.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
