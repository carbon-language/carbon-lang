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

// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:20: error: bit width must be 64 [CompileTimeFloatBitWidth]
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
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
// CHECK:STDOUT:     %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_16.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_16.2: type = converted %int.make_type_signed, %.loc4_16.1 [template = constants.%i32]
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.64.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.16: <bound method> = bound_method %int.64.1, %Convert.14 [template]
// CHECK:STDOUT:   %.17: <specific function> = specific_function %.16, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.64.2: %i32 = int_value 64 [template]
// CHECK:STDOUT:   %.18: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %GetFloat.type: type = fn_type @GetFloat [template]
// CHECK:STDOUT:   %GetFloat: %GetFloat.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Main//types, inst+27, loaded [template = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Int = %import_ref.38
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
// CHECK:STDOUT:   %int.64: Core.IntLiteral = int_value 64 [template = constants.%int.64.1]
// CHECK:STDOUT:   %.loc6_14.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_14.2: <bound method> = bound_method %int.64, %.loc6_14.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc6_14.3: <specific function> = specific_function %.loc6_14.2, @Convert.2(constants.%int.32) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc6_14.3(%int.64) [template = constants.%int.64.2]
// CHECK:STDOUT:   %.loc6_14.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%int.64.2]
// CHECK:STDOUT:   %.loc6_14.5: %i32 = converted %int.64, %.loc6_14.4 [template = constants.%int.64.2]
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%.loc6_14.5) [template = f64]
// CHECK:STDOUT:   %.loc6_16.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc6_16.2: type = converted %float.make_type, %.loc6_16.1 [template = f64]
// CHECK:STDOUT:   %f.var: ref f64 = var f
// CHECK:STDOUT:   %f: ref f64 = bind_name f, %f.var
// CHECK:STDOUT:   %GetFloat.decl: %GetFloat.type = fn_decl @GetFloat [template = constants.%GetFloat] {
// CHECK:STDOUT:     %dyn_size.patt: %i32 = binding_pattern dyn_size
// CHECK:STDOUT:     %dyn_size.param_patt: %i32 = value_param_pattern %dyn_size.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_23.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_23.2: type = converted %int.make_type_signed, %.loc8_23.1 [template = constants.%i32]
// CHECK:STDOUT:     %dyn_size.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %dyn_size: %i32 = bind_name dyn_size, %dyn_size.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size.param_patt: %i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: %Float.type = name_ref Float, imports.%import_ref.1 [template = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: %i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc9_25.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:   %.loc9_25.2: type = converted %float.make_type, %.loc9_25.1
// CHECK:STDOUT:   return %.loc9_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: f64 = float_literal 0 [template = constants.%.18]
// CHECK:STDOUT:   assign file.%f.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %int.32.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32.1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.16: <bound method> = bound_method %int.32.1, %Convert.14 [template]
// CHECK:STDOUT:   %.17: <specific function> = specific_function %.16, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.32.2: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %int.64.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.64.1, %Convert.14 [template]
// CHECK:STDOUT:   %.19: <specific function> = specific_function %.18, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.64.2: %i32 = int_value 64 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Main//types, inst+27, loaded [template = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Int = %import_ref.38
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
// CHECK:STDOUT:   %int.32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %.loc10_26.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_26.2: <bound method> = bound_method %int.32.loc10, %.loc10_26.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc10_26.3: <specific function> = specific_function %.loc10_26.2, @Convert.2(constants.%int.32.1) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc10_26.3(%int.32.loc10) [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc10_26.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc10_26.5: %i32 = converted %int.32.loc10, %.loc10_26.4 [template = constants.%int.32.2]
// CHECK:STDOUT:   %float.make_type.loc10: init type = call %Float.ref.loc10(%.loc10_26.5) [template = <error>]
// CHECK:STDOUT:   %.loc10_28.1: type = value_of_initializer %float.make_type.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_28.2: type = converted %float.make_type.loc10, %.loc10_28.1 [template = <error>]
// CHECK:STDOUT:   %invalid_float.var: ref <error> = var invalid_float
// CHECK:STDOUT:   %invalid_float: ref <error> = bind_name invalid_float, %invalid_float.var
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.2: type = converted %int.make_type_signed, %.loc12_15.1 [template = constants.%i32]
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
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.64: Core.IntLiteral = int_value 64 [template = constants.%int.64.1]
// CHECK:STDOUT:   %.loc12_23.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_23.2: <bound method> = bound_method %int.64, %.loc12_23.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc12_23.3: <specific function> = specific_function %.loc12_23.2, @Convert.2(constants.%int.32.1) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc12_23.3(%int.64) [template = constants.%int.64.2]
// CHECK:STDOUT:   %.loc12_23.4: init %i32 = converted %int.64, %int.convert_checked [template = constants.%int.64.2]
// CHECK:STDOUT:   assign file.%dyn_size.var, %.loc12_23.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
