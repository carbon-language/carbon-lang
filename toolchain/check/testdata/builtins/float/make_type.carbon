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
// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var dyn: Float(dyn_size);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
var dyn: Float(dyn_size);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Float = %Float.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Float.decl: %Float.type = fn_decl @Float [concrete = constants.%Float] {
// CHECK:STDOUT:     %size.patt: %i32 = binding_pattern size
// CHECK:STDOUT:     %size.param_patt: %i32 = value_param_pattern %size.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %size.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %size: %i32 = bind_name size, %size.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64.fab: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9ba: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.6da: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.b97: <witness> = impl_witness (imports.%Core.import_ref.a86), @impl.c81(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.ed5: type = fn_type @Convert.2, @impl.c81(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.16d: %Convert.type.ed5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9ba = facet_value Core.IntLiteral, (%impl_witness.b97) [concrete]
// CHECK:STDOUT:   %.b14: type = fn_type_with_self_type %Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_64.fab, %Convert.16d [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.16d, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_64.fab, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_64.198: %i32 = int_value 64 [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 0 [concrete]
// CHECK:STDOUT:   %GetFloat.type: type = fn_type @GetFloat [concrete]
// CHECK:STDOUT:   %GetFloat: %GetFloat.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float: %Float.type = import_ref Main//types, Float, loaded [concrete = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Float = imports.%Main.Float
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .GetFloat = %GetFloat.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: f64 = binding_pattern f
// CHECK:STDOUT:     %.loc6_1: f64 = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref f64 = var f
// CHECK:STDOUT:   %.loc6_16.1: type = splice_block %.loc6_16.3 [concrete = f64] {
// CHECK:STDOUT:     %Float.ref: %Float.type = name_ref Float, imports.%Main.Float [concrete = constants.%Float]
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %impl.elem0: %.b14 = impl_witness_access constants.%impl_witness.b97, element0 [concrete = constants.%Convert.16d]
// CHECK:STDOUT:     %bound_method.loc6_14.1: <bound method> = bound_method %int_64, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_14.2: <bound method> = bound_method %int_64, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc6_14.2(%int_64) [concrete = constants.%int_64.198]
// CHECK:STDOUT:     %.loc6_14.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_64.198]
// CHECK:STDOUT:     %.loc6_14.2: %i32 = converted %int_64, %.loc6_14.1 [concrete = constants.%int_64.198]
// CHECK:STDOUT:     %float.make_type: init type = call %Float.ref(%.loc6_14.2) [concrete = f64]
// CHECK:STDOUT:     %.loc6_16.2: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc6_16.3: type = converted %float.make_type, %.loc6_16.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref f64 = bind_name f, %f.var
// CHECK:STDOUT:   %GetFloat.decl: %GetFloat.type = fn_decl @GetFloat [concrete = constants.%GetFloat] {
// CHECK:STDOUT:     %dyn_size.patt: %i32 = binding_pattern dyn_size
// CHECK:STDOUT:     %dyn_size.param_patt: %i32 = value_param_pattern %dyn_size.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %dyn_size.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %dyn_size: %i32 = bind_name dyn_size, %dyn_size.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size.param_patt: %i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: %Float.type = name_ref Float, imports.%Main.Float [concrete = constants.%Float]
// CHECK:STDOUT:   %dyn_size.ref: %i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc9_25.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:   %.loc9_25.2: type = converted %float.make_type, %.loc9_25.1
// CHECK:STDOUT:   return %.loc9_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: f64 = float_literal 0 [concrete = constants.%float]
// CHECK:STDOUT:   assign file.%f.var, %float
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32.be0: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32.be0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9ba: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.6da: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.b97: <witness> = impl_witness (imports.%Core.import_ref.a86), @impl.c81(%int_32.be0) [concrete]
// CHECK:STDOUT:   %Convert.type.ed5: type = fn_type @Convert.2, @impl.c81(%int_32.be0) [concrete]
// CHECK:STDOUT:   %Convert.16d: %Convert.type.ed5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9ba = facet_value Core.IntLiteral, (%impl_witness.b97) [concrete]
// CHECK:STDOUT:   %.b14: type = fn_type_with_self_type %Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.cce: <bound method> = bound_method %int_32.be0, %Convert.16d [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.16d, @Convert.2(%int_32.be0) [concrete]
// CHECK:STDOUT:   %bound_method.480: <bound method> = bound_method %int_32.be0, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_32.4de: %i32 = int_value 32 [concrete]
// CHECK:STDOUT:   %int_64.fab: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Convert.bound.575: <bound method> = bound_method %int_64.fab, %Convert.16d [concrete]
// CHECK:STDOUT:   %bound_method.767: <bound method> = bound_method %int_64.fab, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_64.198: %i32 = int_value 64 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float: %Float.type = import_ref Main//types, Float, loaded [concrete = constants.%Float]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Float = imports.%Main.Float
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .invalid_float = %invalid_float
// CHECK:STDOUT:     .dyn_size = %dyn_size
// CHECK:STDOUT:     .dyn = %dyn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %invalid_float.patt: <error> = binding_pattern invalid_float
// CHECK:STDOUT:     %.loc10_1: <error> = var_pattern %invalid_float.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %invalid_float.var: ref <error> = var invalid_float
// CHECK:STDOUT:   %.loc10_28.1: type = splice_block %.loc10_28.3 [concrete = <error>] {
// CHECK:STDOUT:     %Float.ref.loc10: %Float.type = name_ref Float, imports.%Main.Float [concrete = constants.%Float]
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32.be0]
// CHECK:STDOUT:     %impl.elem0: %.b14 = impl_witness_access constants.%impl_witness.b97, element0 [concrete = constants.%Convert.16d]
// CHECK:STDOUT:     %bound_method.loc10_26.1: <bound method> = bound_method %int_32.loc10, %impl.elem0 [concrete = constants.%Convert.bound.cce]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32.be0) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_26.2: <bound method> = bound_method %int_32.loc10, %specific_fn [concrete = constants.%bound_method.480]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc10_26.2(%int_32.loc10) [concrete = constants.%int_32.4de]
// CHECK:STDOUT:     %.loc10_26.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_32.4de]
// CHECK:STDOUT:     %.loc10_26.2: %i32 = converted %int_32.loc10, %.loc10_26.1 [concrete = constants.%int_32.4de]
// CHECK:STDOUT:     %float.make_type.loc10: init type = call %Float.ref.loc10(%.loc10_26.2) [concrete = <error>]
// CHECK:STDOUT:     %.loc10_28.2: type = value_of_initializer %float.make_type.loc10 [concrete = <error>]
// CHECK:STDOUT:     %.loc10_28.3: type = converted %float.make_type.loc10, %.loc10_28.2 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %invalid_float: <error> = bind_name invalid_float, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %dyn_size.patt: %i32 = binding_pattern dyn_size
// CHECK:STDOUT:     %.loc12_1: %i32 = var_pattern %dyn_size.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %dyn_size.var: ref %i32 = var dyn_size
// CHECK:STDOUT:   %.loc12_15: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32.be0]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32.be0) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %dyn_size: ref %i32 = bind_name dyn_size, %dyn_size.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %dyn.patt: <error> = binding_pattern dyn
// CHECK:STDOUT:     %.loc17_1: <error> = var_pattern %dyn.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %dyn.var: ref <error> = var dyn
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Float.ref.loc17: %Float.type = name_ref Float, imports.%Main.Float [concrete = constants.%Float]
// CHECK:STDOUT:     %dyn_size.ref: ref %i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:     %.loc17_16: %i32 = bind_value %dyn_size.ref
// CHECK:STDOUT:     %float.make_type.loc17: init type = call %Float.ref.loc17(%.loc17_16)
// CHECK:STDOUT:     %.loc17_24.1: type = value_of_initializer %float.make_type.loc17
// CHECK:STDOUT:     %.loc17_24.2: type = converted %float.make_type.loc17, %.loc17_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %dyn: <error> = bind_name dyn, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: %i32) -> type = "float.make_type" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %impl.elem0: %.b14 = impl_witness_access constants.%impl_witness.b97, element0 [concrete = constants.%Convert.16d]
// CHECK:STDOUT:   %bound_method.loc12_1.1: <bound method> = bound_method %int_64, %impl.elem0 [concrete = constants.%Convert.bound.575]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32.be0) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_1.2: <bound method> = bound_method %int_64, %specific_fn [concrete = constants.%bound_method.767]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc12_1.2(%int_64) [concrete = constants.%int_64.198]
// CHECK:STDOUT:   %.loc12: init %i32 = converted %int_64, %int.convert_checked [concrete = constants.%int_64.198]
// CHECK:STDOUT:   assign file.%dyn_size.var, %.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
