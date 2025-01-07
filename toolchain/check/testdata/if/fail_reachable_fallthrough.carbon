// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if/fail_reachable_fallthrough.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if/fail_reachable_fallthrough.carbon

fn If1(b: bool) -> i32 {
  if (b) {
    return 1;
  } else {
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+4]]:1: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

fn If2(b: bool) -> i32 {
  if (b) {
  } else {
    return 2;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+4]]:1: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

fn If3(b: bool) -> i32 {
  if (b) {
    return 1;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: --- fail_reachable_fallthrough.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %If1.type: type = fn_type @If1 [template]
// CHECK:STDOUT:   %If1: %If1.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %If2.type: type = fn_type @If2 [template]
// CHECK:STDOUT:   %If2: %If2.type = struct_value () [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %If3.type: type = fn_type @If3 [template]
// CHECK:STDOUT:   %If3: %If3.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .If1 = %If1.decl
// CHECK:STDOUT:     .If2 = %If2.decl
// CHECK:STDOUT:     .If3 = %If3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %If1.decl: %If1.type = fn_decl @If1 [template = constants.%If1] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_11.1: type = splice_block %.loc11_11.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc11_11.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc11_11.3: type = converted %bool.make_type, %.loc11_11.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %If2.decl: %If2.type = fn_decl @If2 [template = constants.%If2] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %.loc22_11.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc22_11.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %bool.make_type, %.loc22_11.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %If3.decl: %If3.type = fn_decl @If3 [template = constants.%If3] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc33_11.1: type = splice_block %.loc33_11.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc33_11.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc33_11.3: type = converted %bool.make_type, %.loc33_11.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If1(%b.param_patt: bool) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc13_13.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc13_13.2: %i32 = converted %int_1, %.loc13_13.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   return %.loc13_13.2
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If2(%b.param_patt: bool) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc25_13.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc25_13.2: %i32 = converted %int_2, %.loc25_13.1 [template = constants.%int_2.166]
// CHECK:STDOUT:   return %.loc25_13.2
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If3(%b.param_patt: bool) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc35_13.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc35_13.2: %i32 = converted %int_1, %.loc35_13.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   return %.loc35_13.2
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT: }
// CHECK:STDOUT:
