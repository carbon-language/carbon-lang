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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %If1.type: type = fn_type @If1 [template]
// CHECK:STDOUT:   %If1: %If1.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1 [template]
// CHECK:STDOUT:   %If2.type: type = fn_type @If2 [template]
// CHECK:STDOUT:   %If2: %If2.type = struct_value () [template]
// CHECK:STDOUT:   %.28: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.28, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 2 [template]
// CHECK:STDOUT:   %If3.type: type = fn_type @If3 [template]
// CHECK:STDOUT:   %If3: %If3.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
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
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %bool.make_type, %.loc11_11.1 [template = bool]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_20.2: type = converted %int.make_type_32, %.loc11_20.1 [template = i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %If2.decl: %If2.type = fn_decl @If2 [template = constants.%If2] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc22_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc22_11.2: type = converted %bool.make_type, %.loc22_11.1 [template = bool]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc22_20.2: type = converted %int.make_type_32, %.loc22_20.1 [template = i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %If3.decl: %If3.type = fn_decl @If3 [template = constants.%If3] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc33_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc33_11.2: type = converted %bool.make_type, %.loc33_11.1 [template = bool]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc33_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc33_20.2: type = converted %int.make_type_32, %.loc33_20.1 [template = i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If1(%b.param_patt: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc13_12: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_13.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_13.2: <bound method> = bound_method %.loc13_12, %.loc13_13.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc13_13.2(%.loc13_12) [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_13.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_13.4: i32 = converted %.loc13_12, %.loc13_13.3 [template = constants.%.27]
// CHECK:STDOUT:   return %.loc13_13.4
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If2(%b.param_patt: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc25_12: Core.IntLiteral = int_value 2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc25_13.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc25_13.2: <bound method> = bound_method %.loc25_12, %.loc25_13.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc25_13.2(%.loc25_12) [template = constants.%.30]
// CHECK:STDOUT:   %.loc25_13.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.30]
// CHECK:STDOUT:   %.loc25_13.4: i32 = converted %.loc25_12, %.loc25_13.3 [template = constants.%.30]
// CHECK:STDOUT:   return %.loc25_13.4
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If3(%b.param_patt: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc35_12: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc35_13.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc35_13.2: <bound method> = bound_method %.loc35_12, %.loc35_13.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc35_13.2(%.loc35_12) [template = constants.%.27]
// CHECK:STDOUT:   %.loc35_13.3: i32 = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc35_13.4: i32 = converted %.loc35_12, %.loc35_13.3 [template = constants.%.27]
// CHECK:STDOUT:   return %.loc35_13.4
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT: }
// CHECK:STDOUT:
