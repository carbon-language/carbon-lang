// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function_decl.carbon

// --- int_return_function_decl.h

auto foo_int() -> int;

// --- import_int_return_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_return_function_decl.h";

fn Carbon_foo(val: i32);

fn F() {
  Carbon_foo(Cpp.foo_int());
}

// --- float_return_function_decl.h

auto foo_float() -> float;

// --- fail_import_float_return_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "float_return_function_decl.h";

fn F() {
  // CHECK:STDERR: fail_import_float_return_function_decl.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: return type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo_float();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_float_return_function_decl.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo_float` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo_float();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo_float();
}

// CHECK:STDOUT: --- import_int_return_function_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Carbon_foo.type: type = fn_type @Carbon_foo [concrete]
// CHECK:STDOUT:   %Carbon_foo: %Carbon_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_int.type: type = fn_type @foo_int [concrete]
// CHECK:STDOUT:   %foo_int: %foo_int.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_int = @F.%foo_int.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .Carbon_foo = %Carbon_foo.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "int_return_function_decl.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Carbon_foo.decl: %Carbon_foo.type = fn_decl @Carbon_foo [concrete = constants.%Carbon_foo] {
// CHECK:STDOUT:     %val.patt: %i32 = binding_pattern val
// CHECK:STDOUT:     %val.param_patt: %i32 = value_param_pattern %val.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %val.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %val: %i32 = bind_name val, %val.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Carbon_foo(%val.param_patt: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Carbon_foo.ref: %Carbon_foo.type = name_ref Carbon_foo, file.%Carbon_foo.decl [concrete = constants.%Carbon_foo]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %foo_int.decl: %foo_int.type = fn_decl @foo_int [concrete = constants.%foo_int] {} {}
// CHECK:STDOUT:   %foo_int.ref: %foo_int.type = name_ref foo_int, %foo_int.decl [concrete = constants.%foo_int]
// CHECK:STDOUT:   %foo_int.call: init %i32 = call %foo_int.ref()
// CHECK:STDOUT:   %.loc9_26.1: %i32 = value_of_initializer %foo_int.call
// CHECK:STDOUT:   %.loc9_26.2: %i32 = converted %foo_int.call, %.loc9_26.1
// CHECK:STDOUT:   %Carbon_foo.call: init %empty_tuple.type = call %Carbon_foo.ref(%.loc9_26.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo_int[]() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_float_return_function_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_float = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "float_return_function_decl.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_float.ref: <error> = name_ref foo_float, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
