// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/return.carbon

// ============================================================================
// Multiple function with a return value
// ============================================================================

// --- multiple.h

auto foo1() -> short;
auto foo2() -> int;

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn IngestI16(x: i16);
fn IngestI32(x: i32);

fn F() {
  //@dump-sem-ir-begin
  IngestI16(Cpp.foo1());
  IngestI32(Cpp.foo2());
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %IngestI16.type: type = fn_type @IngestI16 [concrete]
// CHECK:STDOUT:   %IngestI16: %IngestI16.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %IngestI32.type: type = fn_type @IngestI32 [concrete]
// CHECK:STDOUT:   %IngestI32: %IngestI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo1.type: type = fn_type @foo1 [concrete]
// CHECK:STDOUT:   %foo1: %foo1.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo1__carbon_thunk.type: type = fn_type @foo1__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo1__carbon_thunk: %foo1__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo2.type: type = fn_type @foo2 [concrete]
// CHECK:STDOUT:   %foo2: %foo2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo1 = %foo1.decl
// CHECK:STDOUT:     .foo2 = %foo2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo1.decl: %foo1.type = fn_decl @foo1 [concrete = constants.%foo1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo1__carbon_thunk.decl: %foo1__carbon_thunk.type = fn_decl @foo1__carbon_thunk [concrete = constants.%foo1__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo2.decl: %foo2.type = fn_decl @foo2 [concrete = constants.%foo2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestI16.ref: %IngestI16.type = name_ref IngestI16, file.%IngestI16.decl [concrete = constants.%IngestI16]
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo1.ref: %foo1.type = name_ref foo1, imports.%foo1.decl [concrete = constants.%foo1]
// CHECK:STDOUT:   %.loc11_22.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc11_22.1
// CHECK:STDOUT:   %foo1__carbon_thunk.call: init %empty_tuple.type = call imports.%foo1__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc11_22.2: init %i16 = in_place_init %foo1__carbon_thunk.call, %.loc11_22.1
// CHECK:STDOUT:   %.loc11_22.3: %i16 = value_of_initializer %.loc11_22.2
// CHECK:STDOUT:   %.loc11_22.4: %i16 = converted %.loc11_22.2, %.loc11_22.3
// CHECK:STDOUT:   %IngestI16.call: init %empty_tuple.type = call %IngestI16.ref(%.loc11_22.4)
// CHECK:STDOUT:   %IngestI32.ref: %IngestI32.type = name_ref IngestI32, file.%IngestI32.decl [concrete = constants.%IngestI32]
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo2.ref: %foo2.type = name_ref foo2, imports.%foo2.decl [concrete = constants.%foo2]
// CHECK:STDOUT:   %foo2.call: init %i32 = call %foo2.ref()
// CHECK:STDOUT:   %.loc12_22.1: %i32 = value_of_initializer %foo2.call
// CHECK:STDOUT:   %.loc12_22.2: %i32 = converted %foo2.call, %.loc12_22.1
// CHECK:STDOUT:   %IngestI32.call: init %empty_tuple.type = call %IngestI32.ref(%.loc12_22.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
