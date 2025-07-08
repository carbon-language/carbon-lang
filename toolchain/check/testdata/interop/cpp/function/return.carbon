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
// short
// ============================================================================

// --- short.h

auto foo_short() -> short;

// --- import_short.carbon

library "[[@TEST_NAME]]";

import Cpp library "short.h";

fn F() {
  //@dump-sem-ir-begin
  let x: i16 = Cpp.foo_short();
  //@dump-sem-ir-end
}

// ============================================================================
// int
// ============================================================================

// --- int.h

auto foo_int() -> int;

// --- import_int.carbon

library "[[@TEST_NAME]]";

import Cpp library "int.h";

fn Carbon_foo(val: i32);

fn F() {
  //@dump-sem-ir-begin
  Carbon_foo(Cpp.foo_int());
  //@dump-sem-ir-end
}

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

// ============================================================================
// float
// ============================================================================

// --- float.h

auto foo_float() -> float;

// --- fail_todo_import_float.carbon

library "[[@TEST_NAME]]";

import Cpp library "float.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_float.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: return type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo_float();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_float.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo_float` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo_float();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo_float();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_short.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short.type: type = fn_type @foo_short [concrete]
// CHECK:STDOUT:   %foo_short: %foo_short.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_short = %foo_short.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_short.decl: %foo_short.type = fn_decl @foo_short [concrete = constants.%foo_short] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.2f8 = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_short.ref: %foo_short.type = name_ref foo_short, imports.%foo_short.decl [concrete = constants.%foo_short]
// CHECK:STDOUT:   %foo_short.call: init %i16 = call %foo_short.ref()
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.1: %i16 = value_of_initializer %foo_short.call
// CHECK:STDOUT:   %.loc8_30.2: %i16 = converted %foo_short.call, %.loc8_30.1
// CHECK:STDOUT:   %x: %i16 = bind_name x, %.loc8_30.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Carbon_foo.type: type = fn_type @Carbon_foo [concrete]
// CHECK:STDOUT:   %Carbon_foo: %Carbon_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_int.type: type = fn_type @foo_int [concrete]
// CHECK:STDOUT:   %foo_int: %foo_int.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_int = %foo_int.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_int.decl: %foo_int.type = fn_decl @foo_int [concrete = constants.%foo_int] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Carbon_foo.ref: %Carbon_foo.type = name_ref Carbon_foo, file.%Carbon_foo.decl [concrete = constants.%Carbon_foo]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_int.ref: %foo_int.type = name_ref foo_int, imports.%foo_int.decl [concrete = constants.%foo_int]
// CHECK:STDOUT:   %foo_int.call: init %i32 = call %foo_int.ref()
// CHECK:STDOUT:   %.loc10_26.1: %i32 = value_of_initializer %foo_int.call
// CHECK:STDOUT:   %.loc10_26.2: %i32 = converted %foo_int.call, %.loc10_26.1
// CHECK:STDOUT:   %Carbon_foo.call: init %empty_tuple.type = call %Carbon_foo.ref(%.loc10_26.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
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
// CHECK:STDOUT:   %foo1.call: init %i16 = call %foo1.ref()
// CHECK:STDOUT:   %.loc11_22.1: %i16 = value_of_initializer %foo1.call
// CHECK:STDOUT:   %.loc11_22.2: %i16 = converted %foo1.call, %.loc11_22.1
// CHECK:STDOUT:   %IngestI16.call: init %empty_tuple.type = call %IngestI16.ref(%.loc11_22.2)
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
// CHECK:STDOUT: --- fail_todo_import_float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_float = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_float.ref: <error> = name_ref foo_float, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
