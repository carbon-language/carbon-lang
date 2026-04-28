// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/inline.carbon

// ============================================================================
// With definition
// ============================================================================

// --- with_definition.h

inline void foo() {}

// --- import_with_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Without definition
// ============================================================================

// --- without_definition.h

inline void foo();

// --- todo_fail_import_without_definition.carbon

library "[[@TEST_NAME]]";

// TODO: Promote this warning to an error by default.
// CHECK:STDERR: todo_fail_import_without_definition.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./without_definition.h:2:13: warning: inline function 'foo' is not defined [CppInteropParseWarning]
// CHECK:STDERR:     2 | inline void foo();
// CHECK:STDERR:       |             ^
import Cpp library "without_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: todo_fail_import_without_definition.carbon:[[@LINE+4]]:11: note: used here [CppInteropParseNote]
  // CHECK:STDERR:    17 |   Cpp.foo();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
  Cpp.foo();

  // Don't error on repeated calls.
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// automatic return type
// ============================================================================

// --- automatic_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
inline auto WithoutThunk(int) -> auto { return 0; }
inline auto ThunkOnArg(short) -> auto { return 0; }
inline auto ThunkOnReturn(int) -> auto { short x = 0; return x; }
inline auto ThunkOnBoth(short) -> auto { short x = 0; return x; }
''';

fn MyF() {
  //@dump-sem-ir-begin
  let unused r1: i32 = Cpp.WithoutThunk(1);
  let unused r2: i32 = Cpp.ThunkOnArg(1);
  let unused r3: i16 = Cpp.ThunkOnReturn(1);
  let unused r4: i16 = Cpp.ThunkOnBoth(1);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_with_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_without_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc17: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc17: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc20: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc20: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- automatic_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.type: type = cpp_overload_set_type @WithoutThunk.cpp_overload_set [concrete]
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.value: %WithoutThunk.cpp_overload_set.type = cpp_overload_set_value @WithoutThunk.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %WithoutThunk.type: type = fn_type @WithoutThunk [concrete]
// CHECK:STDOUT:   %WithoutThunk: %WithoutThunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnArg.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.value: %ThunkOnArg.cpp_overload_set.type = cpp_overload_set_value @ThunkOnArg.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.type: type = fn_type @ThunkOnArg__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk: %ThunkOnArg__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.value: %ThunkOnReturn.cpp_overload_set.type = cpp_overload_set_value @ThunkOnReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.type: type = fn_type @ThunkOnReturn__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk: %ThunkOnReturn__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnBoth.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.value: %ThunkOnBoth.cpp_overload_set.type = cpp_overload_set_value @ThunkOnBoth.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.type: type = fn_type @ThunkOnBoth__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk: %ThunkOnBoth__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc16_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .WithoutThunk = %WithoutThunk.cpp_overload_set.value
// CHECK:STDOUT:     .ThunkOnArg = %ThunkOnArg.cpp_overload_set.value
// CHECK:STDOUT:     .ThunkOnReturn = %ThunkOnReturn.cpp_overload_set.value
// CHECK:STDOUT:     .ThunkOnBoth = %ThunkOnBoth.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.value: %WithoutThunk.cpp_overload_set.type = cpp_overload_set_value @WithoutThunk.cpp_overload_set [concrete = constants.%WithoutThunk.cpp_overload_set.value]
// CHECK:STDOUT:   %WithoutThunk.decl: %WithoutThunk.type = fn_decl @WithoutThunk [concrete = constants.%WithoutThunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.value: %ThunkOnArg.cpp_overload_set.type = cpp_overload_set_value @ThunkOnArg.cpp_overload_set [concrete = constants.%ThunkOnArg.cpp_overload_set.value]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.decl: %ThunkOnArg__carbon_thunk.type = fn_decl @ThunkOnArg__carbon_thunk [concrete = constants.%ThunkOnArg__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.value: %ThunkOnReturn.cpp_overload_set.type = cpp_overload_set_value @ThunkOnReturn.cpp_overload_set [concrete = constants.%ThunkOnReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.decl: %ThunkOnReturn__carbon_thunk.type = fn_decl @ThunkOnReturn__carbon_thunk [concrete = constants.%ThunkOnReturn__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.value: %ThunkOnBoth.cpp_overload_set.type = cpp_overload_set_value @ThunkOnBoth.cpp_overload_set [concrete = constants.%ThunkOnBoth.cpp_overload_set.value]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.decl: %ThunkOnBoth__carbon_thunk.type = fn_decl @ThunkOnBoth__carbon_thunk [concrete = constants.%ThunkOnBoth__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r1.patt: %pattern_type.7ce = value_binding_pattern r1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %WithoutThunk.ref: %WithoutThunk.cpp_overload_set.type = name_ref WithoutThunk, imports.%WithoutThunk.cpp_overload_set.value [concrete = constants.%WithoutThunk.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %WithoutThunk.call: init %i32 = call imports.%WithoutThunk.decl(%_.var.1)
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %WithoutThunk.call
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %WithoutThunk.call, %.loc13_42.1
// CHECK:STDOUT:   %r1: %i32 = value_binding r1, %.loc13_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r2.patt: %pattern_type.7ce = value_binding_pattern r2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnArg.ref: %ThunkOnArg.cpp_overload_set.type = name_ref ThunkOnArg, imports.%ThunkOnArg.cpp_overload_set.value [concrete = constants.%ThunkOnArg.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc14: %ptr = addr_of %_.var.2
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.call: init %i32 = call imports.%ThunkOnArg__carbon_thunk.decl(%addr.loc14)
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_40.1: %i32 = value_of_initializer %ThunkOnArg__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_40.2: %i32 = converted %ThunkOnArg__carbon_thunk.call, %.loc14_40.1
// CHECK:STDOUT:   %r2: %i32 = value_binding r2, %.loc14_40.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r3.patt: %pattern_type.2f8 = value_binding_pattern r3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnReturn.ref: %ThunkOnReturn.cpp_overload_set.type = name_ref ThunkOnReturn, imports.%ThunkOnReturn.cpp_overload_set.value [concrete = constants.%ThunkOnReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_43.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc15: %ptr = addr_of %.loc15_43.1
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnReturn__carbon_thunk.decl(%_.var.3, %addr.loc15)
// CHECK:STDOUT:   %.loc15_43.2: init %i16 to %.loc15_43.1 = mark_in_place_init %ThunkOnReturn__carbon_thunk.call
// CHECK:STDOUT:   %i16.loc15: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %.loc15_43.3: ref %i16 = temporary %.loc15_43.1, %.loc15_43.2
// CHECK:STDOUT:   %.loc15_43.4: %i16 = acquire_value %.loc15_43.3
// CHECK:STDOUT:   %r3: %i16 = value_binding r3, %.loc15_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r4.patt: %pattern_type.2f8 = value_binding_pattern r4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnBoth.ref: %ThunkOnBoth.cpp_overload_set.type = name_ref ThunkOnBoth, imports.%ThunkOnBoth.cpp_overload_set.value [concrete = constants.%ThunkOnBoth.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc16_41.1: %ptr = addr_of %_.var.4
// CHECK:STDOUT:   %.loc16_41.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc16_41.2: %ptr = addr_of %.loc16_41.1
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnBoth__carbon_thunk.decl(%addr.loc16_41.1, %addr.loc16_41.2)
// CHECK:STDOUT:   %.loc16_41.2: init %i16 to %.loc16_41.1 = mark_in_place_init %ThunkOnBoth__carbon_thunk.call
// CHECK:STDOUT:   %i16.loc16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %.loc16_41.3: ref %i16 = temporary %.loc16_41.1, %.loc16_41.2
// CHECK:STDOUT:   %.loc16_41.4: %i16 = acquire_value %.loc16_41.3
// CHECK:STDOUT:   %r4: %i16 = value_binding r4, %.loc16_41.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %.loc16_41.3, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc16: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%.loc16_41.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc15: <bound method> = bound_method %.loc15_43.3, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc15: init %empty_tuple.type = call %Destroy.Op.bound.loc15(%.loc15_43.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_41.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_41.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
