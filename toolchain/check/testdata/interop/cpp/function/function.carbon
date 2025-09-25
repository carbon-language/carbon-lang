// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/function.carbon

// ============================================================================
// Global
// ============================================================================

// --- global.h

auto foo() -> void;

// --- import_global.carbon

library "[[@TEST_NAME]]";

import Cpp library "global.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// --- fail_import_global_use_different_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "global.h";

fn MyF() {
  // CHECK:STDERR: fail_import_global_use_different_name.carbon:[[@LINE+4]]:3: error: member name `bar` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.bar();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.bar();
}

// ============================================================================
// Carbon keyword name
// ============================================================================

// --- special_name.h

auto base() -> void;

// --- fail_import_special_name_call_unescaped.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  // CHECK:STDERR: fail_import_special_name_call_unescaped.carbon:[[@LINE+4]]:3: error: member name `base` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.base();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  Cpp.base();
}

// --- import_special_name_call_escaped.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.r#base();
  //@dump-sem-ir-end
}

// ============================================================================
// Overloaded
// ============================================================================

// --- overloaded.h

auto foo() -> void;
auto foo(int value) -> void;

// --- import_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloaded.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Variadic arguments
// ============================================================================

// --- variadic.h

auto foo(int x, ...) -> void;

// --- fail_todo_import_variadic.carbon

library "[[@TEST_NAME]]";

import Cpp library "variadic.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_variadic.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: Variadic function` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(8);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(8);
  //@dump-sem-ir-end
}

// ============================================================================
// Static
// ============================================================================

// --- static.h

static auto foo() -> void;

// --- todo_fail_import_static.carbon

library "[[@TEST_NAME]]";

// TODO: Promote this warning to an error by default.
// CHECK:STDERR: todo_fail_import_static.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./static.h:2:13: warning: function 'foo' has internal linkage but is not defined [CppInteropParseWarning]
// CHECK:STDERR:     2 | static auto foo() -> void;
// CHECK:STDERR:       |             ^
import Cpp library "static.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: todo_fail_import_static.carbon:[[@LINE+4]]:11: note: used here [CppInteropParseNote]
  // CHECK:STDERR:    17 |   Cpp.foo();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
  Cpp.foo();
  //@dump-sem-ir-end
}

// --- template_function.h

template<typename T>
auto foo(T a) -> void;

template<typename T>
auto bar(T a) -> T;

// --- import_template_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "template_function.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1 as i32);
  //@dump-sem-ir-end
}

fn G() -> i32 {
  //@dump-sem-ir-begin
  return Cpp.bar(1 as i32);
  //@dump-sem-ir-end
}

// TODO: Add a test per unsupported param type. See https://github.com/carbon-language/carbon-lang/pull/5477/files/4321e21ed27d987fd71be182d292973fd9849df8#r2094655176

// TODO: Add a test per unsupported return type. See https://github.com/carbon-language/carbon-lang/pull/5477/files/4321e21ed27d987fd71be182d292973fd9849df8#r2094655176

// CHECK:STDOUT: --- import_global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_special_name_call_escaped.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.f7d: type = cpp_overload_set_type @base [concrete]
// CHECK:STDOUT:   %empty_struct: %.f7d = struct_value () [concrete]
// CHECK:STDOUT:   %base.type: type = fn_type @base [concrete]
// CHECK:STDOUT:   %base: %base.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .r#base = %.6f0
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.6f0: %.f7d = cpp_overload_set_value @base [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %base.decl: %base.type = fn_decl @base [concrete = constants.%base] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %base.ref: %.f7d = name_ref r#base, imports.%.6f0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %base.call: init %empty_tuple.type = call imports.%base.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_variadic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_static.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_struct.109: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.565: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.d2c: %Core.IntLiteral.as.As.impl.Convert.type.565 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.080: <witness> = impl_witness imports.%As.impl_witness_table.5ad, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.aaf: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.414: %Core.IntLiteral.as.As.impl.Convert.type.aaf = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.080) [concrete]
// CHECK:STDOUT:   %.351: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.414 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.414, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %.414: type = cpp_overload_set_type @Int.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_struct.540: %.414 = struct_value () [concrete]
// CHECK:STDOUT:   %bar.type: type = fn_type @bar [concrete]
// CHECK:STDOUT:   %bar: %bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .bar = %.146
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @Core.IntLiteral.as.As.impl.Convert [concrete = constants.%empty_struct.109]
// CHECK:STDOUT:   %Core.import_ref.99c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.565) = import_ref Core//prelude/parts/int, loc32_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.d2c)]
// CHECK:STDOUT:   %As.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.99c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.146: %.414 = cpp_overload_set_value @Int.as.As.impl.Convert [concrete = constants.%empty_struct.540]
// CHECK:STDOUT:   %bar.decl: %bar.type = fn_decl @bar [concrete = constants.%bar] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct.109]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.351 = impl_witness_access constants.%As.impl_witness.080, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.414]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_13.2: %i32 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc8_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %bar.ref: %.414 = name_ref bar, imports.%.146 [concrete = constants.%empty_struct.540]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.351 = impl_witness_access constants.%As.impl_witness.080, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.414]
// CHECK:STDOUT:   %bound_method.loc14_20.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_20.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc14_20.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_20.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_20.2: %i32 = converted %int_1, %.loc14_20.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %bar.call: init %i32 = call imports.%bar.decl(%.loc14_20.2)
// CHECK:STDOUT:   return %bar.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
