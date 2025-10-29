// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/import.carbon

// ============================================================================
// Import C++ namespace indirectly
// ============================================================================

// --- namespace.h

namespace MyNamespace {

class MyClass {};

}  // namespace MyNamespace

// --- namespace_api.carbon

library "[[@TEST_NAME]]";

import Cpp library "namespace.h";

alias MyNamespaceAlias = Cpp.MyNamespace;

// --- fail_import_namespace_api.carbon

library "[[@TEST_NAME]]";

import library "namespace_api";

fn F() {
  // Imports are not implicitly re-exported, so lookup is expected to fail.
  // CHECK:STDERR: fail_import_namespace_api.carbon:[[@LINE+4]]:10: error: member name `MyClass` not found in `Cpp.MyNamespace` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   var x: MyNamespaceAlias.MyClass;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var x: MyNamespaceAlias.MyClass;
}

// ============================================================================
// Import C++ struct indirectly
// ============================================================================

// --- struct.h

struct MyStruct { void Foo(); };

// --- struct_api.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

alias MyStructAlias = Cpp.MyStruct;

// --- fail_todo_import_struct_api.carbon
// CHECK:STDERR: fail_todo_import_struct_api.carbon: error: semantics TODO: `Unsupported: Importing C++ indirectly` [SemanticsTodo]
// CHECK:STDERR:

library "[[@TEST_NAME]]";

import library "struct_api";

fn F() {
  //@dump-sem-ir-begin
  var x: MyStructAlias;
  x.Foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Import C++ function indirectly
// ============================================================================

// --- function.h

auto foo_short(short x) -> void;
auto foo_int(int x) -> void;

// --- function_api.carbon

library "[[@TEST_NAME]]";

import Cpp library "function.h";

alias FooShort = Cpp.foo_short;
alias FooInt = Cpp.foo_int;

// --- fail_todo_import_function_api.carbon
// CHECK:STDERR: fail_todo_import_function_api.carbon: error: semantics TODO: `Unsupported: Importing C++ function `foo_short` indirectly` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_function_api.carbon: error: semantics TODO: `Unsupported: Importing C++ function `foo_short` indirectly` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_function_api.carbon: error: semantics TODO: `Unsupported: Importing C++ function `foo_int` indirectly` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_function_api.carbon: error: semantics TODO: `Unsupported: Importing C++ function `foo_int` indirectly` [SemanticsTodo]
// CHECK:STDERR:

library "[[@TEST_NAME]]";

import library "function_api";

fn F() {
  //@dump-sem-ir-begin
  FooShort(8 as i16);
  FooInt(9);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_import_struct_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.MyStructAlias: type = import_ref Main//struct_api, MyStructAlias, loaded [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: <error> = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref <error> = var %x.var_patt [concrete = <error>]
// CHECK:STDOUT:   %MyStructAlias.ref: type = name_ref MyStructAlias, imports.%Main.MyStructAlias [concrete = <error>]
// CHECK:STDOUT:   %x: ref <error> = ref_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref: ref <error> = name_ref x, %x [concrete = <error>]
// CHECK:STDOUT:   %Foo.ref: <error> = name_ref Foo, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_function_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.771: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c72: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2f9: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.228: %Core.IntLiteral.as.As.impl.Convert.type.2f9 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.771 = facet_value Core.IntLiteral, (%As.impl_witness.c72) [concrete]
// CHECK:STDOUT:   %.28a: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.228 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.228, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.823: %i16 = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FooShort: <error> = import_ref Main//function_api, FooShort, loaded [concrete = <error>]
// CHECK:STDOUT:   %Main.FooInt: <error> = import_ref Main//function_api, FooInt, loaded [concrete = <error>]
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FooShort.ref: <error> = name_ref FooShort, imports.%Main.FooShort [concrete = <error>]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.28a = impl_witness_access constants.%As.impl_witness.c72, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.228]
// CHECK:STDOUT:   %bound_method.loc16_14.1: <bound method> = bound_method %int_8, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_14.2: <bound method> = bound_method %int_8, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc16_14.2(%int_8) [concrete = constants.%int_8.823]
// CHECK:STDOUT:   %.loc16_14.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.823]
// CHECK:STDOUT:   %.loc16_14.2: %i16 = converted %int_8, %.loc16_14.1 [concrete = constants.%int_8.823]
// CHECK:STDOUT:   %FooInt.ref: <error> = name_ref FooInt, imports.%Main.FooInt [concrete = <error>]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
