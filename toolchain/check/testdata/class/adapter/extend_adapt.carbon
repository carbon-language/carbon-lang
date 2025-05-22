// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/extend_adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/extend_adapt.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class SomeClassAdapter;

class SomeClass {
  var a: i32;
  var b: i32;

  fn StaticMemberFunction();

  fn AdapterMethod[self: SomeClassAdapter]();
}

//@dump-sem-ir-begin
class SomeClassAdapter {
  extend adapt SomeClass;
}
//@dump-sem-ir-end

fn TestStaticMemberFunction(a: SomeClassAdapter) {
  //@dump-sem-ir-begin
  a.StaticMemberFunction();
  //@dump-sem-ir-end
}

fn TestAdapterMethod(a: SomeClassAdapter) {
  //@dump-sem-ir-begin
  a.AdapterMethod();
  //@dump-sem-ir-end
}

// --- fail_todo_method_access.carbon

library "[[@TEST_NAME]]";

class SomeClass {
  fn F[self: Self]();
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn F(a: SomeClassAdapter) {
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `SomeClassAdapter` to `SomeClass` [ConversionFailure]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+7]]:3: note: type `SomeClassAdapter` does not implement interface `Core.ImplicitAs(SomeClass)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE-14]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Self]();
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR:
  a.F();
}

// --- fail_todo_field_access.carbon

library "[[@TEST_NAME]]";

class SomeClass {
  var a: i32;
  var b: i32;
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn F(a: SomeClassAdapter) -> i32 {
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `SomeClassAdapter` to `SomeClass` [ConversionFailure]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+4]]:10: note: type `SomeClassAdapter` does not implement interface `Core.ImplicitAs(SomeClass)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.b;
}

// --- fail_todo_adapt_struct.carbon

library "[[@TEST_NAME]]";

class StructAdapter {
  extend adapt {.a: i32, .b: i32};
}

fn F(a: StructAdapter) -> i32 {
  // TODO: This should be allowed.
  // CHECK:STDERR: fail_todo_adapt_struct.carbon:[[@LINE+4]]:10: error: member name `b` not found in `StructAdapter` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.b;
}

// --- fail_todo_adapt_tuple.carbon

library "[[@TEST_NAME]]";

class TupleAdapter {
  extend adapt (i32, i32);
}

fn F(a: TupleAdapter) -> i32 {
  // TODO: This should be allowed.
  // CHECK:STDERR: fail_todo_adapt_tuple.carbon:[[@LINE+4]]:10: error: type `TupleAdapter` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   return a.1;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.1;
}

// --- fail_adapt_builtin.carbon

library "[[@TEST_NAME]]";

fn MakeInt(N: Core.IntLiteral()) -> type = "int.make_type_signed";

class IntAdapter {
  extend adapt MakeInt(32);
}

fn F(a: IntAdapter) -> i32 {
  // Builtin types have no member names.
  // CHECK:STDERR: fail_adapt_builtin.carbon:[[@LINE+4]]:10: error: member name `foo` not found in `IntAdapter` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return a.foo;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a.foo;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [concrete]
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %StaticMemberFunction.type: type = fn_type @StaticMemberFunction [concrete]
// CHECK:STDOUT:   %StaticMemberFunction: %StaticMemberFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %AdapterMethod.type: type = fn_type @AdapterMethod [concrete]
// CHECK:STDOUT:   %AdapterMethod: %AdapterMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc16: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [concrete = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   .SomeClass = <poisoned>
// CHECK:STDOUT:   .StaticMemberFunction = <poisoned>
// CHECK:STDOUT:   .AdapterMethod = <poisoned>
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStaticMemberFunction(%a.param: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %StaticMemberFunction.ref: %StaticMemberFunction.type = name_ref StaticMemberFunction, @SomeClass.%StaticMemberFunction.decl [concrete = constants.%StaticMemberFunction]
// CHECK:STDOUT:   %StaticMemberFunction.call: init %empty_tuple.type = call %StaticMemberFunction.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdapterMethod(%a.param: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %AdapterMethod.ref: %AdapterMethod.type = name_ref AdapterMethod, @SomeClass.%AdapterMethod.decl [concrete = constants.%AdapterMethod]
// CHECK:STDOUT:   %AdapterMethod.bound: <bound method> = bound_method %a.ref, %AdapterMethod.ref
// CHECK:STDOUT:   %AdapterMethod.call: init %empty_tuple.type = call %AdapterMethod.bound(%a.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
