// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/implicit_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/implicit_as.carbon

// --- implicit_conversions.h

struct Dest {};

struct Source {
  operator Dest() const;
};

struct Source2 {
  int x;
};
struct Dest2 {
  Dest2(Source2);
};

struct NonConstConversion {
  operator Dest();
};

struct InaccessibleConstructor {
private:
  InaccessibleConstructor(Source);
};

struct InaccessibleConversion {
private:
  operator Dest() const;
};

struct DeletedConstructor {
  DeletedConstructor(Source) = delete;
};

struct DeletedConversion {
  operator Dest() const = delete;
};

struct IntConstructor {
  IntConstructor(int);
};

struct ExplicitConstructor {
  explicit ExplicitConstructor(Source);
};

struct ExplicitConversion {
  explicit operator Dest() const;
};

// --- implicit_conversions.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn UserConversion(s: Cpp.Source) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

fn NonConstConversion(ref s: Cpp.NonConstConversion) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

fn ConstructorConversion(s: Cpp.Source2) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest2 = s;
  //@dump-sem-ir-end
}

// --- fail_expr_category.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn NonConstConversionTest(s: Cpp.NonConstConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+7]]:21: error: cannot implicitly convert expression of type `Cpp.NonConstConversion` to `Cpp.Dest` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:21: note: type `Cpp.NonConstConversion` does not implement interface `Core.ImplicitAs(Cpp.Dest)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_access.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn InaccessibleConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_access.carbon:[[@LINE+8]]:40: error: cannot access private member `InaccessibleConstructor` of type `Cpp.InaccessibleConstructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let _: Cpp.InaccessibleConstructor = s;
  // CHECK:STDERR:                                        ^
  // CHECK:STDERR: fail_access.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./implicit_conversions.h:21:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   InaccessibleConstructor(Source);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.InaccessibleConstructor = s;
  //@dump-sem-ir-end
}

fn InaccessibleConversionTest(s: Cpp.InaccessibleConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_access.carbon:[[@LINE+8]]:21: error: cannot access private member `<C++ operator>` of type `Cpp.InaccessibleConversion` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_access.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./implicit_conversions.h:26:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   operator Dest() const;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn DeletedConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+7]]:35: error: cannot implicitly convert expression of type `Cpp.Source` to `Cpp.DeletedConstructor` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.DeletedConstructor = s;
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+4]]:35: note: type `Cpp.Source` does not implement interface `Core.ImplicitAs(Cpp.DeletedConstructor)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.DeletedConstructor = s;
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  let _: Cpp.DeletedConstructor = s;
  //@dump-sem-ir-end
}

fn DeletedConversionTest(s: Cpp.DeletedConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+7]]:21: error: cannot implicitly convert expression of type `Cpp.DeletedConversion` to `Cpp.Dest` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+4]]:21: note: type `Cpp.DeletedConversion` does not implement interface `Core.ImplicitAs(Cpp.Dest)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_explicit.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn ExplicitConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+7]]:36: error: cannot implicitly convert expression of type `Cpp.Source` to `Cpp.ExplicitConstructor` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.ExplicitConstructor = s;
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+4]]:36: note: type `Cpp.Source` does not implement interface `Core.ImplicitAs(Cpp.ExplicitConstructor)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.ExplicitConstructor = s;
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR:
  let _: Cpp.ExplicitConstructor = s;
  //@dump-sem-ir-end
}

fn ExplicitConversionTest(s: Cpp.ExplicitConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+7]]:21: error: cannot implicitly convert expression of type `Cpp.ExplicitConversion` to `Cpp.Dest` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+4]]:21: note: type `Cpp.ExplicitConversion` does not implement interface `Core.ImplicitAs(Cpp.Dest)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- i32_to_int.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn IntConstructor(i: i32) {
  //@dump-sem-ir-begin
  let _: Cpp.IntConstructor = i;
  //@dump-sem-ir-end
}

// --- fail_no_u32_to_int_conversion_in_carbon.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn IntConstructorTest(u: u32) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE+11]]:31: error: cannot implicitly convert expression of type `u32` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.IntConstructor = u;
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE+8]]:31: note: type `u32` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.IntConstructor = u;
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./implicit_conversions.h:38:21: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   IntConstructor(int);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.IntConstructor = u;
  //@dump-sem-ir-end
}

// --- default_constructor.h

struct DefaultConstructor {
  DefaultConstructor();
};

// --- default_constructor.carbon

library "[[TEST_NAME]]";

import Cpp library "default_constructor.h";

fn DefaultConstructorTest() {
  //@dump-sem-ir-begin
  // OK, empty tuple maps to `{}` initializer list.
  let _: Cpp.DefaultConstructor = ();
  //@dump-sem-ir-end
}

// --- multi_argument.h

struct Two {
  Two(int, int);
};

struct ThreeWithDefault {
  ThreeWithDefault(int, int, int = 0);
};

// --- construct_multi_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_argument.h";

fn ImplicitConvert() {
  //@dump-sem-ir-begin
  let _: Cpp.Two = (1, 2);

  let _: Cpp.ThreeWithDefault = (1, 2);

  let _: Cpp.ThreeWithDefault = (1, 2, 3);
  //@dump-sem-ir-end
}

// --- aggregate.h

struct Aggregate {
  int x = 0;
  int y = 0;
};

struct NonAggregate {
  NonAggregate(int x = 0, int y = 0);
};

// --- fail_todo_aggregate_from_tuple.carbon

library "[[@TEST_NAME]]";

import Cpp library "aggregate.h";

fn InitFromTuple() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_aggregate_from_tuple.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `()` to `Cpp.Aggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Aggregate = ();
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR: fail_todo_aggregate_from_tuple.carbon:[[@LINE+4]]:26: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.Aggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Aggregate = ();
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR:
  let _: Cpp.Aggregate = ();
  // CHECK:STDERR: fail_todo_aggregate_from_tuple.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `(Core.IntLiteral,)` to `Cpp.Aggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Aggregate = (1,);
  // CHECK:STDERR:                          ^~~~
  // CHECK:STDERR: fail_todo_aggregate_from_tuple.carbon:[[@LINE+4]]:26: note: type `(Core.IntLiteral,)` does not implement interface `Core.ImplicitAs(Cpp.Aggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Aggregate = (1,);
  // CHECK:STDERR:                          ^~~~
  // CHECK:STDERR:
  let _: Cpp.Aggregate = (1,);
  // CHECK:STDERR: fail_todo_aggregate_from_tuple.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `(Core.IntLiteral, Core.IntLiteral)` to `Cpp.Aggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Aggregate = (1, 2);
  // CHECK:STDERR:                          ^~~~~~
  // CHECK:STDERR: fail_todo_aggregate_from_tuple.carbon:[[@LINE+4]]:26: note: type `(Core.IntLiteral, Core.IntLiteral)` does not implement interface `Core.ImplicitAs(Cpp.Aggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Aggregate = (1, 2);
  // CHECK:STDERR:                          ^~~~~~
  // CHECK:STDERR:
  let _: Cpp.Aggregate = (1, 2);
  //@dump-sem-ir-end
}

// --- fail_todo_aggregate_from_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "aggregate.h";

fn InitFromStruct() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_aggregate_from_struct.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `{}` to `Cpp.Aggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Aggregate = {};
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR: fail_todo_aggregate_from_struct.carbon:[[@LINE+4]]:26: note: type `{}` does not implement interface `Core.ImplicitAs(Cpp.Aggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Aggregate = {};
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR:
  let _: Cpp.Aggregate = {};
  // CHECK:STDERR: fail_todo_aggregate_from_struct.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `{.x: Core.IntLiteral}` to `Cpp.Aggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Aggregate = {.x = 1};
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR: fail_todo_aggregate_from_struct.carbon:[[@LINE+4]]:26: note: type `{.x: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.Aggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Aggregate = {.x = 1};
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR:
  let _: Cpp.Aggregate = {.x = 1};
  // CHECK:STDERR: fail_todo_aggregate_from_struct.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `{.x: Core.IntLiteral, .y: Core.IntLiteral}` to `Cpp.Aggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Aggregate = {.x = 1, .y = 2};
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_aggregate_from_struct.carbon:[[@LINE+4]]:26: note: type `{.x: Core.IntLiteral, .y: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.Aggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.Aggregate = {.x = 1, .y = 2};
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let _: Cpp.Aggregate = {.x = 1, .y = 2};
  //@dump-sem-ir-end
}

// --- non_aggregate_from_tuple.carbon

library "[[@TEST_NAME]]";

import Cpp library "aggregate.h";

fn InitFromStruct() {
  //@dump-sem-ir-begin
  let _: Cpp.NonAggregate = ();
  let _: Cpp.NonAggregate = (1,);
  let _: Cpp.NonAggregate = (1, 2);
  //@dump-sem-ir-end
}

// --- fail_non_aggregate_from_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "aggregate.h";

fn InitFromStruct() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_non_aggregate_from_struct.carbon:[[@LINE+7]]:29: error: cannot implicitly convert expression of type `{}` to `Cpp.NonAggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.NonAggregate = {};
  // CHECK:STDERR:                             ^~
  // CHECK:STDERR: fail_non_aggregate_from_struct.carbon:[[@LINE+4]]:29: note: type `{}` does not implement interface `Core.ImplicitAs(Cpp.NonAggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.NonAggregate = {};
  // CHECK:STDERR:                             ^~
  // CHECK:STDERR:
  let _: Cpp.NonAggregate = {};
  // CHECK:STDERR: fail_non_aggregate_from_struct.carbon:[[@LINE+7]]:29: error: cannot implicitly convert expression of type `{.x: Core.IntLiteral}` to `Cpp.NonAggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.NonAggregate = {.x = 1};
  // CHECK:STDERR:                             ^~~~~~~~
  // CHECK:STDERR: fail_non_aggregate_from_struct.carbon:[[@LINE+4]]:29: note: type `{.x: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.NonAggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.NonAggregate = {.x = 1};
  // CHECK:STDERR:                             ^~~~~~~~
  // CHECK:STDERR:
  let _: Cpp.NonAggregate = {.x = 1};
  // CHECK:STDERR: fail_non_aggregate_from_struct.carbon:[[@LINE+7]]:29: error: cannot implicitly convert expression of type `{.x: Core.IntLiteral, .y: Core.IntLiteral}` to `Cpp.NonAggregate` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.NonAggregate = {.x = 1, .y = 2};
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non_aggregate_from_struct.carbon:[[@LINE+4]]:29: note: type `{.x: Core.IntLiteral, .y: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.NonAggregate)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.NonAggregate = {.x = 1, .y = 2};
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let _: Cpp.NonAggregate = {.x = 1, .y = 2};
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit_conversions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %pattern_type.78c: type = pattern_type %Source [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %Source.cpp_operator.type: type = fn_type @Source.cpp_operator [concrete]
// CHECK:STDOUT:   %Source.cpp_operator: %Source.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.1: type = fn_type @Dest__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.1: %Dest__carbon_thunk.type.2ffc9e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonConstConversion.480: type = class_type @NonConstConversion.1 [concrete]
// CHECK:STDOUT:   %pattern_type.b08: type = pattern_type %NonConstConversion.480 [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.type: type = fn_type @NonConstConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator: %NonConstConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.2: %Dest__carbon_thunk.type.2ffc9e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.39d: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %ptr.472: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.9ae: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor.type: type = fn_type @Dest2.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor: %Dest2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.Op.type: type = fn_type @Dest2.Op [concrete]
// CHECK:STDOUT:   %Dest2.Op: %Dest2.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     .NonConstConversion = %NonConstConversion.decl
// CHECK:STDOUT:     .Source2 = %Source2.decl
// CHECK:STDOUT:     .Dest2 = %Dest2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %Source.cpp_operator.decl: %Source.cpp_operator.type = fn_decl @Source.cpp_operator [concrete = constants.%Source.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.78c = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.78c = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Source = value_param call_param0
// CHECK:STDOUT:     %self: %Source = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.b58ecd.1: %Dest__carbon_thunk.type.2ffc9e.1 = fn_decl @Dest__carbon_thunk.1 [concrete = constants.%Dest__carbon_thunk.f8fa06.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion.1 [concrete = constants.%NonConstConversion.480] {} {}
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.decl: %NonConstConversion.cpp_operator.type = fn_decl @NonConstConversion.cpp_operator [concrete = constants.%NonConstConversion.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b08 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b08 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NonConstConversion.480 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NonConstConversion.480 = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.b58ecd.2: %Dest__carbon_thunk.type.2ffc9e.2 = fn_decl @Dest__carbon_thunk.2 [concrete = constants.%Dest__carbon_thunk.f8fa06.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source2.decl: type = class_decl @Source2 [concrete = constants.%Source2] {} {}
// CHECK:STDOUT:   %Dest2.decl: type = class_decl @Dest2 [concrete = constants.%Dest2] {} {}
// CHECK:STDOUT:   %Dest2__carbon_thunk.decl: %Dest2__carbon_thunk.type = fn_decl @Dest2__carbon_thunk [concrete = constants.%Dest2__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest2.cpp_destructor.decl: %Dest2.cpp_destructor.type = fn_decl @Dest2.cpp_destructor [concrete = constants.%Dest2.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39d = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.39d = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest2 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UserConversion(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc26.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc26.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc26.2: init %Dest to %.loc26.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc26.3: init %Dest = converted %s.ref, %.loc26.2
// CHECK:STDOUT:   %.loc26.4: ref %Dest = temporary %.loc26.1, %.loc26.3
// CHECK:STDOUT:   %.loc26.5: %Dest = acquire_value %.loc26.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc26.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc26.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc26.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc26.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversion.loc39(%s.param: ref %NonConstConversion.480) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: ref %NonConstConversion.480 = name_ref s, %s
// CHECK:STDOUT:   %.loc44: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%NonConstConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc47.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc47.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc47.2: init %Dest to %.loc47.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc47.3: init %Dest = converted %s.ref, %.loc47.2
// CHECK:STDOUT:   %.loc47.4: ref %Dest = temporary %.loc47.1, %.loc47.3
// CHECK:STDOUT:   %.loc47.5: %Dest = acquire_value %.loc47.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc47.5
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc47.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc47.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc47.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.39d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %.loc64: type = splice_block %Dest2.ref [concrete = constants.%Dest2] {
// CHECK:STDOUT:     %Cpp.ref.loc63: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc67.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc67.2: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc67.1: %ptr.472 = addr_of %.loc67.2
// CHECK:STDOUT:   %addr.loc67.2: %ptr.9ae = addr_of %.loc67.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc67.1, %addr.loc67.2)
// CHECK:STDOUT:   %.loc67.3: init %Dest2 to %.loc67.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc67.4: init %Dest2 = converted %s.ref, %.loc67.3
// CHECK:STDOUT:   %.loc67.5: ref %Dest2 = temporary %.loc67.1, %.loc67.4
// CHECK:STDOUT:   %.loc67.6: %Dest2 = acquire_value %.loc67.5
// CHECK:STDOUT:   %_: %Dest2 = value_binding _, %.loc67.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest2.Op.bound: <bound method> = bound_method %.loc67.5, constants.%Dest2.Op
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
// CHECK:STDOUT:   %Dest2.cpp_destructor.bound: <bound method> = bound_method %.loc67.5, %Op.ref
// CHECK:STDOUT:   %Dest2.cpp_destructor.call: init %empty_tuple.type = call %Dest2.cpp_destructor.bound(%.loc67.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonConstConversion: type = class_type @NonConstConversion [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest.5e7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonConstConversion = %NonConstConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion [concrete = constants.%NonConstConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversionTest(%s.param: %NonConstConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %NonConstConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor: type = class_type @InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.b73: type = pattern_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %ptr.1fc: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.12d: type = ptr_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.type: type = fn_type @InaccessibleConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk: %InaccessibleConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.type: type = fn_type @InaccessibleConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor: %InaccessibleConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.Op.type: type = fn_type @InaccessibleConstructor.Op [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.Op: %InaccessibleConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConversion: type = class_type @InaccessibleConversion [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %InaccessibleConversion [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.type: type = fn_type @InaccessibleConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator: %InaccessibleConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .InaccessibleConstructor = %InaccessibleConstructor.decl
// CHECK:STDOUT:     .InaccessibleConversion = %InaccessibleConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %InaccessibleConstructor.decl: type = class_decl @InaccessibleConstructor [concrete = constants.%InaccessibleConstructor] {} {}
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.decl: %InaccessibleConstructor__carbon_thunk.type = fn_decl @InaccessibleConstructor__carbon_thunk [concrete = constants.%InaccessibleConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.decl: %InaccessibleConstructor.cpp_destructor.type = fn_decl @InaccessibleConstructor.cpp_destructor [concrete = constants.%InaccessibleConstructor.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b73 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b73 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %InaccessibleConstructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %InaccessibleConstructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.decl: type = class_decl @InaccessibleConversion [concrete = constants.%InaccessibleConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.decl: %InaccessibleConversion.cpp_operator.type = fn_decl @InaccessibleConversion.cpp_operator [concrete = constants.%InaccessibleConversion.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %InaccessibleConversion = value_param call_param0
// CHECK:STDOUT:     %self: %InaccessibleConversion = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b73 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %InaccessibleConstructor.ref [concrete = constants.%InaccessibleConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InaccessibleConstructor.ref: type = name_ref InaccessibleConstructor, imports.%InaccessibleConstructor.decl [concrete = constants.%InaccessibleConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26.1: ref %InaccessibleConstructor = temporary_storage
// CHECK:STDOUT:   %.loc26.2: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc26.1: %ptr.1fc = addr_of %.loc26.2
// CHECK:STDOUT:   %addr.loc26.2: %ptr.12d = addr_of %.loc26.1
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InaccessibleConstructor__carbon_thunk.decl(%addr.loc26.1, %addr.loc26.2)
// CHECK:STDOUT:   %.loc26.3: init %InaccessibleConstructor to %.loc26.1 = mark_in_place_init %InaccessibleConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc26.4: init %InaccessibleConstructor = converted %s.ref, %.loc26.3
// CHECK:STDOUT:   %.loc26.5: ref %InaccessibleConstructor = temporary %.loc26.1, %.loc26.4
// CHECK:STDOUT:   %.loc26.6: %InaccessibleConstructor = acquire_value %.loc26.5
// CHECK:STDOUT:   %_: %InaccessibleConstructor = value_binding _, %.loc26.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %InaccessibleConstructor.Op.bound: <bound method> = bound_method %.loc26.5, constants.%InaccessibleConstructor.Op
// CHECK:STDOUT:   %Op.ref: %InaccessibleConstructor.cpp_destructor.type = name_ref Op, imports.%InaccessibleConstructor.cpp_destructor.decl [concrete = constants.%InaccessibleConstructor.cpp_destructor]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc26.5, %Op.ref
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.call: init %empty_tuple.type = call %InaccessibleConstructor.cpp_destructor.bound(%.loc26.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConversionTest(%s.param: %InaccessibleConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %InaccessibleConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc43: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%InaccessibleConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc46.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc46.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc46.2: init %Dest to %.loc46.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc46.3: init %Dest = converted %s.ref, %.loc46.2
// CHECK:STDOUT:   %.loc46.4: ref %Dest = temporary %.loc46.1, %.loc46.3
// CHECK:STDOUT:   %.loc46.5: %Dest = acquire_value %.loc46.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc46.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc46.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc46.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc46.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deleted.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %DeletedConstructor: type = class_type @DeletedConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.158: type = pattern_type %DeletedConstructor [concrete]
// CHECK:STDOUT:   %DeletedConversion: type = class_type @DeletedConversion [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest.5e7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .DeletedConstructor = %DeletedConstructor.decl
// CHECK:STDOUT:     .DeletedConversion = %DeletedConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %DeletedConstructor.decl: type = class_decl @DeletedConstructor [concrete = constants.%DeletedConstructor] {} {}
// CHECK:STDOUT:   %DeletedConversion.decl: type = class_decl @DeletedConversion [concrete = constants.%DeletedConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.158 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %DeletedConstructor.ref [concrete = constants.%DeletedConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeletedConstructor.ref: type = name_ref DeletedConstructor, imports.%DeletedConstructor.decl [concrete = constants.%DeletedConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26: %DeletedConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %DeletedConstructor = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConversionTest(%s.param: %DeletedConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %DeletedConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc43: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc46: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.1eb: type = pattern_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest.5e7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .ExplicitConstructor = %ExplicitConstructor.decl
// CHECK:STDOUT:     .ExplicitConversion = %ExplicitConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ExplicitConstructor.decl: type = class_decl @ExplicitConstructor [concrete = constants.%ExplicitConstructor] {} {}
// CHECK:STDOUT:   %ExplicitConversion.decl: type = class_decl @ExplicitConversion [concrete = constants.%ExplicitConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.1eb = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %ExplicitConstructor.ref [concrete = constants.%ExplicitConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26: %ExplicitConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %ExplicitConstructor = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc43: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc46: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- i32_to_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor.f49: type = class_type @IntConstructor.1 [concrete]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %IntConstructor.f49 [concrete]
// CHECK:STDOUT:   %ptr.d59: type = ptr_type %IntConstructor.f49 [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.type: type = fn_type @IntConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk: %IntConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.type: type = fn_type @IntConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor: %IntConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntConstructor.Op.type: type = fn_type @IntConstructor.Op [concrete]
// CHECK:STDOUT:   %IntConstructor.Op: %IntConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .IntConstructor = %IntConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.decl: type = class_decl @IntConstructor.1 [concrete = constants.%IntConstructor.f49] {} {}
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.decl: %IntConstructor__carbon_thunk.type = fn_decl @IntConstructor__carbon_thunk [concrete = constants.%IntConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.decl: %IntConstructor.cpp_destructor.type = fn_decl @IntConstructor.cpp_destructor [concrete = constants.%IntConstructor.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3c3 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.3c3 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %IntConstructor.f49 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %IntConstructor.f49 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructor.loc16(%i.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.3c3 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc21: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor.f49] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor.f49]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24.1: ref %IntConstructor.f49 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d59 = addr_of %.loc24.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(%i.ref, %addr)
// CHECK:STDOUT:   %.loc24.2: init %IntConstructor.f49 to %.loc24.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc24.3: init %IntConstructor.f49 = converted %i.ref, %.loc24.2
// CHECK:STDOUT:   %.loc24.4: ref %IntConstructor.f49 = temporary %.loc24.1, %.loc24.3
// CHECK:STDOUT:   %.loc24.5: %IntConstructor.f49 = acquire_value %.loc24.4
// CHECK:STDOUT:   %_: %IntConstructor.f49 = value_binding _, %.loc24.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %IntConstructor.Op.bound: <bound method> = bound_method %.loc24.4, constants.%IntConstructor.Op
// CHECK:STDOUT:   %Op.ref: %IntConstructor.cpp_destructor.type = name_ref Op, imports.%IntConstructor.cpp_destructor.decl [concrete = constants.%IntConstructor.cpp_destructor]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc24.4, %Op.ref
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.call: init %empty_tuple.type = call %IntConstructor.cpp_destructor.bound(%.loc24.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_u32_to_int_conversion_in_carbon.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor: type = class_type @IntConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %IntConstructor [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d59: type = ptr_type %IntConstructor [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.type: type = fn_type @IntConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk: %IntConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.type: type = fn_type @IntConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor: %IntConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntConstructor.Op.type: type = fn_type @IntConstructor.Op [concrete]
// CHECK:STDOUT:   %IntConstructor.Op: %IntConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .IntConstructor = %IntConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.decl: type = class_decl @IntConstructor [concrete = constants.%IntConstructor] {} {}
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.decl: %IntConstructor__carbon_thunk.type = fn_decl @IntConstructor__carbon_thunk [concrete = constants.%IntConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.decl: %IntConstructor.cpp_destructor.type = fn_decl @IntConstructor.cpp_destructor [concrete = constants.%IntConstructor.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3c3 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.3c3 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %IntConstructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %IntConstructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructorTest(%u.param: %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.3c3 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.ref: %u32 = name_ref u, %u
// CHECK:STDOUT:   %.loc21: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24.1: ref %IntConstructor = temporary_storage
// CHECK:STDOUT:   %.loc24.2: %i32 = converted %u.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.d59 = addr_of %.loc24.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(<error>, %addr)
// CHECK:STDOUT:   %.loc24.3: init %IntConstructor to %.loc24.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc24.4: init %IntConstructor = converted %u.ref, %.loc24.3
// CHECK:STDOUT:   %.loc24.5: ref %IntConstructor = temporary %.loc24.1, %.loc24.4
// CHECK:STDOUT:   %.loc24.6: %IntConstructor = acquire_value %.loc24.5
// CHECK:STDOUT:   %_: %IntConstructor = value_binding _, %.loc24.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %IntConstructor.Op.bound: <bound method> = bound_method %.loc24.5, constants.%IntConstructor.Op
// CHECK:STDOUT:   %Op.ref: %IntConstructor.cpp_destructor.type = name_ref Op, imports.%IntConstructor.cpp_destructor.decl [concrete = constants.%IntConstructor.cpp_destructor]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc24.5, %Op.ref
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.call: init %empty_tuple.type = call %IntConstructor.cpp_destructor.bound(%.loc24.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultConstructor: type = class_type @DefaultConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.b66: type = pattern_type %DefaultConstructor [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.3f7: type = ptr_type %DefaultConstructor [concrete]
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.type: type = fn_type @DefaultConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk: %DefaultConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.type: type = fn_type @DefaultConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor: %DefaultConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultConstructor.Op.type: type = fn_type @DefaultConstructor.Op [concrete]
// CHECK:STDOUT:   %DefaultConstructor.Op: %DefaultConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefaultConstructor = %DefaultConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructor.decl: type = class_decl @DefaultConstructor [concrete = constants.%DefaultConstructor] {} {}
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.decl: %DefaultConstructor__carbon_thunk.type = fn_decl @DefaultConstructor__carbon_thunk [concrete = constants.%DefaultConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.decl: %DefaultConstructor.cpp_destructor.type = fn_decl @DefaultConstructor.cpp_destructor [concrete = constants.%DefaultConstructor.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b66 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b66 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultConstructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultConstructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DefaultConstructorTest() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b66 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18: type = splice_block %DefaultConstructor.ref [concrete = constants.%DefaultConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultConstructor.ref: type = name_ref DefaultConstructor, imports.%DefaultConstructor.decl [concrete = constants.%DefaultConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.2: ref %DefaultConstructor = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.3f7 = addr_of %.loc22.2
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%DefaultConstructor__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc22.3: init %DefaultConstructor to %.loc22.2 = mark_in_place_init %DefaultConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc22.4: init %DefaultConstructor = converted %.loc22.1, %.loc22.3
// CHECK:STDOUT:   %.loc22.5: ref %DefaultConstructor = temporary %.loc22.2, %.loc22.4
// CHECK:STDOUT:   %.loc22.6: %DefaultConstructor = acquire_value %.loc22.5
// CHECK:STDOUT:   %_: %DefaultConstructor = value_binding _, %.loc22.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %DefaultConstructor.Op.bound: <bound method> = bound_method %.loc22.5, constants.%DefaultConstructor.Op
// CHECK:STDOUT:   %Op.ref: %DefaultConstructor.cpp_destructor.type = name_ref Op, imports.%DefaultConstructor.cpp_destructor.decl [concrete = constants.%DefaultConstructor.cpp_destructor]
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc22.5, %Op.ref
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.call: init %empty_tuple.type = call %DefaultConstructor.cpp_destructor.bound(%.loc22.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- construct_multi_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Two: type = class_type @Two [concrete]
// CHECK:STDOUT:   %pattern_type.4c2: type = pattern_type %Two [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.677: type = ptr_type %Two [concrete]
// CHECK:STDOUT:   %Two__carbon_thunk.type: type = fn_type @Two__carbon_thunk [concrete]
// CHECK:STDOUT:   %Two__carbon_thunk: %Two__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault: type = class_type @ThreeWithDefault [concrete]
// CHECK:STDOUT:   %pattern_type.ca7: type = pattern_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ptr.ea4: type = ptr_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.b64454.1: type = fn_type @ThreeWithDefault__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.fd88b7.1: %ThreeWithDefault__carbon_thunk.type.b64454.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.b64454.2: type = fn_type @ThreeWithDefault__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.fd88b7.2: %ThreeWithDefault__carbon_thunk.type.b64454.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.type: type = fn_type @ThreeWithDefault.cpp_destructor [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor: %ThreeWithDefault.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.Op.type: type = fn_type @ThreeWithDefault.Op [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.Op: %ThreeWithDefault.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Two.cpp_destructor.type: type = fn_type @Two.cpp_destructor [concrete]
// CHECK:STDOUT:   %Two.cpp_destructor: %Two.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Two.Op.type: type = fn_type @Two.Op [concrete]
// CHECK:STDOUT:   %Two.Op: %Two.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Two = %Two.decl
// CHECK:STDOUT:     .ThreeWithDefault = %ThreeWithDefault.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Two.decl: type = class_decl @Two [concrete = constants.%Two] {} {}
// CHECK:STDOUT:   %Two__carbon_thunk.decl: %Two__carbon_thunk.type = fn_decl @Two__carbon_thunk [concrete = constants.%Two__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.decl: type = class_decl @ThreeWithDefault [concrete = constants.%ThreeWithDefault] {} {}
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.decl.482856.1: %ThreeWithDefault__carbon_thunk.type.b64454.1 = fn_decl @ThreeWithDefault__carbon_thunk.1 [concrete = constants.%ThreeWithDefault__carbon_thunk.fd88b7.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.decl.482856.2: %ThreeWithDefault__carbon_thunk.type.b64454.2 = fn_decl @ThreeWithDefault__carbon_thunk.2 [concrete = constants.%ThreeWithDefault__carbon_thunk.fd88b7.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.decl: %ThreeWithDefault.cpp_destructor.type = fn_decl @ThreeWithDefault.cpp_destructor [concrete = constants.%ThreeWithDefault.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ca7 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.ca7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ThreeWithDefault = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ThreeWithDefault = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Two.cpp_destructor.decl: %Two.cpp_destructor.type = fn_decl @Two.cpp_destructor [concrete = constants.%Two.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4c2 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.4c2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Two = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Two = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplicitConvert() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.4c2 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc24: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc25.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2.loc24) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc18: type = splice_block %Two.ref [concrete = constants.%Two] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Two.ref: type = name_ref Two, imports.%Two.decl [concrete = constants.%Two]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25.2: ref %Two = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc22: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc22.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22.2(%int_1.loc22) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22.2: %i32 = converted %int_1.loc22, %.loc22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc24: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24.1: <bound method> = bound_method %int_2.loc24, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24.2: <bound method> = bound_method %int_2.loc24, %specific_fn.loc24 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24.2(%int_2.loc24) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc24.2: %i32 = converted %int_2.loc24, %.loc24.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %addr.loc25: %ptr.677 = addr_of %.loc25.2
// CHECK:STDOUT:   %Two__carbon_thunk.call: init %empty_tuple.type = call imports.%Two__carbon_thunk.decl(%.loc22.2, %.loc24.2, %addr.loc25)
// CHECK:STDOUT:   %.loc25.3: init %Two to %.loc25.2 = mark_in_place_init %Two__carbon_thunk.call
// CHECK:STDOUT:   %.loc25.4: init %Two = converted %.loc25.1, %.loc25.3
// CHECK:STDOUT:   %.loc25.5: ref %Two = temporary %.loc25.2, %.loc25.4
// CHECK:STDOUT:   %.loc25.6: %Two = acquire_value %.loc25.5
// CHECK:STDOUT:   %_.loc15: %Two = value_binding _, %.loc25.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc28: %pattern_type.ca7 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc35: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc37: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc38.1: %tuple.type.f94 = tuple_literal (%int_1.loc35, %int_2.loc37) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc31: type = splice_block %ThreeWithDefault.ref.loc31 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc31: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc38.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc35: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc35.1: <bound method> = bound_method %int_1.loc35, %impl.elem0.loc35 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35.2: <bound method> = bound_method %int_1.loc35, %specific_fn.loc35 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35: init %i32 = call %bound_method.loc35.2(%int_1.loc35) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc35.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc35.2: %i32 = converted %int_1.loc35, %.loc35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc37: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc37.1: <bound method> = bound_method %int_2.loc37, %impl.elem0.loc37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37.2: <bound method> = bound_method %int_2.loc37, %specific_fn.loc37 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc37: init %i32 = call %bound_method.loc37.2(%int_2.loc37) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc37 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc37.2: %i32 = converted %int_2.loc37, %.loc37.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %addr.loc38: %ptr.ea4 = addr_of %.loc38.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc38: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.482856.1(%.loc35.2, %.loc37.2, %addr.loc38)
// CHECK:STDOUT:   %.loc38.3: init %ThreeWithDefault to %.loc38.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc38
// CHECK:STDOUT:   %.loc38.4: init %ThreeWithDefault = converted %.loc38.1, %.loc38.3
// CHECK:STDOUT:   %.loc38.5: ref %ThreeWithDefault = temporary %.loc38.2, %.loc38.4
// CHECK:STDOUT:   %.loc38.6: %ThreeWithDefault = acquire_value %.loc38.5
// CHECK:STDOUT:   %_.loc28: %ThreeWithDefault = value_binding _, %.loc38.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc41: %pattern_type.ca7 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc48: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc50: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc53.1: %tuple.type.37f = tuple_literal (%int_1.loc48, %int_2.loc50, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %.loc44: type = splice_block %ThreeWithDefault.ref.loc44 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc44: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc53.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc48: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc48.1: <bound method> = bound_method %int_1.loc48, %impl.elem0.loc48 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc48: <specific function> = specific_function %impl.elem0.loc48, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc48.2: <bound method> = bound_method %int_1.loc48, %specific_fn.loc48 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc48: init %i32 = call %bound_method.loc48.2(%int_1.loc48) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc48.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc48 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc48.2: %i32 = converted %int_1.loc48, %.loc48.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc50: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc50.1: <bound method> = bound_method %int_2.loc50, %impl.elem0.loc50 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc50.2: <bound method> = bound_method %int_2.loc50, %specific_fn.loc50 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc50: init %i32 = call %bound_method.loc50.2(%int_2.loc50) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc50.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc50 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc50.2: %i32 = converted %int_2.loc50, %.loc50.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc52: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc52.1: <bound method> = bound_method %int_3, %impl.elem0.loc52 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc52: <specific function> = specific_function %impl.elem0.loc52, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52.2: <bound method> = bound_method %int_3, %specific_fn.loc52 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52: init %i32 = call %bound_method.loc52.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc52.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc52.2: %i32 = converted %int_3, %.loc52.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %addr.loc53: %ptr.ea4 = addr_of %.loc53.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc53: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.482856.2(%.loc48.2, %.loc50.2, %.loc52.2, %addr.loc53)
// CHECK:STDOUT:   %.loc53.3: init %ThreeWithDefault to %.loc53.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc53
// CHECK:STDOUT:   %.loc53.4: init %ThreeWithDefault = converted %.loc53.1, %.loc53.3
// CHECK:STDOUT:   %.loc53.5: ref %ThreeWithDefault = temporary %.loc53.2, %.loc53.4
// CHECK:STDOUT:   %.loc53.6: %ThreeWithDefault = acquire_value %.loc53.5
// CHECK:STDOUT:   %_.loc41: %ThreeWithDefault = value_binding _, %.loc53.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ThreeWithDefault.Op.bound.loc53: <bound method> = bound_method %.loc53.5, constants.%ThreeWithDefault.Op
// CHECK:STDOUT:   %Op.ref.loc53: %ThreeWithDefault.cpp_destructor.type = name_ref Op, imports.%ThreeWithDefault.cpp_destructor.decl [concrete = constants.%ThreeWithDefault.cpp_destructor]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.bound.loc53: <bound method> = bound_method %.loc53.5, %Op.ref.loc53
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.call.loc53: init %empty_tuple.type = call %ThreeWithDefault.cpp_destructor.bound.loc53(%.loc53.5)
// CHECK:STDOUT:   %ThreeWithDefault.Op.bound.loc38: <bound method> = bound_method %.loc38.5, constants.%ThreeWithDefault.Op
// CHECK:STDOUT:   %Op.ref.loc38: %ThreeWithDefault.cpp_destructor.type = name_ref Op, imports.%ThreeWithDefault.cpp_destructor.decl [concrete = constants.%ThreeWithDefault.cpp_destructor]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.bound.loc38: <bound method> = bound_method %.loc38.5, %Op.ref.loc38
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.call.loc38: init %empty_tuple.type = call %ThreeWithDefault.cpp_destructor.bound.loc38(%.loc38.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Two.Op.bound: <bound method> = bound_method %.loc25.5, constants.%Two.Op
// CHECK:STDOUT:   %Op.ref.loc25: %Two.cpp_destructor.type = name_ref Op, imports.%Two.cpp_destructor.decl [concrete = constants.%Two.cpp_destructor]
// CHECK:STDOUT:   %Two.cpp_destructor.bound: <bound method> = bound_method %.loc25.5, %Op.ref.loc25
// CHECK:STDOUT:   %Two.cpp_destructor.call: init %empty_tuple.type = call %Two.cpp_destructor.bound(%.loc25.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_aggregate_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Aggregate: type = class_type @Aggregate [concrete]
// CHECK:STDOUT:   %pattern_type.235: type = pattern_type %Aggregate [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Aggregate = %Aggregate.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Aggregate.decl: type = class_decl @Aggregate [concrete = constants.%Aggregate] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromTuple() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.235 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18: type = splice_block %Aggregate.ref.loc18 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc18: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.2: %Aggregate = converted %.loc22.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc25: %pattern_type.235 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc34.1: %tuple.type.985 = tuple_literal (%int_1.loc32) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc28: type = splice_block %Aggregate.ref.loc28 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc28: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc34.2: %Aggregate = converted %.loc34.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc25: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc37: %pattern_type.235 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc47.1: %tuple.type.f94 = tuple_literal (%int_1.loc44, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc40: type = splice_block %Aggregate.ref.loc40 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc40: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc47.2: %Aggregate = converted %.loc47.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc37: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_aggregate_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Aggregate: type = class_type @Aggregate [concrete]
// CHECK:STDOUT:   %pattern_type.235: type = pattern_type %Aggregate [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.147: %struct_type.x = struct_value (%int_1) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Aggregate = %Aggregate.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Aggregate.decl: type = class_decl @Aggregate [concrete = constants.%Aggregate] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.235 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18: type = splice_block %Aggregate.ref.loc18 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc18: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.2: %Aggregate = converted %.loc22.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc25: %pattern_type.235 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc35: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc36.1: %struct_type.x = struct_literal (%int_1.loc35) [concrete = constants.%struct.147]
// CHECK:STDOUT:   %.loc28: type = splice_block %Aggregate.ref.loc28 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc28: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36.2: %Aggregate = converted %.loc36.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc25: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc39: %pattern_type.235 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc55.1: %struct_type.x.y = struct_literal (%int_1.loc49, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc42: type = splice_block %Aggregate.ref.loc42 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc42: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc55.2: %Aggregate = converted %.loc55.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc39: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_aggregate_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonAggregate: type = class_type @NonAggregate [concrete]
// CHECK:STDOUT:   %pattern_type.eac: type = pattern_type %NonAggregate [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.9b6: type = ptr_type %NonAggregate [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.8d64dd.1: type = fn_type @NonAggregate__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.0020a5.1: %NonAggregate__carbon_thunk.type.8d64dd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.8d64dd.2: type = fn_type @NonAggregate__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.0020a5.2: %NonAggregate__carbon_thunk.type.8d64dd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.8d64dd.3: type = fn_type @NonAggregate__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.0020a5.3: %NonAggregate__carbon_thunk.type.8d64dd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.type: type = fn_type @NonAggregate.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor: %NonAggregate.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonAggregate.Op.type: type = fn_type @NonAggregate.Op [concrete]
// CHECK:STDOUT:   %NonAggregate.Op: %NonAggregate.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonAggregate = %NonAggregate.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate.decl: type = class_decl @NonAggregate [concrete = constants.%NonAggregate] {} {}
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.712b28.1: %NonAggregate__carbon_thunk.type.8d64dd.1 = fn_decl @NonAggregate__carbon_thunk.1 [concrete = constants.%NonAggregate__carbon_thunk.0020a5.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.712b28.2: %NonAggregate__carbon_thunk.type.8d64dd.2 = fn_decl @NonAggregate__carbon_thunk.2 [concrete = constants.%NonAggregate__carbon_thunk.0020a5.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.712b28.3: %NonAggregate__carbon_thunk.type.8d64dd.3 = fn_decl @NonAggregate__carbon_thunk.3 [concrete = constants.%NonAggregate__carbon_thunk.0020a5.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.decl: %NonAggregate.cpp_destructor.type = fn_decl @NonAggregate.cpp_destructor [concrete = constants.%NonAggregate.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.eac = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.eac = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NonAggregate = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NonAggregate = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18: type = splice_block %NonAggregate.ref.loc18 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc18: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.9b6 = addr_of %.loc22.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.712b28.1(%addr.loc22)
// CHECK:STDOUT:   %.loc22.3: init %NonAggregate to %.loc22.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22.4: init %NonAggregate = converted %.loc22.1, %.loc22.3
// CHECK:STDOUT:   %.loc22.5: ref %NonAggregate = temporary %.loc22.2, %.loc22.4
// CHECK:STDOUT:   %.loc22.6: %NonAggregate = acquire_value %.loc22.5
// CHECK:STDOUT:   %_.loc15: %NonAggregate = value_binding _, %.loc22.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc25: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc34.1: %tuple.type.985 = tuple_literal (%int_1.loc32) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc28: type = splice_block %NonAggregate.ref.loc28 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc28: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc34.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc32: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc32.1: <bound method> = bound_method %int_1.loc32, %impl.elem0.loc32 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc32: <specific function> = specific_function %impl.elem0.loc32, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32.2: <bound method> = bound_method %int_1.loc32, %specific_fn.loc32 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32: init %i32 = call %bound_method.loc32.2(%int_1.loc32) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32.2: %i32 = converted %int_1.loc32, %.loc32.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %addr.loc34: %ptr.9b6 = addr_of %.loc34.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc34: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.712b28.2(%.loc32.2, %addr.loc34)
// CHECK:STDOUT:   %.loc34.3: init %NonAggregate to %.loc34.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc34
// CHECK:STDOUT:   %.loc34.4: init %NonAggregate = converted %.loc34.1, %.loc34.3
// CHECK:STDOUT:   %.loc34.5: ref %NonAggregate = temporary %.loc34.2, %.loc34.4
// CHECK:STDOUT:   %.loc34.6: %NonAggregate = acquire_value %.loc34.5
// CHECK:STDOUT:   %_.loc25: %NonAggregate = value_binding _, %.loc34.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc37: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc47.1: %tuple.type.f94 = tuple_literal (%int_1.loc44, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc40: type = splice_block %NonAggregate.ref.loc40 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc40: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc47.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc44: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44.1: <bound method> = bound_method %int_1.loc44, %impl.elem0.loc44 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc44: <specific function> = specific_function %impl.elem0.loc44, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44.2: <bound method> = bound_method %int_1.loc44, %specific_fn.loc44 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44: init %i32 = call %bound_method.loc44.2(%int_1.loc44) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc44.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc44.2: %i32 = converted %int_1.loc44, %.loc44.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc46: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc46.1: <bound method> = bound_method %int_2, %impl.elem0.loc46 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc46: <specific function> = specific_function %impl.elem0.loc46, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46.2: <bound method> = bound_method %int_2, %specific_fn.loc46 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46: init %i32 = call %bound_method.loc46.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc46.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc46.2: %i32 = converted %int_2, %.loc46.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %addr.loc47: %ptr.9b6 = addr_of %.loc47.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc47: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.712b28.3(%.loc44.2, %.loc46.2, %addr.loc47)
// CHECK:STDOUT:   %.loc47.3: init %NonAggregate to %.loc47.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc47
// CHECK:STDOUT:   %.loc47.4: init %NonAggregate = converted %.loc47.1, %.loc47.3
// CHECK:STDOUT:   %.loc47.5: ref %NonAggregate = temporary %.loc47.2, %.loc47.4
// CHECK:STDOUT:   %.loc47.6: %NonAggregate = acquire_value %.loc47.5
// CHECK:STDOUT:   %_.loc37: %NonAggregate = value_binding _, %.loc47.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NonAggregate.Op.bound.loc47: <bound method> = bound_method %.loc47.5, constants.%NonAggregate.Op
// CHECK:STDOUT:   %Op.ref.loc47: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.bound.loc47: <bound method> = bound_method %.loc47.5, %Op.ref.loc47
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.call.loc47: init %empty_tuple.type = call %NonAggregate.cpp_destructor.bound.loc47(%.loc47.5)
// CHECK:STDOUT:   %NonAggregate.Op.bound.loc34: <bound method> = bound_method %.loc34.5, constants.%NonAggregate.Op
// CHECK:STDOUT:   %Op.ref.loc34: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.bound.loc34: <bound method> = bound_method %.loc34.5, %Op.ref.loc34
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.call.loc34: init %empty_tuple.type = call %NonAggregate.cpp_destructor.bound.loc34(%.loc34.5)
// CHECK:STDOUT:   %NonAggregate.Op.bound.loc22: <bound method> = bound_method %.loc22.5, constants.%NonAggregate.Op
// CHECK:STDOUT:   %Op.ref.loc22: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22.5, %Op.ref.loc22
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.call.loc22: init %empty_tuple.type = call %NonAggregate.cpp_destructor.bound.loc22(%.loc22.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_aggregate_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonAggregate: type = class_type @NonAggregate [concrete]
// CHECK:STDOUT:   %pattern_type.eac: type = pattern_type %NonAggregate [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.147: %struct_type.x = struct_value (%int_1) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonAggregate = %NonAggregate.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate.decl: type = class_decl @NonAggregate [concrete = constants.%NonAggregate] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18: type = splice_block %NonAggregate.ref.loc18 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc18: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.2: %NonAggregate = converted %.loc22.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc25: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc35: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc36.1: %struct_type.x = struct_literal (%int_1.loc35) [concrete = constants.%struct.147]
// CHECK:STDOUT:   %.loc28: type = splice_block %NonAggregate.ref.loc28 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc28: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36.2: %NonAggregate = converted %.loc36.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc25: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc39: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc55.1: %struct_type.x.y = struct_literal (%int_1.loc49, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc42: type = splice_block %NonAggregate.ref.loc42 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc42: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc55.2: %NonAggregate = converted %.loc55.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc39: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
