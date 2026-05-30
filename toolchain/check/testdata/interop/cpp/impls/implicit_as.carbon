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
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.56d: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %Source.cpp_operator.type: type = fn_type @Source.cpp_operator [concrete]
// CHECK:STDOUT:   %Source.cpp_operator: %Source.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.246: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc11.1: type = fn_type @Dest__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.61ac37.1: %Dest__carbon_thunk.type.2ffc11.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonConstConversion.2f9: type = class_type @NonConstConversion.1 [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.type: type = fn_type @NonConstConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator: %NonConstConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc11.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.61ac37.2: %Dest__carbon_thunk.type.2ffc11.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.ca0: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %ptr.692: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.a5d: type = ptr_type %Dest2 [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.74d7b8.1: %Dest__carbon_thunk.type.2ffc11.1 = fn_decl @Dest__carbon_thunk.1 [concrete = constants.%Dest__carbon_thunk.61ac37.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion.1 [concrete = constants.%NonConstConversion.2f9] {} {}
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.decl: %NonConstConversion.cpp_operator.type = fn_decl @NonConstConversion.cpp_operator [concrete = constants.%NonConstConversion.cpp_operator] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.74d7b8.2: %Dest__carbon_thunk.type.2ffc11.2 = fn_decl @Dest__carbon_thunk.2 [concrete = constants.%Dest__carbon_thunk.61ac37.2] {
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UserConversion(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.56d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc8_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc8_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.246 = addr_of %.loc8_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.74d7b8.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc8_21.2: init %Dest to %.loc8_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_21.3: init %Dest = converted %s.ref, %.loc8_21.2
// CHECK:STDOUT:   %.loc8_21.4: ref %Dest = temporary %.loc8_21.1, %.loc8_21.3
// CHECK:STDOUT:   %.loc8_21.5: %Dest = acquire_value %.loc8_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc8_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc8_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc8_21.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc8_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversion.loc12(%s.param: ref %NonConstConversion.2f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.56d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: ref %NonConstConversion.2f9 = name_ref s, %s
// CHECK:STDOUT:   %.loc14_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%NonConstConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc14_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.246 = addr_of %.loc14_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.74d7b8.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc14_21.2: init %Dest to %.loc14_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_21.3: init %Dest = converted %s.ref, %.loc14_21.2
// CHECK:STDOUT:   %.loc14_21.4: ref %Dest = temporary %.loc14_21.1, %.loc14_21.3
// CHECK:STDOUT:   %.loc14_21.5: %Dest = acquire_value %.loc14_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc14_21.5
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc14_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc14_21.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc14_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.ca0 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_13: type = splice_block %Dest2.ref [concrete = constants.%Dest2] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_22.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc20_22.2: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc20_22.1: %ptr.692 = addr_of %.loc20_22.2
// CHECK:STDOUT:   %addr.loc20_22.2: %ptr.a5d = addr_of %.loc20_22.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc20_22.1, %addr.loc20_22.2)
// CHECK:STDOUT:   %.loc20_22.3: init %Dest2 to %.loc20_22.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_22.4: init %Dest2 = converted %s.ref, %.loc20_22.3
// CHECK:STDOUT:   %.loc20_22.5: ref %Dest2 = temporary %.loc20_22.1, %.loc20_22.4
// CHECK:STDOUT:   %.loc20_22.6: %Dest2 = acquire_value %.loc20_22.5
// CHECK:STDOUT:   %_: %Dest2 = value_binding _, %.loc20_22.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest2.Op.bound: <bound method> = bound_method %.loc20_22.5, constants.%Dest2.Op
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
// CHECK:STDOUT:   %Dest2.cpp_destructor.bound: <bound method> = bound_method %.loc20_22.5, %Op.ref
// CHECK:STDOUT:   %Dest2.cpp_destructor.call: init %empty_tuple.type = call %Dest2.cpp_destructor.bound(%.loc20_22.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonConstConversion: type = class_type @NonConstConversion [concrete]
// CHECK:STDOUT:   %Dest.b15: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.56d: type = pattern_type %Dest.b15 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonConstConversion = %NonConstConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion [concrete = constants.%NonConstConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.b15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversionTest(%s.param: %NonConstConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.56d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %NonConstConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Dest.ref [concrete = constants.%Dest.b15] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.b15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_21: %Dest.b15 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.b15 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor: type = class_type @InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.f5d: type = pattern_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %ptr.71b: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.16a: type = ptr_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.type: type = fn_type @InaccessibleConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk: %InaccessibleConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.type: type = fn_type @InaccessibleConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor: %InaccessibleConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.Op.type: type = fn_type @InaccessibleConstructor.Op [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.Op: %InaccessibleConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConversion: type = class_type @InaccessibleConversion [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.56d: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.type: type = fn_type @InaccessibleConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator: %InaccessibleConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.246: type = ptr_type %Dest [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.decl: type = class_decl @InaccessibleConversion [concrete = constants.%InaccessibleConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.decl: %InaccessibleConversion.cpp_operator.type = fn_decl @InaccessibleConversion.cpp_operator [concrete = constants.%InaccessibleConversion.cpp_operator] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.f5d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc16_13: type = splice_block %InaccessibleConstructor.ref [concrete = constants.%InaccessibleConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InaccessibleConstructor.ref: type = name_ref InaccessibleConstructor, imports.%InaccessibleConstructor.decl [concrete = constants.%InaccessibleConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_40.1: ref %InaccessibleConstructor = temporary_storage
// CHECK:STDOUT:   %.loc16_40.2: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc16_40.1: %ptr.71b = addr_of %.loc16_40.2
// CHECK:STDOUT:   %addr.loc16_40.2: %ptr.16a = addr_of %.loc16_40.1
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InaccessibleConstructor__carbon_thunk.decl(%addr.loc16_40.1, %addr.loc16_40.2)
// CHECK:STDOUT:   %.loc16_40.3: init %InaccessibleConstructor to %.loc16_40.1 = mark_in_place_init %InaccessibleConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc16_40.4: init %InaccessibleConstructor = converted %s.ref, %.loc16_40.3
// CHECK:STDOUT:   %.loc16_40.5: ref %InaccessibleConstructor = temporary %.loc16_40.1, %.loc16_40.4
// CHECK:STDOUT:   %.loc16_40.6: %InaccessibleConstructor = acquire_value %.loc16_40.5
// CHECK:STDOUT:   %_: %InaccessibleConstructor = value_binding _, %.loc16_40.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %InaccessibleConstructor.Op.bound: <bound method> = bound_method %.loc16_40.5, constants.%InaccessibleConstructor.Op
// CHECK:STDOUT:   %Op.ref: %InaccessibleConstructor.cpp_destructor.type = name_ref Op, imports.%InaccessibleConstructor.cpp_destructor.decl [concrete = constants.%InaccessibleConstructor.cpp_destructor]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc16_40.5, %Op.ref
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.call: init %empty_tuple.type = call %InaccessibleConstructor.cpp_destructor.bound(%.loc16_40.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConversionTest(%s.param: %InaccessibleConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.56d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %InaccessibleConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc30_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%InaccessibleConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc30_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.246 = addr_of %.loc30_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc30_21.2: init %Dest to %.loc30_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc30_21.3: init %Dest = converted %s.ref, %.loc30_21.2
// CHECK:STDOUT:   %.loc30_21.4: ref %Dest = temporary %.loc30_21.1, %.loc30_21.3
// CHECK:STDOUT:   %.loc30_21.5: %Dest = acquire_value %.loc30_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc30_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc30_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc30_21.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc30_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deleted.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %DeletedConstructor: type = class_type @DeletedConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.216: type = pattern_type %DeletedConstructor [concrete]
// CHECK:STDOUT:   %DeletedConversion: type = class_type @DeletedConversion [concrete]
// CHECK:STDOUT:   %Dest.b15: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.56d: type = pattern_type %Dest.b15 [concrete]
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
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.b15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.216 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc15_13: type = splice_block %DeletedConstructor.ref [concrete = constants.%DeletedConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeletedConstructor.ref: type = name_ref DeletedConstructor, imports.%DeletedConstructor.decl [concrete = constants.%DeletedConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_35: %DeletedConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %DeletedConstructor = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConversionTest(%s.param: %DeletedConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.56d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %DeletedConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc28_13: type = splice_block %Dest.ref [concrete = constants.%Dest.b15] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.b15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21: %Dest.b15 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.b15 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.8d6: type = pattern_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %Dest.b15: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.56d: type = pattern_type %Dest.b15 [concrete]
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
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.b15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.8d6 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc15_13: type = splice_block %ExplicitConstructor.ref [concrete = constants.%ExplicitConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_36: %ExplicitConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %ExplicitConstructor = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.56d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc28_13: type = splice_block %Dest.ref [concrete = constants.%Dest.b15] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.b15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21: %Dest.b15 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.b15 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- i32_to_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor.14f: type = class_type @IntConstructor.1 [concrete]
// CHECK:STDOUT:   %pattern_type.599: type = pattern_type %IntConstructor.14f [concrete]
// CHECK:STDOUT:   %ptr.380: type = ptr_type %IntConstructor.14f [concrete]
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
// CHECK:STDOUT:   %IntConstructor.decl: type = class_decl @IntConstructor.1 [concrete = constants.%IntConstructor.14f] {} {}
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.decl: %IntConstructor__carbon_thunk.type = fn_decl @IntConstructor__carbon_thunk [concrete = constants.%IntConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.decl: %IntConstructor.cpp_destructor.type = fn_decl @IntConstructor.cpp_destructor [concrete = constants.%IntConstructor.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructor.loc6(%i.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.599 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc8_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor.14f] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor.14f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.1: ref %IntConstructor.14f = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.380 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(%i.ref, %addr)
// CHECK:STDOUT:   %.loc8_31.2: init %IntConstructor.14f to %.loc8_31.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_31.3: init %IntConstructor.14f = converted %i.ref, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: ref %IntConstructor.14f = temporary %.loc8_31.1, %.loc8_31.3
// CHECK:STDOUT:   %.loc8_31.5: %IntConstructor.14f = acquire_value %.loc8_31.4
// CHECK:STDOUT:   %_: %IntConstructor.14f = value_binding _, %.loc8_31.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %IntConstructor.Op.bound: <bound method> = bound_method %.loc8_31.4, constants.%IntConstructor.Op
// CHECK:STDOUT:   %Op.ref: %IntConstructor.cpp_destructor.type = name_ref Op, imports.%IntConstructor.cpp_destructor.decl [concrete = constants.%IntConstructor.cpp_destructor]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc8_31.4, %Op.ref
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.call: init %empty_tuple.type = call %IntConstructor.cpp_destructor.bound(%.loc8_31.4)
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
// CHECK:STDOUT:   %pattern_type.599: type = pattern_type %IntConstructor [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.380: type = ptr_type %IntConstructor [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructorTest(%u.param: %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.599 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.ref: %u32 = name_ref u, %u
// CHECK:STDOUT:   %.loc19_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_31.1: ref %IntConstructor = temporary_storage
// CHECK:STDOUT:   %.loc19_31.2: %i32 = converted %u.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.380 = addr_of %.loc19_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(<error>, %addr)
// CHECK:STDOUT:   %.loc19_31.3: init %IntConstructor to %.loc19_31.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc19_31.4: init %IntConstructor = converted %u.ref, %.loc19_31.3
// CHECK:STDOUT:   %.loc19_31.5: ref %IntConstructor = temporary %.loc19_31.1, %.loc19_31.4
// CHECK:STDOUT:   %.loc19_31.6: %IntConstructor = acquire_value %.loc19_31.5
// CHECK:STDOUT:   %_: %IntConstructor = value_binding _, %.loc19_31.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %IntConstructor.Op.bound: <bound method> = bound_method %.loc19_31.5, constants.%IntConstructor.Op
// CHECK:STDOUT:   %Op.ref: %IntConstructor.cpp_destructor.type = name_ref Op, imports.%IntConstructor.cpp_destructor.decl [concrete = constants.%IntConstructor.cpp_destructor]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc19_31.5, %Op.ref
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.call: init %empty_tuple.type = call %IntConstructor.cpp_destructor.bound(%.loc19_31.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultConstructor: type = class_type @DefaultConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.7fa: type = pattern_type %DefaultConstructor [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.b3c: type = ptr_type %DefaultConstructor [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DefaultConstructorTest() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.7fa = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_36.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %DefaultConstructor.ref [concrete = constants.%DefaultConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultConstructor.ref: type = name_ref DefaultConstructor, imports.%DefaultConstructor.decl [concrete = constants.%DefaultConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_36.2: ref %DefaultConstructor = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.b3c = addr_of %.loc9_36.2
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%DefaultConstructor__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc9_36.3: init %DefaultConstructor to %.loc9_36.2 = mark_in_place_init %DefaultConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc9_36.4: init %DefaultConstructor = converted %.loc9_36.1, %.loc9_36.3
// CHECK:STDOUT:   %.loc9_36.5: ref %DefaultConstructor = temporary %.loc9_36.2, %.loc9_36.4
// CHECK:STDOUT:   %.loc9_36.6: %DefaultConstructor = acquire_value %.loc9_36.5
// CHECK:STDOUT:   %_: %DefaultConstructor = value_binding _, %.loc9_36.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %DefaultConstructor.Op.bound: <bound method> = bound_method %.loc9_36.5, constants.%DefaultConstructor.Op
// CHECK:STDOUT:   %Op.ref: %DefaultConstructor.cpp_destructor.type = name_ref Op, imports.%DefaultConstructor.cpp_destructor.decl [concrete = constants.%DefaultConstructor.cpp_destructor]
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc9_36.5, %Op.ref
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.call: init %empty_tuple.type = call %DefaultConstructor.cpp_destructor.bound(%.loc9_36.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- construct_multi_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Two: type = class_type @Two [concrete]
// CHECK:STDOUT:   %pattern_type.034: type = pattern_type %Two [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.5a9: type = ptr_type %Two [concrete]
// CHECK:STDOUT:   %Two__carbon_thunk.type: type = fn_type @Two__carbon_thunk [concrete]
// CHECK:STDOUT:   %Two__carbon_thunk: %Two__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault: type = class_type @ThreeWithDefault [concrete]
// CHECK:STDOUT:   %pattern_type.830: type = pattern_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ptr.da2: type = ptr_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.b324b0.1: type = fn_type @ThreeWithDefault__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.8592b9.1: %ThreeWithDefault__carbon_thunk.type.b324b0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.b324b0.2: type = fn_type @ThreeWithDefault__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.8592b9.2: %ThreeWithDefault__carbon_thunk.type.b324b0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.353: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.decl: type = class_decl @ThreeWithDefault [concrete = constants.%ThreeWithDefault] {} {}
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.decl.be9bad.1: %ThreeWithDefault__carbon_thunk.type.b324b0.1 = fn_decl @ThreeWithDefault__carbon_thunk.1 [concrete = constants.%ThreeWithDefault__carbon_thunk.8592b9.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.decl.be9bad.2: %ThreeWithDefault__carbon_thunk.type.b324b0.2 = fn_decl @ThreeWithDefault__carbon_thunk.2 [concrete = constants.%ThreeWithDefault__carbon_thunk.8592b9.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.decl: %ThreeWithDefault.cpp_destructor.type = fn_decl @ThreeWithDefault.cpp_destructor [concrete = constants.%ThreeWithDefault.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Two.cpp_destructor.decl: %Two.cpp_destructor.type = fn_decl @Two.cpp_destructor [concrete = constants.%Two.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplicitConvert() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc8: %pattern_type.034 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.f94 = tuple_literal (%int_1.loc8, %int_2.loc8) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %Two.ref [concrete = constants.%Two] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Two.ref: type = name_ref Two, imports.%Two.decl [concrete = constants.%Two]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_25.2: ref %Two = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_21: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_21.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc8_21: <specific function> = specific_function %impl.elem0.loc8_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_21.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_21 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_21: init %i32 = call %bound_method.loc8_21.2(%int_1.loc8) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc8_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_21 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc8_21.2: %i32 = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc8_24: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_24.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc8_24: <specific function> = specific_function %impl.elem0.loc8_24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_24.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_24 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_24: init %i32 = call %bound_method.loc8_24.2(%int_2.loc8) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc8_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_24 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc8_24.2: %i32 = converted %int_2.loc8, %.loc8_24.1 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %addr.loc8: %ptr.5a9 = addr_of %.loc8_25.2
// CHECK:STDOUT:   %Two__carbon_thunk.call: init %empty_tuple.type = call imports.%Two__carbon_thunk.decl(%.loc8_21.2, %.loc8_24.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_25.3: init %Two to %.loc8_25.2 = mark_in_place_init %Two__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_25.4: init %Two = converted %.loc8_25.1, %.loc8_25.3
// CHECK:STDOUT:   %.loc8_25.5: ref %Two = temporary %.loc8_25.2, %.loc8_25.4
// CHECK:STDOUT:   %.loc8_25.6: %Two = acquire_value %.loc8_25.5
// CHECK:STDOUT:   %_.loc8: %Two = value_binding _, %.loc8_25.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc10: %pattern_type.830 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_38.1: %tuple.type.f94 = tuple_literal (%int_1.loc10, %int_2.loc10) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc10_13: type = splice_block %ThreeWithDefault.ref.loc10 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc10: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_38.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc10_34: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_34 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_34: init %i32 = call %bound_method.loc10_34.2(%int_1.loc10) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_34 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_1.loc10, %.loc10_34.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc10_37: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc10_37.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc10_37: <specific function> = specific_function %impl.elem0.loc10_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_37.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_37 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_37: init %i32 = call %bound_method.loc10_37.2(%int_2.loc10) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc10_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_37 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc10_37.2: %i32 = converted %int_2.loc10, %.loc10_37.1 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %addr.loc10: %ptr.da2 = addr_of %.loc10_38.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.be9bad.1(%.loc10_34.2, %.loc10_37.2, %addr.loc10)
// CHECK:STDOUT:   %.loc10_38.3: init %ThreeWithDefault to %.loc10_38.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_38.4: init %ThreeWithDefault = converted %.loc10_38.1, %.loc10_38.3
// CHECK:STDOUT:   %.loc10_38.5: ref %ThreeWithDefault = temporary %.loc10_38.2, %.loc10_38.4
// CHECK:STDOUT:   %.loc10_38.6: %ThreeWithDefault = acquire_value %.loc10_38.5
// CHECK:STDOUT:   %_.loc10: %ThreeWithDefault = value_binding _, %.loc10_38.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.830 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc12_41.1: %tuple.type.37f = tuple_literal (%int_1.loc12, %int_2.loc12, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %ThreeWithDefault.ref.loc12 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc12: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_41.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc12_34: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_34 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_34: init %i32 = call %bound_method.loc12_34.2(%int_1.loc12) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_34 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_1.loc12, %.loc12_34.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc12_37: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc12_37.1: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc12_37: <specific function> = specific_function %impl.elem0.loc12_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_37.2: <bound method> = bound_method %int_2.loc12, %specific_fn.loc12_37 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_37: init %i32 = call %bound_method.loc12_37.2(%int_2.loc12) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc12_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_37 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc12_37.2: %i32 = converted %int_2.loc12, %.loc12_37.1 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %impl.elem0.loc12_40: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc12_40.1: <bound method> = bound_method %int_3, %impl.elem0.loc12_40 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a]
// CHECK:STDOUT:   %specific_fn.loc12_40: <specific function> = specific_function %impl.elem0.loc12_40, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.2: <bound method> = bound_method %int_3, %specific_fn.loc12_40 [concrete = constants.%bound_method.353]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_40: init %i32 = call %bound_method.loc12_40.2(%int_3) [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc12_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_40 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc12_40.2: %i32 = converted %int_3, %.loc12_40.1 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %addr.loc12: %ptr.da2 = addr_of %.loc12_41.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.be9bad.2(%.loc12_34.2, %.loc12_37.2, %.loc12_40.2, %addr.loc12)
// CHECK:STDOUT:   %.loc12_41.3: init %ThreeWithDefault to %.loc12_41.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc12
// CHECK:STDOUT:   %.loc12_41.4: init %ThreeWithDefault = converted %.loc12_41.1, %.loc12_41.3
// CHECK:STDOUT:   %.loc12_41.5: ref %ThreeWithDefault = temporary %.loc12_41.2, %.loc12_41.4
// CHECK:STDOUT:   %.loc12_41.6: %ThreeWithDefault = acquire_value %.loc12_41.5
// CHECK:STDOUT:   %_.loc12: %ThreeWithDefault = value_binding _, %.loc12_41.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ThreeWithDefault.Op.bound.loc12: <bound method> = bound_method %.loc12_41.5, constants.%ThreeWithDefault.Op
// CHECK:STDOUT:   %Op.ref.loc12: %ThreeWithDefault.cpp_destructor.type = name_ref Op, imports.%ThreeWithDefault.cpp_destructor.decl [concrete = constants.%ThreeWithDefault.cpp_destructor]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.bound.loc12: <bound method> = bound_method %.loc12_41.5, %Op.ref.loc12
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.call.loc12: init %empty_tuple.type = call %ThreeWithDefault.cpp_destructor.bound.loc12(%.loc12_41.5)
// CHECK:STDOUT:   %ThreeWithDefault.Op.bound.loc10: <bound method> = bound_method %.loc10_38.5, constants.%ThreeWithDefault.Op
// CHECK:STDOUT:   %Op.ref.loc10: %ThreeWithDefault.cpp_destructor.type = name_ref Op, imports.%ThreeWithDefault.cpp_destructor.decl [concrete = constants.%ThreeWithDefault.cpp_destructor]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_38.5, %Op.ref.loc10
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.call.loc10: init %empty_tuple.type = call %ThreeWithDefault.cpp_destructor.bound.loc10(%.loc10_38.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Two.Op.bound: <bound method> = bound_method %.loc8_25.5, constants.%Two.Op
// CHECK:STDOUT:   %Op.ref.loc8: %Two.cpp_destructor.type = name_ref Op, imports.%Two.cpp_destructor.decl [concrete = constants.%Two.cpp_destructor]
// CHECK:STDOUT:   %Two.cpp_destructor.bound: <bound method> = bound_method %.loc8_25.5, %Op.ref.loc8
// CHECK:STDOUT:   %Two.cpp_destructor.call: init %empty_tuple.type = call %Two.cpp_destructor.bound(%.loc8_25.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_aggregate_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Aggregate: type = class_type @Aggregate [concrete]
// CHECK:STDOUT:   %pattern_type.f0b: type = pattern_type %Aggregate [concrete]
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
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.f0b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Aggregate.ref.loc15 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc15: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27.2: %Aggregate = converted %.loc15_27.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc23: %pattern_type.f0b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc23_29.1: %tuple.type.985 = tuple_literal (%int_1.loc23) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %Aggregate.ref.loc23 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc23: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_29.2: %Aggregate = converted %.loc23_29.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc23: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc31: %pattern_type.f0b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc31_31.1: %tuple.type.f94 = tuple_literal (%int_1.loc31, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc31_13: type = splice_block %Aggregate.ref.loc31 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc31: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31_31.2: %Aggregate = converted %.loc31_31.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc31: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_aggregate_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Aggregate: type = class_type @Aggregate [concrete]
// CHECK:STDOUT:   %pattern_type.f0b: type = pattern_type %Aggregate [concrete]
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
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.f0b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Aggregate.ref.loc15 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc15: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27.2: %Aggregate = converted %.loc15_27.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc23: %pattern_type.f0b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc23_33.1: %struct_type.x = struct_literal (%int_1.loc23) [concrete = constants.%struct.147]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %Aggregate.ref.loc23 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc23: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_33.2: %Aggregate = converted %.loc23_33.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc23: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc31: %pattern_type.f0b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc31_41.1: %struct_type.x.y = struct_literal (%int_1.loc31, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc31_13: type = splice_block %Aggregate.ref.loc31 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc31: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31_41.2: %Aggregate = converted %.loc31_41.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc31: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_aggregate_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonAggregate: type = class_type @NonAggregate [concrete]
// CHECK:STDOUT:   %pattern_type.d5f: type = pattern_type %NonAggregate [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.9fe: type = ptr_type %NonAggregate [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.67cd37.1: type = fn_type @NonAggregate__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.30ebf5.1: %NonAggregate__carbon_thunk.type.67cd37.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.67cd37.2: type = fn_type @NonAggregate__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.30ebf5.2: %NonAggregate__carbon_thunk.type.67cd37.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.67cd37.3: type = fn_type @NonAggregate__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.30ebf5.3: %NonAggregate__carbon_thunk.type.67cd37.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
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
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.2d9ba4.1: %NonAggregate__carbon_thunk.type.67cd37.1 = fn_decl @NonAggregate__carbon_thunk.1 [concrete = constants.%NonAggregate__carbon_thunk.30ebf5.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.2d9ba4.2: %NonAggregate__carbon_thunk.type.67cd37.2 = fn_decl @NonAggregate__carbon_thunk.2 [concrete = constants.%NonAggregate__carbon_thunk.30ebf5.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.2d9ba4.3: %NonAggregate__carbon_thunk.type.67cd37.3 = fn_decl @NonAggregate__carbon_thunk.3 [concrete = constants.%NonAggregate__carbon_thunk.30ebf5.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.decl: %NonAggregate.cpp_destructor.type = fn_decl @NonAggregate.cpp_destructor [concrete = constants.%NonAggregate.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc8: %pattern_type.d5f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %NonAggregate.ref.loc8 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc8: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.9fe = addr_of %.loc8_30.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.2d9ba4.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_30.3: init %NonAggregate to %.loc8_30.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_30.4: init %NonAggregate = converted %.loc8_30.1, %.loc8_30.3
// CHECK:STDOUT:   %.loc8_30.5: ref %NonAggregate = temporary %.loc8_30.2, %.loc8_30.4
// CHECK:STDOUT:   %.loc8_30.6: %NonAggregate = acquire_value %.loc8_30.5
// CHECK:STDOUT:   %_.loc8: %NonAggregate = value_binding _, %.loc8_30.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc9: %pattern_type.d5f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_32.1: %tuple.type.985 = tuple_literal (%int_1.loc9) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %NonAggregate.ref.loc9 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc9: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_32.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_30.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_30.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_30.2(%int_1.loc9) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc9_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc9_30.2: %i32 = converted %int_1.loc9, %.loc9_30.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %addr.loc9: %ptr.9fe = addr_of %.loc9_32.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.2d9ba4.2(%.loc9_30.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_32.3: init %NonAggregate to %.loc9_32.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_32.4: init %NonAggregate = converted %.loc9_32.1, %.loc9_32.3
// CHECK:STDOUT:   %.loc9_32.5: ref %NonAggregate = temporary %.loc9_32.2, %.loc9_32.4
// CHECK:STDOUT:   %.loc9_32.6: %NonAggregate = acquire_value %.loc9_32.5
// CHECK:STDOUT:   %_.loc9: %NonAggregate = value_binding _, %.loc9_32.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc10: %pattern_type.d5f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_34.1: %tuple.type.f94 = tuple_literal (%int_1.loc10, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc10_13: type = splice_block %NonAggregate.ref.loc10 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc10: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_34.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10_30: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc10_30.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc10_30: <specific function> = specific_function %impl.elem0.loc10_30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_30.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_30 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_30: init %i32 = call %bound_method.loc10_30.2(%int_1.loc10) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc10_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_30 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc10_30.2: %i32 = converted %int_1.loc10, %.loc10_30.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc10_33: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc10_33.1: <bound method> = bound_method %int_2, %impl.elem0.loc10_33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc10_33: <specific function> = specific_function %impl.elem0.loc10_33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_33.2: <bound method> = bound_method %int_2, %specific_fn.loc10_33 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_33: init %i32 = call %bound_method.loc10_33.2(%int_2) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc10_33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_33 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc10_33.2: %i32 = converted %int_2, %.loc10_33.1 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %addr.loc10: %ptr.9fe = addr_of %.loc10_34.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.2d9ba4.3(%.loc10_30.2, %.loc10_33.2, %addr.loc10)
// CHECK:STDOUT:   %.loc10_34.3: init %NonAggregate to %.loc10_34.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_34.4: init %NonAggregate = converted %.loc10_34.1, %.loc10_34.3
// CHECK:STDOUT:   %.loc10_34.5: ref %NonAggregate = temporary %.loc10_34.2, %.loc10_34.4
// CHECK:STDOUT:   %.loc10_34.6: %NonAggregate = acquire_value %.loc10_34.5
// CHECK:STDOUT:   %_.loc10: %NonAggregate = value_binding _, %.loc10_34.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NonAggregate.Op.bound.loc10: <bound method> = bound_method %.loc10_34.5, constants.%NonAggregate.Op
// CHECK:STDOUT:   %Op.ref.loc10: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_34.5, %Op.ref.loc10
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.call.loc10: init %empty_tuple.type = call %NonAggregate.cpp_destructor.bound.loc10(%.loc10_34.5)
// CHECK:STDOUT:   %NonAggregate.Op.bound.loc9: <bound method> = bound_method %.loc9_32.5, constants.%NonAggregate.Op
// CHECK:STDOUT:   %Op.ref.loc9: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_32.5, %Op.ref.loc9
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.call.loc9: init %empty_tuple.type = call %NonAggregate.cpp_destructor.bound.loc9(%.loc9_32.5)
// CHECK:STDOUT:   %NonAggregate.Op.bound.loc8: <bound method> = bound_method %.loc8_30.5, constants.%NonAggregate.Op
// CHECK:STDOUT:   %Op.ref.loc8: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_30.5, %Op.ref.loc8
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.call.loc8: init %empty_tuple.type = call %NonAggregate.cpp_destructor.bound.loc8(%.loc8_30.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_aggregate_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonAggregate: type = class_type @NonAggregate [concrete]
// CHECK:STDOUT:   %pattern_type.d5f: type = pattern_type %NonAggregate [concrete]
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
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.d5f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %NonAggregate.ref.loc15 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc15: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_30.2: %NonAggregate = converted %.loc15_30.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc23: %pattern_type.d5f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc23_36.1: %struct_type.x = struct_literal (%int_1.loc23) [concrete = constants.%struct.147]
// CHECK:STDOUT:   %.loc23_13: type = splice_block %NonAggregate.ref.loc23 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc23: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_36.2: %NonAggregate = converted %.loc23_36.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc23: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc31: %pattern_type.d5f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc31_44.1: %struct_type.x.y = struct_literal (%int_1.loc31, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc31_13: type = splice_block %NonAggregate.ref.loc31 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc31: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31_44.2: %NonAggregate = converted %.loc31_44.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc31: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
