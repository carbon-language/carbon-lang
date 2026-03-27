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
// CHECK:STDOUT:   %.d95: ref %Dest = temporary invalid, @UserConversion.%.loc12_21.3 [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonConstConversion.480: type = class_type @NonConstConversion.1 [concrete]
// CHECK:STDOUT:   %pattern_type.b08: type = pattern_type %NonConstConversion.480 [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.type: type = fn_type @NonConstConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator: %NonConstConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.2: %Dest__carbon_thunk.type.2ffc9e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.7f2: ref %Dest = temporary invalid, @NonConstConversion.loc16.%.loc22_21.3 [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.39d: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %ptr.472: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.9ae: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7a8: ref %Dest2 = temporary invalid, @ConstructorConversion.%.loc32_22.4 [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor.type: type = fn_type @Dest2.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor: %Dest2.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc12_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc12_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc12_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc12_21.2: init %Dest to %.loc12_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_21.3: init %Dest = converted %s.ref, %.loc12_21.2
// CHECK:STDOUT:   %.loc12_21.4: ref %Dest = temporary %.loc12_21.1, %.loc12_21.3 [concrete = constants.%.d95]
// CHECK:STDOUT:   %.loc12_21.5: %Dest = acquire_value %.loc12_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc12_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversion.loc16(%s.param: ref %NonConstConversion.480) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: ref %NonConstConversion.480 = name_ref s, %s
// CHECK:STDOUT:   %.loc22_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%NonConstConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc22_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc22_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc22_21.2: init %Dest to %.loc22_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc22_21.3: init %Dest = converted %s.ref, %.loc22_21.2
// CHECK:STDOUT:   %.loc22_21.4: ref %Dest = temporary %.loc22_21.1, %.loc22_21.3 [concrete = constants.%.7f2]
// CHECK:STDOUT:   %.loc22_21.5: %Dest = acquire_value %.loc22_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc22_21.5
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.39d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %.loc32_13: type = splice_block %Dest2.ref [concrete = constants.%Dest2] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc32_22.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc32_22.2: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc32_22.1: %ptr.472 = addr_of %.loc32_22.2
// CHECK:STDOUT:   %addr.loc32_22.2: %ptr.9ae = addr_of %.loc32_22.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc32_22.1, %addr.loc32_22.2)
// CHECK:STDOUT:   %.loc32_22.3: init %Dest2 to %.loc32_22.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc32_22.4: init %Dest2 = converted %s.ref, %.loc32_22.3
// CHECK:STDOUT:   %.loc32_22.5: ref %Dest2 = temporary %.loc32_22.1, %.loc32_22.4 [concrete = constants.%.7a8]
// CHECK:STDOUT:   %.loc32_22.6: %Dest2 = acquire_value %.loc32_22.5
// CHECK:STDOUT:   %_: %Dest2 = value_binding _, %.loc32_22.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
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
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_21: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %.1bb: ref %InaccessibleConstructor = temporary invalid, @InaccessibleConstructorTest.%.loc20_40.4 [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.type: type = fn_type @InaccessibleConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor: %InaccessibleConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConversion: type = class_type @InaccessibleConversion [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %InaccessibleConversion [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.type: type = fn_type @InaccessibleConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator: %InaccessibleConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.d61: ref %Dest = temporary invalid, @InaccessibleConversionTest.%.loc38_21.3 [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc20_13: type = splice_block %InaccessibleConstructor.ref [concrete = constants.%InaccessibleConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InaccessibleConstructor.ref: type = name_ref InaccessibleConstructor, imports.%InaccessibleConstructor.decl [concrete = constants.%InaccessibleConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_40.1: ref %InaccessibleConstructor = temporary_storage
// CHECK:STDOUT:   %.loc20_40.2: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc20_40.1: %ptr.1fc = addr_of %.loc20_40.2
// CHECK:STDOUT:   %addr.loc20_40.2: %ptr.12d = addr_of %.loc20_40.1
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InaccessibleConstructor__carbon_thunk.decl(%addr.loc20_40.1, %addr.loc20_40.2)
// CHECK:STDOUT:   %.loc20_40.3: init %InaccessibleConstructor to %.loc20_40.1 = mark_in_place_init %InaccessibleConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_40.4: init %InaccessibleConstructor = converted %s.ref, %.loc20_40.3
// CHECK:STDOUT:   %.loc20_40.5: ref %InaccessibleConstructor = temporary %.loc20_40.1, %.loc20_40.4 [concrete = constants.%.1bb]
// CHECK:STDOUT:   %.loc20_40.6: %InaccessibleConstructor = acquire_value %.loc20_40.5
// CHECK:STDOUT:   %_: %InaccessibleConstructor = value_binding _, %.loc20_40.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %InaccessibleConstructor.cpp_destructor.type = name_ref Op, imports.%InaccessibleConstructor.cpp_destructor.decl [concrete = constants.%InaccessibleConstructor.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConversionTest(%s.param: %InaccessibleConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %InaccessibleConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc38_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%InaccessibleConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc38_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc38_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc38_21.2: init %Dest to %.loc38_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc38_21.3: init %Dest = converted %s.ref, %.loc38_21.2
// CHECK:STDOUT:   %.loc38_21.4: ref %Dest = temporary %.loc38_21.1, %.loc38_21.3 [concrete = constants.%.d61]
// CHECK:STDOUT:   %.loc38_21.5: %Dest = acquire_value %.loc38_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc38_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %DeletedConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc28_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc28_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %.89b: ref %IntConstructor.f49 = temporary invalid, @IntConstructor.loc6.%.loc12_31.3 [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.type: type = fn_type @IntConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor: %IntConstructor.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT: fn @IntConstructor.loc6(%i.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.3c3 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc12_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor.f49] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor.f49]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_31.1: ref %IntConstructor.f49 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d59 = addr_of %.loc12_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(%i.ref, %addr)
// CHECK:STDOUT:   %.loc12_31.2: init %IntConstructor.f49 to %.loc12_31.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_31.3: init %IntConstructor.f49 = converted %i.ref, %.loc12_31.2
// CHECK:STDOUT:   %.loc12_31.4: ref %IntConstructor.f49 = temporary %.loc12_31.1, %.loc12_31.3 [concrete = constants.%.89b]
// CHECK:STDOUT:   %.loc12_31.5: %IntConstructor.f49 = acquire_value %.loc12_31.4
// CHECK:STDOUT:   %_: %IntConstructor.f49 = value_binding _, %.loc12_31.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %IntConstructor.cpp_destructor.type = name_ref Op, imports.%IntConstructor.cpp_destructor.decl [concrete = constants.%IntConstructor.cpp_destructor]
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
// CHECK:STDOUT:   %.55b: ref %IntConstructor = temporary invalid, @IntConstructorTest.%.loc23_31.4 [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.type: type = fn_type @IntConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor: %IntConstructor.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc23_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_31.1: ref %IntConstructor = temporary_storage
// CHECK:STDOUT:   %.loc23_31.2: %i32 = converted %u.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.d59 = addr_of %.loc23_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(<error>, %addr)
// CHECK:STDOUT:   %.loc23_31.3: init %IntConstructor to %.loc23_31.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc23_31.4: init %IntConstructor = converted %u.ref, %.loc23_31.3
// CHECK:STDOUT:   %.loc23_31.5: ref %IntConstructor = temporary %.loc23_31.1, %.loc23_31.4 [concrete = constants.%.55b]
// CHECK:STDOUT:   %.loc23_31.6: %IntConstructor = acquire_value %.loc23_31.5
// CHECK:STDOUT:   %_: %IntConstructor = value_binding _, %.loc23_31.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %IntConstructor.cpp_destructor.type = name_ref Op, imports.%IntConstructor.cpp_destructor.decl [concrete = constants.%IntConstructor.cpp_destructor]
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
// CHECK:STDOUT:   %.fa2: ref %DefaultConstructor = temporary invalid, @DefaultConstructorTest.%.loc13_36.4 [concrete]
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor.type: type = fn_type @DefaultConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefaultConstructor.cpp_destructor: %DefaultConstructor.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc13_36.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_13: type = splice_block %DefaultConstructor.ref [concrete = constants.%DefaultConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultConstructor.ref: type = name_ref DefaultConstructor, imports.%DefaultConstructor.decl [concrete = constants.%DefaultConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_36.2: ref %DefaultConstructor = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.3f7 = addr_of %.loc13_36.2
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%DefaultConstructor__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc13_36.3: init %DefaultConstructor to %.loc13_36.2 = mark_in_place_init %DefaultConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc13_36.4: init %DefaultConstructor = converted %.loc13_36.1, %.loc13_36.3
// CHECK:STDOUT:   %.loc13_36.5: ref %DefaultConstructor = temporary %.loc13_36.2, %.loc13_36.4 [concrete = constants.%.fa2]
// CHECK:STDOUT:   %.loc13_36.6: %DefaultConstructor = acquire_value %.loc13_36.5
// CHECK:STDOUT:   %_: %DefaultConstructor = value_binding _, %.loc13_36.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %DefaultConstructor.cpp_destructor.type = name_ref Op, imports.%DefaultConstructor.cpp_destructor.decl [concrete = constants.%DefaultConstructor.cpp_destructor]
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
// CHECK:STDOUT:   %.375: ref %Two = temporary invalid, @ImplicitConvert.%.loc12_25.4 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault: type = class_type @ThreeWithDefault [concrete]
// CHECK:STDOUT:   %pattern_type.ca7: type = pattern_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ptr.ea4: type = ptr_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.b64454.1: type = fn_type @ThreeWithDefault__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.fd88b7.1: %ThreeWithDefault__carbon_thunk.type.b64454.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.0c5: ref %ThreeWithDefault = temporary invalid, @ImplicitConvert.%.loc18_38.4 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.b64454.2: type = fn_type @ThreeWithDefault__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.fd88b7.2: %ThreeWithDefault__carbon_thunk.type.b64454.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %.fc9: ref %ThreeWithDefault = temporary invalid, @ImplicitConvert.%.loc24_41.4 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor.type: type = fn_type @ThreeWithDefault.cpp_destructor [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.cpp_destructor: %ThreeWithDefault.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Two.cpp_destructor.type: type = fn_type @Two.cpp_destructor [concrete]
// CHECK:STDOUT:   %Two.cpp_destructor: %Two.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
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
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.4c2 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc12_25.1: %tuple.type.f94 = tuple_literal (%int_1.loc12, %int_2.loc12) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %Two.ref [concrete = constants.%Two] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Two.ref: type = name_ref Two, imports.%Two.decl [concrete = constants.%Two]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_25.2: ref %Two = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_21: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_21.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc12_21: <specific function> = specific_function %impl.elem0.loc12_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_21.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_21 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_21: init %i32 = call %bound_method.loc12_21.2(%int_1.loc12) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_21 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int_1.loc12, %.loc12_21.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc12_24: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_24.1: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc12_24: <specific function> = specific_function %impl.elem0.loc12_24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_24.2: <bound method> = bound_method %int_2.loc12, %specific_fn.loc12_24 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_24: init %i32 = call %bound_method.loc12_24.2(%int_2.loc12) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_24 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_24.2: %i32 = converted %int_2.loc12, %.loc12_24.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %addr.loc12: %ptr.677 = addr_of %.loc12_25.2
// CHECK:STDOUT:   %Two__carbon_thunk.call: init %empty_tuple.type = call imports.%Two__carbon_thunk.decl(%.loc12_21.2, %.loc12_24.2, %addr.loc12)
// CHECK:STDOUT:   %.loc12_25.3: init %Two to %.loc12_25.2 = mark_in_place_init %Two__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_25.4: init %Two = converted %.loc12_25.1, %.loc12_25.3
// CHECK:STDOUT:   %.loc12_25.5: ref %Two = temporary %.loc12_25.2, %.loc12_25.4 [concrete = constants.%.375]
// CHECK:STDOUT:   %.loc12_25.6: %Two = acquire_value %.loc12_25.5
// CHECK:STDOUT:   %_.loc12: %Two = value_binding _, %.loc12_25.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc18: %pattern_type.ca7 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc18: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_38.1: %tuple.type.f94 = tuple_literal (%int_1.loc18, %int_2.loc18) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc18_13: type = splice_block %ThreeWithDefault.ref.loc18 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc18: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_38.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc18_34: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc18_34.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc18_34: <specific function> = specific_function %impl.elem0.loc18_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18_34 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34: init %i32 = call %bound_method.loc18_34.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_34.2: %i32 = converted %int_1.loc18, %.loc18_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc18_37: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc18_37.1: <bound method> = bound_method %int_2.loc18, %impl.elem0.loc18_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc18_37: <specific function> = specific_function %impl.elem0.loc18_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_37.2: <bound method> = bound_method %int_2.loc18, %specific_fn.loc18_37 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_37: init %i32 = call %bound_method.loc18_37.2(%int_2.loc18) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_37 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_37.2: %i32 = converted %int_2.loc18, %.loc18_37.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %addr.loc18: %ptr.ea4 = addr_of %.loc18_38.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc18: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.482856.1(%.loc18_34.2, %.loc18_37.2, %addr.loc18)
// CHECK:STDOUT:   %.loc18_38.3: init %ThreeWithDefault to %.loc18_38.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc18
// CHECK:STDOUT:   %.loc18_38.4: init %ThreeWithDefault = converted %.loc18_38.1, %.loc18_38.3
// CHECK:STDOUT:   %.loc18_38.5: ref %ThreeWithDefault = temporary %.loc18_38.2, %.loc18_38.4 [concrete = constants.%.0c5]
// CHECK:STDOUT:   %.loc18_38.6: %ThreeWithDefault = acquire_value %.loc18_38.5
// CHECK:STDOUT:   %_.loc18: %ThreeWithDefault = value_binding _, %.loc18_38.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc24: %pattern_type.ca7 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc24: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc24_41.1: %tuple.type.37f = tuple_literal (%int_1.loc24, %int_2.loc24, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %.loc24_13: type = splice_block %ThreeWithDefault.ref.loc24 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc24: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_41.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc24_34: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24_34.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc24_34: <specific function> = specific_function %impl.elem0.loc24_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_34.2: <bound method> = bound_method %int_1.loc24, %specific_fn.loc24_34 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24_34: init %i32 = call %bound_method.loc24_34.2(%int_1.loc24) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc24_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24_34 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc24_34.2: %i32 = converted %int_1.loc24, %.loc24_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc24_37: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24_37.1: <bound method> = bound_method %int_2.loc24, %impl.elem0.loc24_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc24_37: <specific function> = specific_function %impl.elem0.loc24_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_37.2: <bound method> = bound_method %int_2.loc24, %specific_fn.loc24_37 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24_37: init %i32 = call %bound_method.loc24_37.2(%int_2.loc24) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc24_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24_37 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc24_37.2: %i32 = converted %int_2.loc24, %.loc24_37.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc24_40: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24_40.1: <bound method> = bound_method %int_3, %impl.elem0.loc24_40 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc24_40: <specific function> = specific_function %impl.elem0.loc24_40, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_40.2: <bound method> = bound_method %int_3, %specific_fn.loc24_40 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24_40: init %i32 = call %bound_method.loc24_40.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc24_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24_40 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc24_40.2: %i32 = converted %int_3, %.loc24_40.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %addr.loc24: %ptr.ea4 = addr_of %.loc24_41.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc24: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.482856.2(%.loc24_34.2, %.loc24_37.2, %.loc24_40.2, %addr.loc24)
// CHECK:STDOUT:   %.loc24_41.3: init %ThreeWithDefault to %.loc24_41.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc24
// CHECK:STDOUT:   %.loc24_41.4: init %ThreeWithDefault = converted %.loc24_41.1, %.loc24_41.3
// CHECK:STDOUT:   %.loc24_41.5: ref %ThreeWithDefault = temporary %.loc24_41.2, %.loc24_41.4 [concrete = constants.%.fc9]
// CHECK:STDOUT:   %.loc24_41.6: %ThreeWithDefault = acquire_value %.loc24_41.5
// CHECK:STDOUT:   %_.loc24: %ThreeWithDefault = value_binding _, %.loc24_41.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref.loc24: %ThreeWithDefault.cpp_destructor.type = name_ref Op, imports.%ThreeWithDefault.cpp_destructor.decl [concrete = constants.%ThreeWithDefault.cpp_destructor]
// CHECK:STDOUT:   %Op.ref.loc18: %ThreeWithDefault.cpp_destructor.type = name_ref Op, imports.%ThreeWithDefault.cpp_destructor.decl [concrete = constants.%ThreeWithDefault.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref.loc12: %Two.cpp_destructor.type = name_ref Op, imports.%Two.cpp_destructor.decl [concrete = constants.%Two.cpp_destructor]
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
// CHECK:STDOUT:   %.loc15_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Aggregate.ref.loc15 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc15: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27.2: %Aggregate = converted %.loc15_27.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc23: %pattern_type.235 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:     %_.patt.loc31: %pattern_type.235 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %.loc15_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Aggregate.ref.loc15 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc15: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27.2: %Aggregate = converted %.loc15_27.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %Aggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc23: %pattern_type.235 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:     %_.patt.loc31: %pattern_type.235 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %pattern_type.eac: type = pattern_type %NonAggregate [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.9b6: type = ptr_type %NonAggregate [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.8d64dd.1: type = fn_type @NonAggregate__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.0020a5.1: %NonAggregate__carbon_thunk.type.8d64dd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.217: ref %NonAggregate = temporary invalid, @InitFromStruct.%.loc12_30.4 [concrete]
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
// CHECK:STDOUT:   %.2d9: ref %NonAggregate = temporary invalid, @InitFromStruct.%.loc17_32.4 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.8d64dd.3: type = fn_type @NonAggregate__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.0020a5.3: %NonAggregate__carbon_thunk.type.8d64dd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %.ea6: ref %NonAggregate = temporary invalid, @InitFromStruct.%.loc22_34.4 [concrete]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor.type: type = fn_type @NonAggregate.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonAggregate.cpp_destructor: %NonAggregate.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
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
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %NonAggregate.ref.loc12 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc12: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_30.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.9b6 = addr_of %.loc12_30.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.712b28.1(%addr.loc12)
// CHECK:STDOUT:   %.loc12_30.3: init %NonAggregate to %.loc12_30.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc12
// CHECK:STDOUT:   %.loc12_30.4: init %NonAggregate = converted %.loc12_30.1, %.loc12_30.3
// CHECK:STDOUT:   %.loc12_30.5: ref %NonAggregate = temporary %.loc12_30.2, %.loc12_30.4 [concrete = constants.%.217]
// CHECK:STDOUT:   %.loc12_30.6: %NonAggregate = acquire_value %.loc12_30.5
// CHECK:STDOUT:   %_.loc12: %NonAggregate = value_binding _, %.loc12_30.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc17: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc17_32.1: %tuple.type.985 = tuple_literal (%int_1.loc17) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc17_13: type = splice_block %NonAggregate.ref.loc17 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc17: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_32.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc17: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc17_30.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_30.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %i32 = call %bound_method.loc17_30.2(%int_1.loc17) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_30.2: %i32 = converted %int_1.loc17, %.loc17_30.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %addr.loc17: %ptr.9b6 = addr_of %.loc17_32.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.712b28.2(%.loc17_30.2, %addr.loc17)
// CHECK:STDOUT:   %.loc17_32.3: init %NonAggregate to %.loc17_32.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc17
// CHECK:STDOUT:   %.loc17_32.4: init %NonAggregate = converted %.loc17_32.1, %.loc17_32.3
// CHECK:STDOUT:   %.loc17_32.5: ref %NonAggregate = temporary %.loc17_32.2, %.loc17_32.4 [concrete = constants.%.2d9]
// CHECK:STDOUT:   %.loc17_32.6: %NonAggregate = acquire_value %.loc17_32.5
// CHECK:STDOUT:   %_.loc17: %NonAggregate = value_binding _, %.loc17_32.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc22: %pattern_type.eac = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_34.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc22_13: type = splice_block %NonAggregate.ref.loc22 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc22: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_34.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc22_30: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc22_30.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc22_30: <specific function> = specific_function %impl.elem0.loc22_30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_30.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_30 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_30: init %i32 = call %bound_method.loc22_30.2(%int_1.loc22) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_30 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_30.2: %i32 = converted %int_1.loc22, %.loc22_30.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc22_33: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc22_33.1: <bound method> = bound_method %int_2, %impl.elem0.loc22_33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc22_33: <specific function> = specific_function %impl.elem0.loc22_33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_33.2: <bound method> = bound_method %int_2, %specific_fn.loc22_33 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_33: init %i32 = call %bound_method.loc22_33.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_33 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_33.2: %i32 = converted %int_2, %.loc22_33.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %addr.loc22: %ptr.9b6 = addr_of %.loc22_34.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.712b28.3(%.loc22_30.2, %.loc22_33.2, %addr.loc22)
// CHECK:STDOUT:   %.loc22_34.3: init %NonAggregate to %.loc22_34.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_34.4: init %NonAggregate = converted %.loc22_34.1, %.loc22_34.3
// CHECK:STDOUT:   %.loc22_34.5: ref %NonAggregate = temporary %.loc22_34.2, %.loc22_34.4 [concrete = constants.%.ea6]
// CHECK:STDOUT:   %.loc22_34.6: %NonAggregate = acquire_value %.loc22_34.5
// CHECK:STDOUT:   %_.loc22: %NonAggregate = value_binding _, %.loc22_34.6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref.loc22: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %Op.ref.loc17: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
// CHECK:STDOUT:   %Op.ref.loc12: %NonAggregate.cpp_destructor.type = name_ref Op, imports.%NonAggregate.cpp_destructor.decl [concrete = constants.%NonAggregate.cpp_destructor]
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
// CHECK:STDOUT:   %.loc15_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %NonAggregate.ref.loc15 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc15: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_30.2: %NonAggregate = converted %.loc15_30.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc15: %NonAggregate = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc23: %pattern_type.eac = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:     %_.patt.loc31: %pattern_type.eac = value_binding_pattern _ [concrete]
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
