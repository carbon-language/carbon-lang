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
  // CHECK:STDERR:        ╭─┤ fail_expr_category.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.Dest = s;
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                     │  ┃        `NonConstConversion` to `Cpp.Dest`
  // CHECK:STDERR:        ·                     │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                     ╰────┤ type `Cpp.NonConstConversion` does not implement
  // CHECK:STDERR:        ·                          │ interface `Core.ImplicitAs(Cpp.Dest)`
  // CHECK:STDERR:        ·                          │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_access.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn InaccessibleConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_access.carbon:[[@LINE+15]]:40
  // CHECK:STDERR: ->     │   let _: Cpp.InaccessibleConstructor = s;
  // CHECK:STDERR:        ·                                        ┳
  // CHECK:STDERR:        ·                                        ┗━━┫ error: cannot access private member
  // CHECK:STDERR:        ·                                           ┃        `InaccessibleConstructor` of
  // CHECK:STDERR:        ·                                           ┃        type
  // CHECK:STDERR:        ·                                           ┃        `Cpp.InaccessibleConstructor`
  // CHECK:STDERR:        ·                                           ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./implicit_conversions.h
  // CHECK:STDERR:        │   InaccessibleConstructor(Source);
  // CHECK:STDERR:        ·   ┬
  // CHECK:STDERR:        ·   ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.InaccessibleConstructor = s;
  //@dump-sem-ir-end
}

fn InaccessibleConversionTest(s: Cpp.InaccessibleConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_access.carbon:[[@LINE+13]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.Dest = s;
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot access private member `<C++ operator>` of
  // CHECK:STDERR:        ·                        ┃        type `Cpp.InaccessibleConversion`
  // CHECK:STDERR:        ·                        ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./implicit_conversions.h
  // CHECK:STDERR:        │   operator Dest() const;
  // CHECK:STDERR:        ·   ┬
  // CHECK:STDERR:        ·   ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn DeletedConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_deleted.carbon:[[@LINE+13]]:35
  // CHECK:STDERR: ->     │   let _: Cpp.DeletedConstructor = s;
  // CHECK:STDERR:        ·                                   ┳
  // CHECK:STDERR:        ·                                   ┗━━┫ error: cannot implicitly convert
  // CHECK:STDERR:        ·                                   │  ┃        expression of type `Source` to
  // CHECK:STDERR:        ·                                   │  ┃        `Cpp.DeletedConstructor`
  // CHECK:STDERR:        ·                                   │  ┃        [ConversionFailure]
  // CHECK:STDERR:        · ╭─────────────────────────────────╯
  // CHECK:STDERR:        · │ type `Cpp.Source` does not implement interface
  // CHECK:STDERR:        · │ `Core.ImplicitAs(Cpp.DeletedConstructor)`
  // CHECK:STDERR:        · │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.DeletedConstructor = s;
  //@dump-sem-ir-end
}

fn DeletedConversionTest(s: Cpp.DeletedConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_deleted.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.Dest = s;
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                     │  ┃        `DeletedConversion` to `Cpp.Dest`
  // CHECK:STDERR:        ·                     │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                     ╰────┤ type `Cpp.DeletedConversion` does not implement
  // CHECK:STDERR:        ·                          │ interface `Core.ImplicitAs(Cpp.Dest)`
  // CHECK:STDERR:        ·                          │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_explicit.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn ExplicitConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_explicit.carbon:[[@LINE+13]]:36
  // CHECK:STDERR: ->     │   let _: Cpp.ExplicitConstructor = s;
  // CHECK:STDERR:        ·                                    ┳
  // CHECK:STDERR:        ·                                    ┗━━┫ error: cannot implicitly convert
  // CHECK:STDERR:        ·                                    │  ┃        expression of type `Source` to
  // CHECK:STDERR:        ·                                    │  ┃        `Cpp.ExplicitConstructor`
  // CHECK:STDERR:        ·                                    │  ┃        [ConversionFailure]
  // CHECK:STDERR:        · ╭──────────────────────────────────╯
  // CHECK:STDERR:        · │ type `Cpp.Source` does not implement interface
  // CHECK:STDERR:        · │ `Core.ImplicitAs(Cpp.ExplicitConstructor)`
  // CHECK:STDERR:        · │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.ExplicitConstructor = s;
  //@dump-sem-ir-end
}

fn ExplicitConversionTest(s: Cpp.ExplicitConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_explicit.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.Dest = s;
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                     │  ┃        `ExplicitConversion` to `Cpp.Dest`
  // CHECK:STDERR:        ·                     │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                     ╰────┤ type `Cpp.ExplicitConversion` does not implement
  // CHECK:STDERR:        ·                          │ interface `Core.ImplicitAs(Cpp.Dest)`
  // CHECK:STDERR:        ·                          │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE+16]]:31
  // CHECK:STDERR: ->     │   let _: Cpp.IntConstructor = u;
  // CHECK:STDERR:        ·                               ┳
  // CHECK:STDERR:        ·                               ┗━━┫ error: cannot implicitly convert expression
  // CHECK:STDERR:        ·                               │  ┃        of type `u32` to `i32`
  // CHECK:STDERR:        ·                               │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                               ╰────┤ type `u32` does not implement interface
  // CHECK:STDERR:        ·                                    │ `Core.ImplicitAs(i32)`
  // CHECK:STDERR:        ·                                    │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./implicit_conversions.h
  // CHECK:STDERR:        │   IntConstructor(int);
  // CHECK:STDERR:        ·                     ┬
  // CHECK:STDERR:        ·                     ╰──┤ initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_todo_aggregate_from_tuple.carbon:[[@LINE+11]]:26
  // CHECK:STDERR: ->     │   let _: Cpp.Aggregate = ();
  // CHECK:STDERR:        ·                          ┳━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                          │  ┃        type `()` to `Cpp.Aggregate`
  // CHECK:STDERR:        ·                          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                          ╰────┤ type `()` does not implement interface
  // CHECK:STDERR:        ·                               │ `Core.ImplicitAs(Cpp.Aggregate)`
  // CHECK:STDERR:        ·                               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Aggregate = ();
  // CHECK:STDERR:        ╭─┤ fail_todo_aggregate_from_tuple.carbon:[[@LINE+12]]:26
  // CHECK:STDERR: ->     │   let _: Cpp.Aggregate = (1,);
  // CHECK:STDERR:        ·                          ┳━━━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                          │  ┃        type `(Core.IntLiteral,)` to
  // CHECK:STDERR:        ·                          │  ┃        `Cpp.Aggregate`
  // CHECK:STDERR:        ·                          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                          ╰────┤ type `(Core.IntLiteral,)` does not implement
  // CHECK:STDERR:        ·                               │ interface `Core.ImplicitAs(Cpp.Aggregate)`
  // CHECK:STDERR:        ·                               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Aggregate = (1,);
  // CHECK:STDERR:        ╭─┤ fail_todo_aggregate_from_tuple.carbon:[[@LINE+13]]:26
  // CHECK:STDERR: ->     │   let _: Cpp.Aggregate = (1, 2);
  // CHECK:STDERR:        ·                          ┳━━━━━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                          │  ┃        type `(Core.IntLiteral, Core.IntLiteral)`
  // CHECK:STDERR:        ·                          │  ┃        to `Cpp.Aggregate`
  // CHECK:STDERR:        ·                          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                          ╰────┤ type `(Core.IntLiteral, Core.IntLiteral)` does
  // CHECK:STDERR:        ·                               │ not implement interface
  // CHECK:STDERR:        ·                               │ `Core.ImplicitAs(Cpp.Aggregate)`
  // CHECK:STDERR:        ·                               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Aggregate = (1, 2);
  //@dump-sem-ir-end
}

// --- fail_todo_aggregate_from_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "aggregate.h";

fn InitFromStruct() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_aggregate_from_struct.carbon:[[@LINE+11]]:26
  // CHECK:STDERR: ->     │   let _: Cpp.Aggregate = {};
  // CHECK:STDERR:        ·                          ┳━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                          │  ┃        type `{}` to `Cpp.Aggregate`
  // CHECK:STDERR:        ·                          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                          ╰────┤ type `{}` does not implement interface
  // CHECK:STDERR:        ·                               │ `Core.ImplicitAs(Cpp.Aggregate)`
  // CHECK:STDERR:        ·                               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Aggregate = {};
  // CHECK:STDERR:        ╭─┤ fail_todo_aggregate_from_struct.carbon:[[@LINE+12]]:26
  // CHECK:STDERR: ->     │   let _: Cpp.Aggregate = {.x = 1};
  // CHECK:STDERR:        ·                          ┳━━━━━━━
  // CHECK:STDERR:        ·                          ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                          │  ┃        type `{.x: Core.IntLiteral}` to
  // CHECK:STDERR:        ·                          │  ┃        `Cpp.Aggregate`
  // CHECK:STDERR:        ·                          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                          ╰────┤ type `{.x: Core.IntLiteral}` does not implement
  // CHECK:STDERR:        ·                               │ interface `Core.ImplicitAs(Cpp.Aggregate)`
  // CHECK:STDERR:        ·                               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.Aggregate = {.x = 1};
  // CHECK:STDERR:        ╭─┤ fail_todo_aggregate_from_struct.carbon:[[@LINE+13]]:26
  // CHECK:STDERR: ->     │   let _: Cpp.Aggregate = {.x = 1, .y = 2};
  // CHECK:STDERR:        ·                          ┳━━━━━━━━━━━━━━━
  // CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━┛
  // CHECK:STDERR:        · ┃ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        · ┃        `{.x: Core.IntLiteral, .y: Core.IntLiteral}` to `Cpp.Aggregate`
  // CHECK:STDERR:        · ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                          ╰──┤ type `{.x: Core.IntLiteral, .y: Core.IntLiteral}`
  // CHECK:STDERR:        ·                             │ does not implement interface
  // CHECK:STDERR:        ·                             │ `Core.ImplicitAs(Cpp.Aggregate)`
  // CHECK:STDERR:        ·                             │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_non_aggregate_from_struct.carbon:[[@LINE+11]]:29
  // CHECK:STDERR: ->     │   let _: Cpp.NonAggregate = {};
  // CHECK:STDERR:        ·                             ┳━
  // CHECK:STDERR:        ·                             ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                             │  ┃        type `{}` to `Cpp.NonAggregate`
  // CHECK:STDERR:        ·                             │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                             ╰────┤ type `{}` does not implement interface
  // CHECK:STDERR:        ·                                  │ `Core.ImplicitAs(Cpp.NonAggregate)`
  // CHECK:STDERR:        ·                                  │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.NonAggregate = {};
  // CHECK:STDERR:        ╭─┤ fail_non_aggregate_from_struct.carbon:[[@LINE+13]]:29
  // CHECK:STDERR: ->     │   let _: Cpp.NonAggregate = {.x = 1};
  // CHECK:STDERR:        ·                             ┳━━━━━━━
  // CHECK:STDERR:        ·                             ┗━━┫ error: cannot implicitly convert expression of
  // CHECK:STDERR:        ·                             │  ┃        type `{.x: Core.IntLiteral}` to
  // CHECK:STDERR:        ·                             │  ┃        `Cpp.NonAggregate`
  // CHECK:STDERR:        ·                             │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                             ╰────┤ type `{.x: Core.IntLiteral}` does not
  // CHECK:STDERR:        ·                                  │ implement interface
  // CHECK:STDERR:        ·                                  │ `Core.ImplicitAs(Cpp.NonAggregate)`
  // CHECK:STDERR:        ·                                  │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.NonAggregate = {.x = 1};
  // CHECK:STDERR:        ╭─┤ fail_non_aggregate_from_struct.carbon:[[@LINE+14]]:29
  // CHECK:STDERR: ->     │   let _: Cpp.NonAggregate = {.x = 1, .y = 2};
  // CHECK:STDERR:        ·                             ┳━━━━━━━━━━━━━━━
  // CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  // CHECK:STDERR:        · ┃ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        · ┃        `{.x: Core.IntLiteral, .y: Core.IntLiteral}` to `Cpp.NonAggregate`
  // CHECK:STDERR:        · ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                             ╰──┤ type
  // CHECK:STDERR:        ·                                │ `{.x: Core.IntLiteral, .y: Core.IntLiteral}`
  // CHECK:STDERR:        ·                                │ does not implement interface
  // CHECK:STDERR:        ·                                │ `Core.ImplicitAs(Cpp.NonAggregate)`
  // CHECK:STDERR:        ·                                │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.NonAggregate = {.x = 1, .y = 2};
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit_conversions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %_.patt.664: %pattern_type.b72 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.ca8: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.eb4c5e.1: type = fn_type @Dest__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.c8e433.1: %Dest__carbon_thunk.type.eb4c5e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%Dest.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ffa: %SubobjectDestroy.type = facet_value %Dest, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.771: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ffa) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bbf: %Destroy.type = facet_value %Dest, (%DestroyImpl.impl_witness.771, %custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %NonConstConversion.689: type = class_type @NonConstConversion.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.eb4c5e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.c8e433.2: %Dest__carbon_thunk.type.eb4c5e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.a68: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %_.patt.ff1: %pattern_type.a68 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.0db: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.802: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.Op.type: type = fn_type @Dest2.Op [concrete]
// CHECK:STDOUT:   %Dest2.Op: %Dest2.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.2: <witness> = custom_witness (%Dest2.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.377: %SubobjectDestroy.type = facet_value %Dest2, (%custom_witness.5760f4.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.db5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.377) [concrete]
// CHECK:STDOUT:   %Destroy.facet.04c: %Destroy.type = facet_value %Dest2, (%DestroyImpl.impl_witness.db5, %custom_witness.5760f4.2) [concrete]
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
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.a8fbc1.1: %Dest__carbon_thunk.type.eb4c5e.1 = fn_decl @Dest__carbon_thunk.1 [concrete = constants.%Dest__carbon_thunk.c8e433.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion.1 [concrete = constants.%NonConstConversion.689] {} {}
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.a8fbc1.2: %Dest__carbon_thunk.type.eb4c5e.2 = fn_decl @Dest__carbon_thunk.2 [concrete = constants.%Dest__carbon_thunk.c8e433.2] {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UserConversion(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc8_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc8_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.a8fbc1.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc8_21.2: init %Dest to %.loc8_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_21.3: init %Dest = converted %s.ref, %.loc8_21.2
// CHECK:STDOUT:   %.loc8_21.4: ref %Dest = temporary %.loc8_21.1, %.loc8_21.3
// CHECK:STDOUT:   %.loc8_21.5: %Dest = acquire_value %.loc8_21.4
// CHECK:STDOUT:   %.loc8_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest = wrapper_binding _, %.loc8_21.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_21.1: %Destroy.type = facet_value constants.%Dest, (constants.%DestroyImpl.impl_witness.771, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %.loc8_21.6: %Destroy.type = converted constants.%Dest, %Destroy.facet.loc8_21.1 [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %Destroy.facet.loc8_21.2: %Destroy.type = facet_value constants.%Dest, (constants.%DestroyImpl.impl_witness.771, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %.loc8_21.7: %Destroy.type = converted constants.%Dest, %Destroy.facet.loc8_21.2 [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversion.loc12(%s.param: ref %NonConstConversion.689) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: ref %NonConstConversion.689 = name_ref s, %s
// CHECK:STDOUT:   %.loc14_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc14_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.a8fbc1.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc14_21.2: init %Dest to %.loc14_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_21.3: init %Dest = converted %s.ref, %.loc14_21.2
// CHECK:STDOUT:   %.loc14_21.4: ref %Dest = temporary %.loc14_21.1, %.loc14_21.3
// CHECK:STDOUT:   %.loc14_21.5: %Dest = acquire_value %.loc14_21.4
// CHECK:STDOUT:   %.loc14_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest = wrapper_binding _, %.loc14_21.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_21.1: %Destroy.type = facet_value constants.%Dest, (constants.%DestroyImpl.impl_witness.771, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %.loc14_21.6: %Destroy.type = converted constants.%Dest, %Destroy.facet.loc14_21.1 [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %Destroy.facet.loc14_21.2: %Destroy.type = facet_value constants.%Dest, (constants.%DestroyImpl.impl_witness.771, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %.loc14_21.7: %Destroy.type = converted constants.%Dest, %Destroy.facet.loc14_21.2 [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_22.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc20_22.2: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc20_22.1: %ptr.0db = addr_of %.loc20_22.2
// CHECK:STDOUT:   %addr.loc20_22.2: %ptr.802 = addr_of %.loc20_22.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc20_22.1, %addr.loc20_22.2)
// CHECK:STDOUT:   %.loc20_22.3: init %Dest2 to %.loc20_22.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_22.4: init %Dest2 = converted %s.ref, %.loc20_22.3
// CHECK:STDOUT:   %.loc20_22.5: ref %Dest2 = temporary %.loc20_22.1, %.loc20_22.4
// CHECK:STDOUT:   %.loc20_22.6: %Dest2 = acquire_value %.loc20_22.5
// CHECK:STDOUT:   %.loc20_13: type = splice_block %Dest2.ref [concrete = constants.%Dest2] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest2 = wrapper_binding _, %.loc20_22.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.a68 = value_binding_pattern _ [concrete = constants.%_.patt.ff1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_22.1: %Destroy.type = facet_value constants.%Dest2, (constants.%DestroyImpl.impl_witness.db5, constants.%custom_witness.5760f4.2) [concrete = constants.%Destroy.facet.04c]
// CHECK:STDOUT:   %.loc20_22.7: %Destroy.type = converted constants.%Dest2, %Destroy.facet.loc20_22.1 [concrete = constants.%Destroy.facet.04c]
// CHECK:STDOUT:   %Destroy.facet.loc20_22.2: %Destroy.type = facet_value constants.%Dest2, (constants.%DestroyImpl.impl_witness.db5, constants.%custom_witness.5760f4.2) [concrete = constants.%Destroy.facet.04c]
// CHECK:STDOUT:   %.loc20_22.8: %Destroy.type = converted constants.%Dest2, %Destroy.facet.loc20_22.2 [concrete = constants.%Destroy.facet.04c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_22.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonConstConversion: type = class_type @NonConstConversion [concrete]
// CHECK:STDOUT:   %Dest.980: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest.980 [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonConstConversion = %NonConstConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion [concrete = constants.%NonConstConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.980] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversionTest(%s.param: %NonConstConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %NonConstConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc19_21: %Dest.980 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19_13: type = splice_block %Dest.ref [concrete = constants.%Dest.980] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.980]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest.980 = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor: type = class_type @InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.824: type = pattern_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %_.patt.f1f: %pattern_type.824 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.5f6: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.859: type = ptr_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.type: type = fn_type @InaccessibleConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk: %InaccessibleConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %InaccessibleConstructor.Op.type: type = fn_type @InaccessibleConstructor.Op [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.Op: %InaccessibleConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%InaccessibleConstructor.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6bb: %SubobjectDestroy.type = facet_value %InaccessibleConstructor, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8e4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6bb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.8e3: %Destroy.type = facet_value %InaccessibleConstructor, (%DestroyImpl.impl_witness.8e4, %custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %InaccessibleConversion: type = class_type @InaccessibleConversion [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %_.patt.664: %pattern_type.b72 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.ca8: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.2: <witness> = custom_witness (%Dest.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ffa: %SubobjectDestroy.type = facet_value %Dest, (%custom_witness.5760f4.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.771: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ffa) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bbf: %Destroy.type = facet_value %Dest, (%DestroyImpl.impl_witness.771, %custom_witness.5760f4.2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.decl: type = class_decl @InaccessibleConversion [concrete = constants.%InaccessibleConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23_40.1: ref %InaccessibleConstructor = temporary_storage
// CHECK:STDOUT:   %.loc23_40.2: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc23_40.1: %ptr.5f6 = addr_of %.loc23_40.2
// CHECK:STDOUT:   %addr.loc23_40.2: %ptr.859 = addr_of %.loc23_40.1
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InaccessibleConstructor__carbon_thunk.decl(%addr.loc23_40.1, %addr.loc23_40.2)
// CHECK:STDOUT:   %.loc23_40.3: init %InaccessibleConstructor to %.loc23_40.1 = mark_in_place_init %InaccessibleConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc23_40.4: init %InaccessibleConstructor = converted %s.ref, %.loc23_40.3
// CHECK:STDOUT:   %.loc23_40.5: ref %InaccessibleConstructor = temporary %.loc23_40.1, %.loc23_40.4
// CHECK:STDOUT:   %.loc23_40.6: %InaccessibleConstructor = acquire_value %.loc23_40.5
// CHECK:STDOUT:   %.loc23_13: type = splice_block %InaccessibleConstructor.ref [concrete = constants.%InaccessibleConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InaccessibleConstructor.ref: type = name_ref InaccessibleConstructor, imports.%InaccessibleConstructor.decl [concrete = constants.%InaccessibleConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %InaccessibleConstructor = wrapper_binding _, %.loc23_40.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.824 = value_binding_pattern _ [concrete = constants.%_.patt.f1f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc23_40.1: %Destroy.type = facet_value constants.%InaccessibleConstructor, (constants.%DestroyImpl.impl_witness.8e4, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.8e3]
// CHECK:STDOUT:   %.loc23_40.7: %Destroy.type = converted constants.%InaccessibleConstructor, %Destroy.facet.loc23_40.1 [concrete = constants.%Destroy.facet.8e3]
// CHECK:STDOUT:   %Destroy.facet.loc23_40.2: %Destroy.type = facet_value constants.%InaccessibleConstructor, (constants.%DestroyImpl.impl_witness.8e4, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.8e3]
// CHECK:STDOUT:   %.loc23_40.8: %Destroy.type = converted constants.%InaccessibleConstructor, %Destroy.facet.loc23_40.2 [concrete = constants.%Destroy.facet.8e3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc23_40.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConversionTest(%s.param: %InaccessibleConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %InaccessibleConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc42_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc42_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc42_21.2: init %Dest to %.loc42_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc42_21.3: init %Dest = converted %s.ref, %.loc42_21.2
// CHECK:STDOUT:   %.loc42_21.4: ref %Dest = temporary %.loc42_21.1, %.loc42_21.3
// CHECK:STDOUT:   %.loc42_21.5: %Dest = acquire_value %.loc42_21.4
// CHECK:STDOUT:   %.loc42_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest = wrapper_binding _, %.loc42_21.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc42_21.1: %Destroy.type = facet_value constants.%Dest, (constants.%DestroyImpl.impl_witness.771, constants.%custom_witness.5760f4.2) [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %.loc42_21.6: %Destroy.type = converted constants.%Dest, %Destroy.facet.loc42_21.1 [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %Destroy.facet.loc42_21.2: %Destroy.type = facet_value constants.%Dest, (constants.%DestroyImpl.impl_witness.771, constants.%custom_witness.5760f4.2) [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   %.loc42_21.7: %Destroy.type = converted constants.%Dest, %Destroy.facet.loc42_21.2 [concrete = constants.%Destroy.facet.bbf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc42_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deleted.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %DeletedConstructor: type = class_type @DeletedConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.937: type = pattern_type %DeletedConstructor [concrete]
// CHECK:STDOUT:   %_.patt.a57: %pattern_type.937 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %DeletedConversion: type = class_type @DeletedConversion [concrete]
// CHECK:STDOUT:   %Dest.980: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest.980 [concrete]
// CHECK:STDOUT:   %_.patt.664: %pattern_type.b72 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.980] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc21_35: %DeletedConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_13: type = splice_block %DeletedConstructor.ref [concrete = constants.%DeletedConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeletedConstructor.ref: type = name_ref DeletedConstructor, imports.%DeletedConstructor.decl [concrete = constants.%DeletedConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %DeletedConstructor = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.937 = value_binding_pattern _ [concrete = constants.%_.patt.a57]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConversionTest(%s.param: %DeletedConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %DeletedConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc38_21: %Dest.980 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc38_13: type = splice_block %Dest.ref [concrete = constants.%Dest.980] {
// CHECK:STDOUT:     %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.980]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest.980 = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.eae: type = pattern_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %_.patt.411: %pattern_type.eae = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %Dest.980: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest.980 [concrete]
// CHECK:STDOUT:   %_.patt.664: %pattern_type.b72 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.980] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc21_36: %ExplicitConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_13: type = splice_block %ExplicitConstructor.ref [concrete = constants.%ExplicitConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %ExplicitConstructor = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.eae = value_binding_pattern _ [concrete = constants.%_.patt.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc38_21: %Dest.980 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc38_13: type = splice_block %Dest.ref [concrete = constants.%Dest.980] {
// CHECK:STDOUT:     %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.980]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest.980 = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- i32_to_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor.372: type = class_type @IntConstructor.1 [concrete]
// CHECK:STDOUT:   %pattern_type.c4e: type = pattern_type %IntConstructor.372 [concrete]
// CHECK:STDOUT:   %_.patt.151: %pattern_type.c4e = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.100: type = ptr_type %IntConstructor.372 [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.type: type = fn_type @IntConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk: %IntConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %IntConstructor.Op.type: type = fn_type @IntConstructor.Op [concrete]
// CHECK:STDOUT:   %IntConstructor.Op: %IntConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%IntConstructor.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b3e: %SubobjectDestroy.type = facet_value %IntConstructor.372, (%custom_witness.576) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b3e) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %IntConstructor.372, (%DestroyImpl.impl_witness.1d9, %custom_witness.576) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .IntConstructor = %IntConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.decl: type = class_decl @IntConstructor.1 [concrete = constants.%IntConstructor.372] {} {}
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.decl: %IntConstructor__carbon_thunk.type = fn_decl @IntConstructor__carbon_thunk [concrete = constants.%IntConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructor.loc6(%i.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc8_31.1: ref %IntConstructor.372 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.100 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(%i.ref, %addr)
// CHECK:STDOUT:   %.loc8_31.2: init %IntConstructor.372 to %.loc8_31.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_31.3: init %IntConstructor.372 = converted %i.ref, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: ref %IntConstructor.372 = temporary %.loc8_31.1, %.loc8_31.3
// CHECK:STDOUT:   %.loc8_31.5: %IntConstructor.372 = acquire_value %.loc8_31.4
// CHECK:STDOUT:   %.loc8_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor.372] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor.372]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %IntConstructor.372 = wrapper_binding _, %.loc8_31.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.c4e = value_binding_pattern _ [concrete = constants.%_.patt.151]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_31.1: %Destroy.type = facet_value constants.%IntConstructor.372, (constants.%DestroyImpl.impl_witness.1d9, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_31.6: %Destroy.type = converted constants.%IntConstructor.372, %Destroy.facet.loc8_31.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_31.2: %Destroy.type = facet_value constants.%IntConstructor.372, (constants.%DestroyImpl.impl_witness.1d9, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_31.7: %Destroy.type = converted constants.%IntConstructor.372, %Destroy.facet.loc8_31.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_31.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_u32_to_int_conversion_in_carbon.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor: type = class_type @IntConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.c4e: type = pattern_type %IntConstructor [concrete]
// CHECK:STDOUT:   %_.patt.151: %pattern_type.c4e = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.100: type = ptr_type %IntConstructor [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.type: type = fn_type @IntConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk: %IntConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %IntConstructor.Op.type: type = fn_type @IntConstructor.Op [concrete]
// CHECK:STDOUT:   %IntConstructor.Op: %IntConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%IntConstructor.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b3e: %SubobjectDestroy.type = facet_value %IntConstructor, (%custom_witness.576) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b3e) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %IntConstructor, (%DestroyImpl.impl_witness.1d9, %custom_witness.576) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructorTest(%u.param: %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %u32 = name_ref u, %u
// CHECK:STDOUT:   %.loc24_31.1: ref %IntConstructor = temporary_storage
// CHECK:STDOUT:   %.loc24_31.2: %i32 = converted %u.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.100 = addr_of %.loc24_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(<error>, %addr)
// CHECK:STDOUT:   %.loc24_31.3: init %IntConstructor to %.loc24_31.1 = mark_in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc24_31.4: init %IntConstructor = converted %u.ref, %.loc24_31.3
// CHECK:STDOUT:   %.loc24_31.5: ref %IntConstructor = temporary %.loc24_31.1, %.loc24_31.4
// CHECK:STDOUT:   %.loc24_31.6: %IntConstructor = acquire_value %.loc24_31.5
// CHECK:STDOUT:   %.loc24_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %IntConstructor = wrapper_binding _, %.loc24_31.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.c4e = value_binding_pattern _ [concrete = constants.%_.patt.151]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc24_31.1: %Destroy.type = facet_value constants.%IntConstructor, (constants.%DestroyImpl.impl_witness.1d9, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc24_31.7: %Destroy.type = converted constants.%IntConstructor, %Destroy.facet.loc24_31.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc24_31.2: %Destroy.type = facet_value constants.%IntConstructor, (constants.%DestroyImpl.impl_witness.1d9, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc24_31.8: %Destroy.type = converted constants.%IntConstructor, %Destroy.facet.loc24_31.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc24_31.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultConstructor: type = class_type @DefaultConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.be4: type = pattern_type %DefaultConstructor [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.be4 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.c09: type = ptr_type %DefaultConstructor [concrete]
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.type: type = fn_type @DefaultConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk: %DefaultConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultConstructor.Op.type: type = fn_type @DefaultConstructor.Op [concrete]
// CHECK:STDOUT:   %DefaultConstructor.Op: %DefaultConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%DefaultConstructor.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.243: %SubobjectDestroy.type = facet_value %DefaultConstructor, (%custom_witness.576) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a71: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.243) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %DefaultConstructor, (%DestroyImpl.impl_witness.a71, %custom_witness.576) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DefaultConstructorTest() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_36.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_36.2: ref %DefaultConstructor = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.c09 = addr_of %.loc9_36.2
// CHECK:STDOUT:   %DefaultConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%DefaultConstructor__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc9_36.3: init %DefaultConstructor to %.loc9_36.2 = mark_in_place_init %DefaultConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc9_36.4: init %DefaultConstructor = converted %.loc9_36.1, %.loc9_36.3
// CHECK:STDOUT:   %.loc9_36.5: ref %DefaultConstructor = temporary %.loc9_36.2, %.loc9_36.4
// CHECK:STDOUT:   %.loc9_36.6: %DefaultConstructor = acquire_value %.loc9_36.5
// CHECK:STDOUT:   %.loc9_13: type = splice_block %DefaultConstructor.ref [concrete = constants.%DefaultConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultConstructor.ref: type = name_ref DefaultConstructor, imports.%DefaultConstructor.decl [concrete = constants.%DefaultConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %DefaultConstructor = wrapper_binding _, %.loc9_36.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.be4 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc9_36.1: %Destroy.type = facet_value constants.%DefaultConstructor, (constants.%DestroyImpl.impl_witness.a71, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_36.7: %Destroy.type = converted constants.%DefaultConstructor, %Destroy.facet.loc9_36.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_36.2: %Destroy.type = facet_value constants.%DefaultConstructor, (constants.%DestroyImpl.impl_witness.a71, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_36.8: %Destroy.type = converted constants.%DefaultConstructor, %Destroy.facet.loc9_36.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_36.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- construct_multi_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Two: type = class_type @Two [concrete]
// CHECK:STDOUT:   %pattern_type.586: type = pattern_type %Two [concrete]
// CHECK:STDOUT:   %_.patt.44e: %pattern_type.586 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.2ff: type = ptr_type %Two [concrete]
// CHECK:STDOUT:   %Two__carbon_thunk.type: type = fn_type @Two__carbon_thunk [concrete]
// CHECK:STDOUT:   %Two__carbon_thunk: %Two__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault: type = class_type @ThreeWithDefault [concrete]
// CHECK:STDOUT:   %pattern_type.248: type = pattern_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %_.patt.368: %pattern_type.248 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.3ae: type = ptr_type %ThreeWithDefault [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.0fdb2f.1: type = fn_type @ThreeWithDefault__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.d9686c.1: %ThreeWithDefault__carbon_thunk.type.0fdb2f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.type.0fdb2f.2: type = fn_type @ThreeWithDefault__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.d9686c.2: %ThreeWithDefault__carbon_thunk.type.0fdb2f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %ThreeWithDefault.Op.type: type = fn_type @ThreeWithDefault.Op [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.Op: %ThreeWithDefault.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%ThreeWithDefault.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.45f: %SubobjectDestroy.type = facet_value %ThreeWithDefault, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e7f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.45f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.2d5: %Destroy.type = facet_value %ThreeWithDefault, (%DestroyImpl.impl_witness.e7f, %custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %Two.Op.type: type = fn_type @Two.Op [concrete]
// CHECK:STDOUT:   %Two.Op: %Two.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.2: <witness> = custom_witness (%Two.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.485: %SubobjectDestroy.type = facet_value %Two, (%custom_witness.5760f4.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c1c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.485) [concrete]
// CHECK:STDOUT:   %Destroy.facet.cc0: %Destroy.type = facet_value %Two, (%DestroyImpl.impl_witness.c1c, %custom_witness.5760f4.2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ThreeWithDefault.decl: type = class_decl @ThreeWithDefault [concrete = constants.%ThreeWithDefault] {} {}
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.decl.3a914c.1: %ThreeWithDefault__carbon_thunk.type.0fdb2f.1 = fn_decl @ThreeWithDefault__carbon_thunk.1 [concrete = constants.%ThreeWithDefault__carbon_thunk.d9686c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.decl.3a914c.2: %ThreeWithDefault__carbon_thunk.type.0fdb2f.2 = fn_decl @ThreeWithDefault__carbon_thunk.2 [concrete = constants.%ThreeWithDefault__carbon_thunk.d9686c.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplicitConvert() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.f94 = tuple_literal (%int_1.loc8, %int_2.loc8) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc8_25.2: ref %Two = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_21: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_21.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc8_21: <specific function> = specific_function %impl.elem0.loc8_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_21.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_21 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_21: init %i32 = call %bound_method.loc8_21.2(%int_1.loc8) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_21 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_21.2: %i32 = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc8_24: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_24.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc8_24: <specific function> = specific_function %impl.elem0.loc8_24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_24.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_24 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_24: init %i32 = call %bound_method.loc8_24.2(%int_2.loc8) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc8_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_24 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc8_24.2: %i32 = converted %int_2.loc8, %.loc8_24.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %addr.loc8: %ptr.2ff = addr_of %.loc8_25.2
// CHECK:STDOUT:   %Two__carbon_thunk.call: init %empty_tuple.type = call imports.%Two__carbon_thunk.decl(%.loc8_21.2, %.loc8_24.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_25.3: init %Two to %.loc8_25.2 = mark_in_place_init %Two__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_25.4: init %Two = converted %.loc8_25.1, %.loc8_25.3
// CHECK:STDOUT:   %.loc8_25.5: ref %Two = temporary %.loc8_25.2, %.loc8_25.4
// CHECK:STDOUT:   %.loc8_25.6: %Two = acquire_value %.loc8_25.5
// CHECK:STDOUT:   %.loc8_13: type = splice_block %Two.ref [concrete = constants.%Two] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Two.ref: type = name_ref Two, imports.%Two.decl [concrete = constants.%Two]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc8: %Two = wrapper_binding _, %.loc8_25.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc8: %pattern_type.586 = value_binding_pattern _ [concrete = constants.%_.patt.44e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_38.1: %tuple.type.f94 = tuple_literal (%int_1.loc10, %int_2.loc10) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc10_38.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc10_34: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_34 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_34: init %i32 = call %bound_method.loc10_34.2(%int_1.loc10) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_34 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_1.loc10, %.loc10_34.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc10_37: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_37.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc10_37: <specific function> = specific_function %impl.elem0.loc10_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_37.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_37 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_37: init %i32 = call %bound_method.loc10_37.2(%int_2.loc10) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_37 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_37.2: %i32 = converted %int_2.loc10, %.loc10_37.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %addr.loc10: %ptr.3ae = addr_of %.loc10_38.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.3a914c.1(%.loc10_34.2, %.loc10_37.2, %addr.loc10)
// CHECK:STDOUT:   %.loc10_38.3: init %ThreeWithDefault to %.loc10_38.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_38.4: init %ThreeWithDefault = converted %.loc10_38.1, %.loc10_38.3
// CHECK:STDOUT:   %.loc10_38.5: ref %ThreeWithDefault = temporary %.loc10_38.2, %.loc10_38.4
// CHECK:STDOUT:   %.loc10_38.6: %ThreeWithDefault = acquire_value %.loc10_38.5
// CHECK:STDOUT:   %.loc10_13: type = splice_block %ThreeWithDefault.ref.loc10 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc10: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc10: %ThreeWithDefault = wrapper_binding _, %.loc10_38.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc10: %pattern_type.248 = value_binding_pattern _ [concrete = constants.%_.patt.368]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc12_41.1: %tuple.type.37f = tuple_literal (%int_1.loc12, %int_2.loc12, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %.loc12_41.2: ref %ThreeWithDefault = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc12_34: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_34 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_34: init %i32 = call %bound_method.loc12_34.2(%int_1.loc12) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_34 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_1.loc12, %.loc12_34.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc12_37: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_37.1: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc12_37: <specific function> = specific_function %impl.elem0.loc12_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_37.2: <bound method> = bound_method %int_2.loc12, %specific_fn.loc12_37 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_37: init %i32 = call %bound_method.loc12_37.2(%int_2.loc12) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc12_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_37 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc12_37.2: %i32 = converted %int_2.loc12, %.loc12_37.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc12_40: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_40.1: <bound method> = bound_method %int_3, %impl.elem0.loc12_40 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc12_40: <specific function> = specific_function %impl.elem0.loc12_40, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.2: <bound method> = bound_method %int_3, %specific_fn.loc12_40 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_40: init %i32 = call %bound_method.loc12_40.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc12_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_40 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc12_40.2: %i32 = converted %int_3, %.loc12_40.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %addr.loc12: %ptr.3ae = addr_of %.loc12_41.2
// CHECK:STDOUT:   %ThreeWithDefault__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%ThreeWithDefault__carbon_thunk.decl.3a914c.2(%.loc12_34.2, %.loc12_37.2, %.loc12_40.2, %addr.loc12)
// CHECK:STDOUT:   %.loc12_41.3: init %ThreeWithDefault to %.loc12_41.2 = mark_in_place_init %ThreeWithDefault__carbon_thunk.call.loc12
// CHECK:STDOUT:   %.loc12_41.4: init %ThreeWithDefault = converted %.loc12_41.1, %.loc12_41.3
// CHECK:STDOUT:   %.loc12_41.5: ref %ThreeWithDefault = temporary %.loc12_41.2, %.loc12_41.4
// CHECK:STDOUT:   %.loc12_41.6: %ThreeWithDefault = acquire_value %.loc12_41.5
// CHECK:STDOUT:   %.loc12_13: type = splice_block %ThreeWithDefault.ref.loc12 [concrete = constants.%ThreeWithDefault] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ThreeWithDefault.ref.loc12: type = name_ref ThreeWithDefault, imports.%ThreeWithDefault.decl [concrete = constants.%ThreeWithDefault]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc12: %ThreeWithDefault = wrapper_binding _, %.loc12_41.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.248 = value_binding_pattern _ [concrete = constants.%_.patt.368]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_41.1: %Destroy.type = facet_value constants.%ThreeWithDefault, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   %.loc12_41.7: %Destroy.type = converted constants.%ThreeWithDefault, %Destroy.facet.loc12_41.1 [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   %Destroy.facet.loc12_41.2: %Destroy.type = facet_value constants.%ThreeWithDefault, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   %.loc12_41.8: %Destroy.type = converted constants.%ThreeWithDefault, %Destroy.facet.loc12_41.2 [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc12_41.5)
// CHECK:STDOUT:   %Destroy.facet.loc10_38.1: %Destroy.type = facet_value constants.%ThreeWithDefault, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   %.loc10_38.7: %Destroy.type = converted constants.%ThreeWithDefault, %Destroy.facet.loc10_38.1 [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   %Destroy.facet.loc10_38.2: %Destroy.type = facet_value constants.%ThreeWithDefault, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.5760f4.1) [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   %.loc10_38.8: %Destroy.type = converted constants.%ThreeWithDefault, %Destroy.facet.loc10_38.2 [concrete = constants.%Destroy.facet.2d5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_38.5)
// CHECK:STDOUT:   %Destroy.facet.loc8_25.1: %Destroy.type = facet_value constants.%Two, (constants.%DestroyImpl.impl_witness.c1c, constants.%custom_witness.5760f4.2) [concrete = constants.%Destroy.facet.cc0]
// CHECK:STDOUT:   %.loc8_25.7: %Destroy.type = converted constants.%Two, %Destroy.facet.loc8_25.1 [concrete = constants.%Destroy.facet.cc0]
// CHECK:STDOUT:   %Destroy.facet.loc8_25.2: %Destroy.type = facet_value constants.%Two, (constants.%DestroyImpl.impl_witness.c1c, constants.%custom_witness.5760f4.2) [concrete = constants.%Destroy.facet.cc0]
// CHECK:STDOUT:   %.loc8_25.8: %Destroy.type = converted constants.%Two, %Destroy.facet.loc8_25.2 [concrete = constants.%Destroy.facet.cc0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc8_25.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_aggregate_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Aggregate: type = class_type @Aggregate [concrete]
// CHECK:STDOUT:   %pattern_type.5dc: type = pattern_type %Aggregate [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.5dc = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %.loc19_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_27.2: %Aggregate = converted %.loc19_27.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19_13: type = splice_block %Aggregate.ref.loc19 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc19: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc19: %Aggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc19: %pattern_type.5dc = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc32_29.1: %tuple.type.985 = tuple_literal (%int_1.loc32) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc32_29.2: %Aggregate = converted %.loc32_29.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc32_13: type = splice_block %Aggregate.ref.loc32 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc32: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc32: %Aggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc32: %pattern_type.5dc = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc46: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc46_31.1: %tuple.type.f94 = tuple_literal (%int_1.loc46, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc46_31.2: %Aggregate = converted %.loc46_31.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc46_13: type = splice_block %Aggregate.ref.loc46 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc46: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc46: %Aggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc46: %pattern_type.5dc = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_aggregate_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Aggregate: type = class_type @Aggregate [concrete]
// CHECK:STDOUT:   %pattern_type.5dc: type = pattern_type %Aggregate [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.5dc = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %.loc19_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_27.2: %Aggregate = converted %.loc19_27.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19_13: type = splice_block %Aggregate.ref.loc19 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc19: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc19: %Aggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc19: %pattern_type.5dc = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc32_33.1: %struct_type.x = struct_literal (%int_1.loc32) [concrete = constants.%struct.147]
// CHECK:STDOUT:   %.loc32_33.2: %Aggregate = converted %.loc32_33.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc32_13: type = splice_block %Aggregate.ref.loc32 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc32: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc32: %Aggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc32: %pattern_type.5dc = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc46: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc46_41.1: %struct_type.x.y = struct_literal (%int_1.loc46, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc46_41.2: %Aggregate = converted %.loc46_41.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc46_13: type = splice_block %Aggregate.ref.loc46 [concrete = constants.%Aggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Aggregate.ref.loc46: type = name_ref Aggregate, imports.%Aggregate.decl [concrete = constants.%Aggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc46: %Aggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc46: %pattern_type.5dc = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_aggregate_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonAggregate: type = class_type @NonAggregate [concrete]
// CHECK:STDOUT:   %pattern_type.273: type = pattern_type %NonAggregate [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.273 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr.f81: type = ptr_type %NonAggregate [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.841a40.1: type = fn_type @NonAggregate__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.24a208.1: %NonAggregate__carbon_thunk.type.841a40.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.841a40.2: type = fn_type @NonAggregate__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.24a208.2: %NonAggregate__carbon_thunk.type.841a40.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.type.841a40.3: type = fn_type @NonAggregate__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.24a208.3: %NonAggregate__carbon_thunk.type.841a40.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %NonAggregate.Op.type: type = fn_type @NonAggregate.Op [concrete]
// CHECK:STDOUT:   %NonAggregate.Op: %NonAggregate.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%NonAggregate.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6e1: %SubobjectDestroy.type = facet_value %NonAggregate, (%custom_witness.576) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c3e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6e1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NonAggregate, (%DestroyImpl.impl_witness.c3e, %custom_witness.576) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonAggregate = %NonAggregate.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate.decl: type = class_decl @NonAggregate [concrete = constants.%NonAggregate] {} {}
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.356ec9.1: %NonAggregate__carbon_thunk.type.841a40.1 = fn_decl @NonAggregate__carbon_thunk.1 [concrete = constants.%NonAggregate__carbon_thunk.24a208.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.356ec9.2: %NonAggregate__carbon_thunk.type.841a40.2 = fn_decl @NonAggregate__carbon_thunk.2 [concrete = constants.%NonAggregate__carbon_thunk.24a208.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.decl.356ec9.3: %NonAggregate__carbon_thunk.type.841a40.3 = fn_decl @NonAggregate__carbon_thunk.3 [concrete = constants.%NonAggregate__carbon_thunk.24a208.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_30.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.f81 = addr_of %.loc8_30.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.356ec9.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_30.3: init %NonAggregate to %.loc8_30.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_30.4: init %NonAggregate = converted %.loc8_30.1, %.loc8_30.3
// CHECK:STDOUT:   %.loc8_30.5: ref %NonAggregate = temporary %.loc8_30.2, %.loc8_30.4
// CHECK:STDOUT:   %.loc8_30.6: %NonAggregate = acquire_value %.loc8_30.5
// CHECK:STDOUT:   %.loc8_13: type = splice_block %NonAggregate.ref.loc8 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc8: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc8: %NonAggregate = wrapper_binding _, %.loc8_30.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc8: %pattern_type.273 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_32.1: %tuple.type.985 = tuple_literal (%int_1.loc9) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %.loc9_32.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_30.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_30.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_30.2(%int_1.loc9) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_30.2: %i32 = converted %int_1.loc9, %.loc9_30.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %addr.loc9: %ptr.f81 = addr_of %.loc9_32.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.356ec9.2(%.loc9_30.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_32.3: init %NonAggregate to %.loc9_32.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_32.4: init %NonAggregate = converted %.loc9_32.1, %.loc9_32.3
// CHECK:STDOUT:   %.loc9_32.5: ref %NonAggregate = temporary %.loc9_32.2, %.loc9_32.4
// CHECK:STDOUT:   %.loc9_32.6: %NonAggregate = acquire_value %.loc9_32.5
// CHECK:STDOUT:   %.loc9_13: type = splice_block %NonAggregate.ref.loc9 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc9: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc9: %NonAggregate = wrapper_binding _, %.loc9_32.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc9: %pattern_type.273 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_34.1: %tuple.type.f94 = tuple_literal (%int_1.loc10, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc10_34.2: ref %NonAggregate = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10_30: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_30.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc10_30: <specific function> = specific_function %impl.elem0.loc10_30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_30.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_30 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_30: init %i32 = call %bound_method.loc10_30.2(%int_1.loc10) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_30 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_30.2: %i32 = converted %int_1.loc10, %.loc10_30.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc10_33: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_33.1: <bound method> = bound_method %int_2, %impl.elem0.loc10_33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc10_33: <specific function> = specific_function %impl.elem0.loc10_33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_33.2: <bound method> = bound_method %int_2, %specific_fn.loc10_33 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_33: init %i32 = call %bound_method.loc10_33.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_33 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_33.2: %i32 = converted %int_2, %.loc10_33.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %addr.loc10: %ptr.f81 = addr_of %.loc10_34.2
// CHECK:STDOUT:   %NonAggregate__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%NonAggregate__carbon_thunk.decl.356ec9.3(%.loc10_30.2, %.loc10_33.2, %addr.loc10)
// CHECK:STDOUT:   %.loc10_34.3: init %NonAggregate to %.loc10_34.2 = mark_in_place_init %NonAggregate__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_34.4: init %NonAggregate = converted %.loc10_34.1, %.loc10_34.3
// CHECK:STDOUT:   %.loc10_34.5: ref %NonAggregate = temporary %.loc10_34.2, %.loc10_34.4
// CHECK:STDOUT:   %.loc10_34.6: %NonAggregate = acquire_value %.loc10_34.5
// CHECK:STDOUT:   %.loc10_13: type = splice_block %NonAggregate.ref.loc10 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc10: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc10: %NonAggregate = wrapper_binding _, %.loc10_34.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc10: %pattern_type.273 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc10_34.1: %Destroy.type = facet_value constants.%NonAggregate, (constants.%DestroyImpl.impl_witness.c3e, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_34.7: %Destroy.type = converted constants.%NonAggregate, %Destroy.facet.loc10_34.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_34.2: %Destroy.type = facet_value constants.%NonAggregate, (constants.%DestroyImpl.impl_witness.c3e, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_34.8: %Destroy.type = converted constants.%NonAggregate, %Destroy.facet.loc10_34.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_34.5)
// CHECK:STDOUT:   %Destroy.facet.loc9_32.1: %Destroy.type = facet_value constants.%NonAggregate, (constants.%DestroyImpl.impl_witness.c3e, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_32.7: %Destroy.type = converted constants.%NonAggregate, %Destroy.facet.loc9_32.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_32.2: %Destroy.type = facet_value constants.%NonAggregate, (constants.%DestroyImpl.impl_witness.c3e, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_32.8: %Destroy.type = converted constants.%NonAggregate, %Destroy.facet.loc9_32.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_32.5)
// CHECK:STDOUT:   %Destroy.facet.loc8_30.1: %Destroy.type = facet_value constants.%NonAggregate, (constants.%DestroyImpl.impl_witness.c3e, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_30.7: %Destroy.type = converted constants.%NonAggregate, %Destroy.facet.loc8_30.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_30.2: %Destroy.type = facet_value constants.%NonAggregate, (constants.%DestroyImpl.impl_witness.c3e, constants.%custom_witness.576) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_30.8: %Destroy.type = converted constants.%NonAggregate, %Destroy.facet.loc8_30.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc8_30.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_aggregate_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonAggregate: type = class_type @NonAggregate [concrete]
// CHECK:STDOUT:   %pattern_type.273: type = pattern_type %NonAggregate [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.273 = value_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %.loc19_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_30.2: %NonAggregate = converted %.loc19_30.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19_13: type = splice_block %NonAggregate.ref.loc19 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc19: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc19: %NonAggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc19: %pattern_type.273 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc33: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc33_36.1: %struct_type.x = struct_literal (%int_1.loc33) [concrete = constants.%struct.147]
// CHECK:STDOUT:   %.loc33_36.2: %NonAggregate = converted %.loc33_36.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc33_13: type = splice_block %NonAggregate.ref.loc33 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc33: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc33: %NonAggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc33: %pattern_type.273 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc48: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc48_44.1: %struct_type.x.y = struct_literal (%int_1.loc48, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc48_44.2: %NonAggregate = converted %.loc48_44.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc48_13: type = splice_block %NonAggregate.ref.loc48 [concrete = constants.%NonAggregate] {
// CHECK:STDOUT:     %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonAggregate.ref.loc48: type = name_ref NonAggregate, imports.%NonAggregate.decl [concrete = constants.%NonAggregate]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc48: %NonAggregate = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc48: %pattern_type.273 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
