// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/spaceship.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/spaceship.carbon

// --- compare.h

namespace std {
  class partial_ordering {
  public:
    static const partial_ordering less;
    static const partial_ordering equivalent;
    static const partial_ordering greater;
    static const partial_ordering unordered;

    friend constexpr bool operator==(partial_ordering v, int) noexcept;
    friend constexpr bool operator==(partial_ordering v, partial_ordering w) noexcept = default;
    friend constexpr bool operator<(partial_ordering v, int) noexcept;
    friend constexpr bool operator>(partial_ordering v, int) noexcept;
    friend constexpr bool operator<=(partial_ordering v, int) noexcept;
    friend constexpr bool operator>=(partial_ordering v, int) noexcept;
    friend constexpr bool operator<(int, partial_ordering v) noexcept;
    friend constexpr bool operator>(int, partial_ordering v) noexcept;
    friend constexpr bool operator<=(int, partial_ordering v) noexcept;
    friend constexpr bool operator>=(int, partial_ordering v) noexcept;
    friend constexpr partial_ordering operator<=>(partial_ordering v, int) noexcept;
    friend constexpr partial_ordering operator<=>(int, partial_ordering v) noexcept;
  private:
    int value_;
    constexpr explicit partial_ordering(int x) noexcept : value_(x) {}
  };
  constexpr auto partial_ordering::less = partial_ordering(-1);
  constexpr auto partial_ordering::equivalent = partial_ordering(0);
  constexpr auto partial_ordering::greater = partial_ordering(1);
  constexpr auto partial_ordering::unordered = partial_ordering(2);

  class weak_ordering {
  public:
    static const weak_ordering less;
    static const weak_ordering equivalent;
    static const weak_ordering greater;

    friend constexpr bool operator==(weak_ordering v, int) noexcept;
    friend constexpr bool operator==(weak_ordering v, weak_ordering w) noexcept = default;
    friend constexpr bool operator<(weak_ordering v, int) noexcept;
    friend constexpr bool operator>(weak_ordering v, int) noexcept;
    friend constexpr bool operator<=(weak_ordering v, int) noexcept;
    friend constexpr bool operator>=(weak_ordering v, int) noexcept;
    friend constexpr bool operator<(int, weak_ordering v) noexcept;
    friend constexpr bool operator>(int, weak_ordering v) noexcept;
    friend constexpr bool operator<=(int, weak_ordering v) noexcept;
    friend constexpr bool operator>=(int, weak_ordering v) noexcept;
    friend constexpr weak_ordering operator<=>(weak_ordering v, int) noexcept;
    friend constexpr weak_ordering operator<=>(int, weak_ordering v) noexcept;
  private:
    int value_;
    constexpr explicit weak_ordering(int x) noexcept : value_(x) {}
  };
  constexpr auto weak_ordering::less = weak_ordering(-1);
  constexpr auto weak_ordering::equivalent = weak_ordering(0);
  constexpr auto weak_ordering::greater = weak_ordering(1);

  class strong_ordering {
  public:
    static const strong_ordering less;
    static const strong_ordering equal;
    static const strong_ordering equivalent;
    static const strong_ordering greater;

    friend constexpr bool operator==(strong_ordering v, int) noexcept;
    friend constexpr bool operator==(strong_ordering v, strong_ordering w) noexcept = default;
    friend constexpr bool operator<(strong_ordering v, int) noexcept;
    friend constexpr bool operator>(strong_ordering v, int) noexcept;
    friend constexpr bool operator<=(strong_ordering v, int) noexcept;
    friend constexpr bool operator>=(strong_ordering v, int) noexcept;
    friend constexpr bool operator<(int, strong_ordering v) noexcept;
    friend constexpr bool operator>(int, strong_ordering v) noexcept;
    friend constexpr bool operator<=(int, strong_ordering v) noexcept;
    friend constexpr bool operator>=(int, strong_ordering v) noexcept;
    friend constexpr strong_ordering operator<=>(strong_ordering v, int) noexcept;
    friend constexpr strong_ordering operator<=>(int, strong_ordering v) noexcept;
  private:
    int value_;
    constexpr explicit strong_ordering(int x) noexcept : value_(x) {}
  };
  constexpr auto strong_ordering::less = strong_ordering(-1);
  constexpr auto strong_ordering::equal = strong_ordering(0);
  constexpr auto strong_ordering::equivalent = strong_ordering(0);
  constexpr auto strong_ordering::greater = strong_ordering(1);
} // namespace std

// --- three_way_comparable_types.h

#include "compare.h"

struct DefaultSpaceshipOnly {
public:
  auto operator<=>(const DefaultSpaceshipOnly&) const = default;
};

struct ReturnsStrongOrdering {
  auto operator<=>(ReturnsStrongOrdering) const -> std::strong_ordering;
};

struct ReturnsWeakOrdering {
  auto operator<=>(ReturnsWeakOrdering) const -> std::weak_ordering;
};

struct ReturnsPartialOrdering {
  auto operator<=>(ReturnsPartialOrdering) const -> std::partial_ordering;
};

struct CrossTypeRhs;
struct CrossTypeLhs {
  auto operator<=>(CrossTypeRhs) const -> std::strong_ordering;
};
struct CrossTypeRhs {
  auto operator<=>(CrossTypeLhs) const -> std::strong_ordering;
};

// --- fail_todo_spaceship_rewrites.carbon

import Cpp library "three_way_comparable_types.h";

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x == y;
  x != y;
  //@dump-sem-ir-end
}

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x < y;
  x <= y;
  x > y;
  x >= y;
  //@dump-sem-ir-end
}

fn Test(default_spaceship_only: Cpp.DefaultSpaceshipOnly,
        returns_strong_ordering: Cpp.ReturnsStrongOrdering,
        returns_weak_ordering: Cpp.ReturnsWeakOrdering,
        returns_partial_ordering: Cpp.ReturnsPartialOrdering,
        cross_type_lhs: Cpp.CrossTypeLhs,
        cross_type_rhs: Cpp.CrossTypeRhs) {
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   EqWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-25]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(default_spaceship_only, default_spaceship_only);

  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-27]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_partial_ordering, returns_partial_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_partial_ordering, returns_partial_ordering);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(cross_type_lhs, cross_type_rhs);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-51]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(cross_type_lhs, cross_type_rhs);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+11]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(cross_type_rhs, cross_type_lhs);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-59]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-52]]:9: warning: binding `returns_weak_ordering` unused [UnusedBinding]
  // CHECK:STDERR:         returns_weak_ordering: Cpp.ReturnsWeakOrdering,
  // CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(cross_type_rhs, cross_type_lhs);
}

// --- fail_never_valid.carbon
// CHECK:STDERR: fail_never_valid.carbon: error: `Main//default` previously provided by `fail_todo_spaceship_rewrites.carbon` [DuplicateMainApi]
// CHECK:STDERR:

import Cpp library "three_way_comparable_types.h";

fn EqWith[U:! type, T:! Core.EqWith(U)](unused x: T, unused y: U) {}

fn Test(returns_strong_ordering: Cpp.ReturnsStrongOrdering) {
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.ReturnsStrongOrdering` into type implementing `Core.EqWith(Cpp.ReturnsStrongOrdering)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](unused x: T, unused y: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(returns_strong_ordering, returns_strong_ordering);
}

// CHECK:STDOUT: --- fail_todo_spaceship_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.e412df.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.1e4d80.1: %EqWith.type.e412df.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.27e9f5.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a34bef.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.1e4d80.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.34d150.1: %EqWith.WithSelf.NotEqual.type.a34bef.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.28b02d.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.1e4d80.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.6ee1cd.1: %EqWith.WithSelf.Equal.type.28b02d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.e412df.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.243: type = pattern_type %EqWith.type.e412df.2 [symbolic]
// CHECK:STDOUT:   %T.1e4: %EqWith.type.e412df.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.27e9f5.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.04df8c.2: %EqWith.assoc_type.27e9f5.2 = assoc_entity element0, imports.%Core.import_ref.eea [symbolic]
// CHECK:STDOUT:   %assoc1.47b841.2: %EqWith.assoc_type.27e9f5.2 = assoc_entity element1, imports.%Core.import_ref.228 [symbolic]
// CHECK:STDOUT:   %T.as_type.cf2: type = facet_access_type %T.1e4 [symbolic]
// CHECK:STDOUT:   %pattern_type.479039.2: type = pattern_type %T.as_type.cf2 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.1c9: <witness> = require_complete_type %EqWith.type.e412df.2 [symbolic]
// CHECK:STDOUT:   %assoc0.5b6: %EqWith.assoc_type.27e9f5.1 = assoc_entity element0, imports.%Core.import_ref.c77 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.1e4, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.28b02d.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T.1e4) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a34bef.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T.1e4) [symbolic]
// CHECK:STDOUT:   %.e7e: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.28b02d.3, %T.1e4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c07: %.e7e = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ba9: <specific function> = specific_impl_function %impl.elem0.c07, @EqWith.WithSelf.Equal(%U, %T.1e4) [symbolic]
// CHECK:STDOUT:   %assoc1.491: %EqWith.assoc_type.27e9f5.1 = assoc_entity element1, imports.%Core.import_ref.c36 [symbolic]
// CHECK:STDOUT:   %.a13: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.a34bef.3, %T.1e4 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ec1: %.a13 = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a6f: <specific function> = specific_impl_function %impl.elem1.ec1, @EqWith.WithSelf.NotEqual(%U, %T.1e4) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.73dab8.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d014af.1: %OrderedWith.type.73dab8.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2fabd7.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.38d1a4.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9defd1.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.984598.1: %OrderedWith.WithSelf.Greater.type.9defd1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.c2ac22.1: %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.687390.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.4a995a.1: %OrderedWith.WithSelf.Less.type.687390.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.73dab8.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.10d: type = pattern_type %OrderedWith.type.73dab8.2 [symbolic]
// CHECK:STDOUT:   %T.d01: %OrderedWith.type.73dab8.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2fabd7.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.7dbd55.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element0, imports.%Core.import_ref.5cd [symbolic]
// CHECK:STDOUT:   %assoc1.54de85.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element1, imports.%Core.import_ref.fb6 [symbolic]
// CHECK:STDOUT:   %assoc2.3caaa9.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element2, imports.%Core.import_ref.32f [symbolic]
// CHECK:STDOUT:   %assoc3.4eb879.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element3, imports.%Core.import_ref.2d8 [symbolic]
// CHECK:STDOUT:   %T.as_type.1cd: type = facet_access_type %T.d01 [symbolic]
// CHECK:STDOUT:   %pattern_type.05039a.2: type = pattern_type %T.as_type.1cd [symbolic]
// CHECK:STDOUT:   %require_complete.a45: <witness> = require_complete_type %OrderedWith.type.73dab8.2 [symbolic]
// CHECK:STDOUT:   %assoc0.3ff: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element0, imports.%Core.import_ref.c33 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.d01, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.687390.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9defd1.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.687390.3, %T.d01 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b31: %.cb4 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c89: <specific function> = specific_impl_function %impl.elem0.b31, @OrderedWith.WithSelf.Less(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %assoc1.760: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element1, imports.%Core.import_ref.79a [symbolic]
// CHECK:STDOUT:   %.ac4: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3, %T.d01 [symbolic]
// CHECK:STDOUT:   %impl.elem1.939: %.ac4 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.65b: <specific function> = specific_impl_function %impl.elem1.939, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %assoc2.897: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element2, imports.%Core.import_ref.ea2 [symbolic]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9defd1.3, %T.d01 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.994 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6dc: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %assoc3.8bc: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element3, imports.%Core.import_ref.b4f [symbolic]
// CHECK:STDOUT:   %.eca: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3, %T.d01 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.eca = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f67: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T.d01) [symbolic]
// CHECK:STDOUT:   %DefaultSpaceshipOnly: type = class_type @DefaultSpaceshipOnly [concrete]
// CHECK:STDOUT:   %pattern_type.cce: type = pattern_type %DefaultSpaceshipOnly [concrete]
// CHECK:STDOUT:   %ReturnsStrongOrdering: type = class_type @ReturnsStrongOrdering [concrete]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %ReturnsStrongOrdering [concrete]
// CHECK:STDOUT:   %ReturnsPartialOrdering: type = class_type @ReturnsPartialOrdering [concrete]
// CHECK:STDOUT:   %pattern_type.dbf: type = pattern_type %ReturnsPartialOrdering [concrete]
// CHECK:STDOUT:   %CrossTypeLhs: type = class_type @CrossTypeLhs [concrete]
// CHECK:STDOUT:   %pattern_type.f3d: type = pattern_type %CrossTypeLhs [concrete]
// CHECK:STDOUT:   %CrossTypeRhs: type = class_type @CrossTypeRhs [concrete]
// CHECK:STDOUT:   %pattern_type.bd1: type = pattern_type %CrossTypeRhs [concrete]
// CHECK:STDOUT:   %EqWith.type.f5d: type = facet_type <@EqWith.1, @EqWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %EqWith.type.f5d [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9ef: type = facet_type <@OrderedWith.1, @OrderedWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.520: type = pattern_type %OrderedWith.type.9ef [concrete]
// CHECK:STDOUT:   %OrderedWith.type.f1a: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsStrongOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.8f0: type = pattern_type %OrderedWith.type.f1a [concrete]
// CHECK:STDOUT:   %OrderedWith.type.30a: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsPartialOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.86c: type = pattern_type %OrderedWith.type.30a [concrete]
// CHECK:STDOUT:   %OrderedWith.type.ca0: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeRhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.189: type = pattern_type %OrderedWith.type.ca0 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.e89: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeLhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.307: type = pattern_type %OrderedWith.type.e89 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0a8: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.5b6)]
// CHECK:STDOUT:   %Core.import_ref.33b: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.491)]
// CHECK:STDOUT:   %Core.import_ref.228: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.a34bef.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.34d150.1)]
// CHECK:STDOUT:   %Core.import_ref.eea: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.28b02d.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.6ee1cd.1)]
// CHECK:STDOUT:   %Core.import_ref.c77 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c36 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.eb4: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.3ff)]
// CHECK:STDOUT:   %Core.import_ref.84c: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.760)]
// CHECK:STDOUT:   %Core.import_ref.01d: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.897)]
// CHECK:STDOUT:   %Core.import_ref.569: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.8bc)]
// CHECK:STDOUT:   %Core.import_ref.2d8: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.38d1a4.1)]
// CHECK:STDOUT:   %Core.import_ref.32f: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.9defd1.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.984598.1)]
// CHECK:STDOUT:   %Core.import_ref.fb6: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.c2ac22.1)]
// CHECK:STDOUT:   %Core.import_ref.5cd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.687390.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.4a995a.1)]
// CHECK:STDOUT:   %Core.import_ref.c33 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.79a = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ea2 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b4f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc4(%U.loc4_12.2: type, %T.loc4_22.2: @EqWith.loc4.%EqWith.type.loc4_38.1 (%EqWith.type.e412df.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %EqWith.type.loc4_38.1 [symbolic = %require_complete.loc6 (constants.%require_complete.1c9)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.27e9f5.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = assoc_entity element0, imports.%Core.import_ref.eea [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.28b02d.3)]
// CHECK:STDOUT:   %.loc6_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc4_22.1 [symbolic = %.loc6_5.2 (constants.%.e7e)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_22.1, @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc6_5.2: @EqWith.loc4.%.loc6_5.2 (%.e7e) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.c07)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_5.2: <specific function> = specific_impl_function %impl.elem0.loc6_5.2, @EqWith.WithSelf.Equal(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.ba9)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = assoc_entity element1, imports.%Core.import_ref.228 [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.a34bef.3)]
// CHECK:STDOUT:   %.loc7_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc4_22.1 [symbolic = %.loc7_5.2 (constants.%.a13)]
// CHECK:STDOUT:   %impl.elem1.loc7_5.2: @EqWith.loc4.%.loc7_5.2 (%.a13) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.ec1)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_5.2: <specific function> = specific_impl_function %impl.elem1.loc7_5.2, @EqWith.WithSelf.NotEqual(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.a6f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.cf2), %y.param: @EqWith.loc4.%U.loc4_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc6: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.cf2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc6: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc6: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_38.1 (constants.%EqWith.type.e412df.2)]
// CHECK:STDOUT:     %.loc6_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = specific_constant imports.%Core.import_ref.0a8, @EqWith.WithSelf(constants.%U, constants.%Self.1e4d80.1) [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = name_ref Equal, %.loc6_5.1 [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:     %impl.elem0.loc6_5.1: @EqWith.loc4.%.loc6_5.2 (%.e7e) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.c07)]
// CHECK:STDOUT:     %bound_method.loc6_5.1: <bound method> = bound_method %x.ref.loc6, %impl.elem0.loc6_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_5.1: <specific function> = specific_impl_function %impl.elem0.loc6_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T.1e4) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.ba9)]
// CHECK:STDOUT:     %bound_method.loc6_5.2: <bound method> = bound_method %x.ref.loc6, %specific_impl_fn.loc6_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc6_5.2(%x.ref.loc6, %y.ref.loc6)
// CHECK:STDOUT:     %x.ref.loc7: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.cf2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc7: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc7: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_38.1 (constants.%EqWith.type.e412df.2)]
// CHECK:STDOUT:     %.loc7_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = specific_constant imports.%Core.import_ref.33b, @EqWith.WithSelf(constants.%U, constants.%Self.1e4d80.1) [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = name_ref NotEqual, %.loc7_5.1 [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:     %impl.elem1.loc7_5.1: @EqWith.loc4.%.loc7_5.2 (%.a13) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.ec1)]
// CHECK:STDOUT:     %bound_method.loc7_5.1: <bound method> = bound_method %x.ref.loc7, %impl.elem1.loc7_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_5.1: <specific function> = specific_impl_function %impl.elem1.loc7_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T.1e4) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.a6f)]
// CHECK:STDOUT:     %bound_method.loc7_5.2: <bound method> = bound_method %x.ref.loc7, %specific_impl_fn.loc7_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc7_5.2(%x.ref.loc7, %y.ref.loc7)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc11(%U.loc11_17.2: type, %T.loc11_27.2: @OrderedWith.loc11.%OrderedWith.type.loc11_48.1 (%OrderedWith.type.73dab8.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc11_48.1 [symbolic = %require_complete.loc13 (constants.%require_complete.a45)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2fabd7.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element0, imports.%Core.import_ref.5cd [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.687390.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc11_27.1 [symbolic = %.loc13_5.2 (constants.%.cb4)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @OrderedWith.loc11.%.loc13_5.2 (%.cb4) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.b31)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @OrderedWith.WithSelf.Less(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.c89)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element1, imports.%Core.import_ref.fb6 [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc11_27.1 [symbolic = %.loc14_5.2 (constants.%.ac4)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @OrderedWith.loc11.%.loc14_5.2 (%.ac4) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.939)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.65b)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element2, imports.%Core.import_ref.32f [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.9defd1.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc11_27.1 [symbolic = %.loc15_5.2 (constants.%.994)]
// CHECK:STDOUT:   %impl.elem2.loc15_5.2: @OrderedWith.loc11.%.loc15_5.2 (%.994) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem2.loc15_5.2, @OrderedWith.WithSelf.Greater(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.6dc)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element3, imports.%Core.import_ref.2d8 [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc11_27.1 [symbolic = %.loc16_5.2 (constants.%.eca)]
// CHECK:STDOUT:   %impl.elem3.loc16_5.2: @OrderedWith.loc11.%.loc16_5.2 (%.eca) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem3.loc16_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.f67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.1cd), %y.param: @OrderedWith.loc11.%U.loc11_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.1cd) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc13: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc13_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.eb4, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref Less, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @OrderedWith.loc11.%.loc13_5.2 (%.cb4) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.b31)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T.d01) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.c89)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.1cd) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc14: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc14_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.84c, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref LessOrEquivalent, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @OrderedWith.loc11.%.loc14_5.2 (%.ac4) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.939)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T.d01) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.65b)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.1cd) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.01d, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref Greater, %.loc15_5.1 [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:     %impl.elem2.loc15_5.1: @OrderedWith.loc11.%.loc15_5.2 (%.994) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem2.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem2.loc15_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T.d01) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.6dc)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.1cd) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.569, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref GreaterOrEquivalent, %.loc16_5.1 [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:     %impl.elem3.loc16_5.1: @OrderedWith.loc11.%.loc16_5.2 (%.eca) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem3.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem3.loc16_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T.d01) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.f67)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%U, constants.%T.1e4) {
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc4_38.1 => constants.%EqWith.type.e412df.2
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T.1e4
// CHECK:STDOUT:   %pattern_type.loc4_22 => constants.%pattern_type.243
// CHECK:STDOUT:   %T.as_type.loc4_44.1 => constants.%T.as_type.cf2
// CHECK:STDOUT:   %pattern_type.loc4_42 => constants.%pattern_type.479039.2
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%U, constants.%T.d01) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.73dab8.2
// CHECK:STDOUT:   %T.loc11_27.1 => constants.%T.d01
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.10d
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => constants.%T.as_type.1cd
// CHECK:STDOUT:   %pattern_type.loc11_52 => constants.%pattern_type.05039a.2
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %EqWith.type.loc4_38.1 => constants.%EqWith.type.f5d
// CHECK:STDOUT:   %T.loc4_22.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_22 => constants.%pattern_type.2b5
// CHECK:STDOUT:   %T.as_type.loc4_44.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_42 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.cce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.9ef
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.520
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.cce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsStrongOrdering, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsStrongOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.f1a
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.8f0
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.a93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsPartialOrdering, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsPartialOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.30a
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.86c
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.dbf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeRhs, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeRhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.ca0
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.189
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.bd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeLhs, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeLhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.e89
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.307
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.f3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
