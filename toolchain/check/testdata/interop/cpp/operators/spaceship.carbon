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
// CHECK:STDOUT:   %EqWith.type.98ea86.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d94ae0.1: %EqWith.type.98ea86.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c5b45a.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.b4230d.1: %EqWith.WithSelf.NotEqual.type.c5b45a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.ae3c3e.1: %EqWith.WithSelf.Equal.type.b82ec7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.3d4: type = pattern_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %T.d94: %EqWith.type.98ea86.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.eee873.2: %EqWith.assoc_type.85b122.2 = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic]
// CHECK:STDOUT:   %assoc1.0ffc88.2: %EqWith.assoc_type.85b122.2 = assoc_entity element1, imports.%Core.import_ref.03f [symbolic]
// CHECK:STDOUT:   %T.as_type.93d: type = facet_access_type %T.d94 [symbolic]
// CHECK:STDOUT:   %pattern_type.e6374a.2: type = pattern_type %T.as_type.93d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.fb5: <witness> = require_complete_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d00: %EqWith.assoc_type.85b122.1 = assoc_entity element0, imports.%Core.import_ref.2e5 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.d94, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T.d94) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c5b45a.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T.d94) [symbolic]
// CHECK:STDOUT:   %.74c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b82ec7.3, %T.d94 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cab: %.74c = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.67a: <specific function> = specific_impl_function %impl.elem0.cab, @EqWith.WithSelf.Equal(%U, %T.d94) [symbolic]
// CHECK:STDOUT:   %assoc1.d80: %EqWith.assoc_type.85b122.1 = assoc_entity element1, imports.%Core.import_ref.b78 [symbolic]
// CHECK:STDOUT:   %.8ed: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.c5b45a.3, %T.d94 [symbolic]
// CHECK:STDOUT:   %impl.elem1.dab: %.8ed = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.81f: <specific function> = specific_impl_function %impl.elem1.dab, @EqWith.WithSelf.NotEqual(%U, %T.d94) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32ae87.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.7df555.1: %OrderedWith.type.32ae87.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2e6f.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.937fbb.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9ce25c.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c0b4fa.1: %OrderedWith.WithSelf.Greater.type.9ce25c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.e31723.1: %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87f4b.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.9e1f56.1: %OrderedWith.WithSelf.Less.type.b87f4b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32ae87.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.ebc: type = pattern_type %OrderedWith.type.32ae87.2 [symbolic]
// CHECK:STDOUT:   %T.7df: %OrderedWith.type.32ae87.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2e6f.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.c0023b.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic]
// CHECK:STDOUT:   %assoc1.3efd77.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element1, imports.%Core.import_ref.3f7 [symbolic]
// CHECK:STDOUT:   %assoc2.48eff0.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element2, imports.%Core.import_ref.385 [symbolic]
// CHECK:STDOUT:   %assoc3.5d53f7.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element3, imports.%Core.import_ref.39b [symbolic]
// CHECK:STDOUT:   %T.as_type.262: type = facet_access_type %T.7df [symbolic]
// CHECK:STDOUT:   %pattern_type.f6ec4f.2: type = pattern_type %T.as_type.262 [symbolic]
// CHECK:STDOUT:   %require_complete.3c0: <witness> = require_complete_type %OrderedWith.type.32ae87.2 [symbolic]
// CHECK:STDOUT:   %assoc0.13c: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element0, imports.%Core.import_ref.447 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.7df, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87f4b.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9ce25c.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %.6ec: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b87f4b.3, %T.7df [symbolic]
// CHECK:STDOUT:   %impl.elem0.6b1: %.6ec = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.527: <specific function> = specific_impl_function %impl.elem0.6b1, @OrderedWith.WithSelf.Less(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %assoc1.ee7: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element1, imports.%Core.import_ref.efc [symbolic]
// CHECK:STDOUT:   %.02d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3, %T.7df [symbolic]
// CHECK:STDOUT:   %impl.elem1.ff9: %.02d = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.001: <specific function> = specific_impl_function %impl.elem1.ff9, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %assoc2.131: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element2, imports.%Core.import_ref.e60 [symbolic]
// CHECK:STDOUT:   %.666: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9ce25c.3, %T.7df [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.666 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.735: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %assoc3.18a: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element3, imports.%Core.import_ref.6c6 [symbolic]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3, %T.7df [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.73b = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.418: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T.7df) [symbolic]
// CHECK:STDOUT:   %DefaultSpaceshipOnly: type = class_type @DefaultSpaceshipOnly [concrete]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %DefaultSpaceshipOnly [concrete]
// CHECK:STDOUT:   %ReturnsStrongOrdering: type = class_type @ReturnsStrongOrdering [concrete]
// CHECK:STDOUT:   %pattern_type.37f: type = pattern_type %ReturnsStrongOrdering [concrete]
// CHECK:STDOUT:   %ReturnsPartialOrdering: type = class_type @ReturnsPartialOrdering [concrete]
// CHECK:STDOUT:   %pattern_type.9ae: type = pattern_type %ReturnsPartialOrdering [concrete]
// CHECK:STDOUT:   %CrossTypeLhs: type = class_type @CrossTypeLhs [concrete]
// CHECK:STDOUT:   %pattern_type.eb5: type = pattern_type %CrossTypeLhs [concrete]
// CHECK:STDOUT:   %CrossTypeRhs: type = class_type @CrossTypeRhs [concrete]
// CHECK:STDOUT:   %pattern_type.aac: type = pattern_type %CrossTypeRhs [concrete]
// CHECK:STDOUT:   %EqWith.type.bf2: type = facet_type <@EqWith.1, @EqWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.062: type = pattern_type %EqWith.type.bf2 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.33f: type = facet_type <@OrderedWith.1, @OrderedWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.7f2: type = pattern_type %OrderedWith.type.33f [concrete]
// CHECK:STDOUT:   %OrderedWith.type.0fb: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsStrongOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.623: type = pattern_type %OrderedWith.type.0fb [concrete]
// CHECK:STDOUT:   %OrderedWith.type.5f1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsPartialOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.53d: type = pattern_type %OrderedWith.type.5f1 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.8cb: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeRhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.935: type = pattern_type %OrderedWith.type.8cb [concrete]
// CHECK:STDOUT:   %OrderedWith.type.c6c: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeLhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.b4e: type = pattern_type %OrderedWith.type.c6c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.725: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.d00)]
// CHECK:STDOUT:   %Core.import_ref.8d7: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.d80)]
// CHECK:STDOUT:   %Core.import_ref.03f: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.c5b45a.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.b4230d.1)]
// CHECK:STDOUT:   %Core.import_ref.5ba: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.b82ec7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.ae3c3e.1)]
// CHECK:STDOUT:   %Core.import_ref.2e5 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b78 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ef7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.13c)]
// CHECK:STDOUT:   %Core.import_ref.878: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.ee7)]
// CHECK:STDOUT:   %Core.import_ref.d4f: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.131)]
// CHECK:STDOUT:   %Core.import_ref.b87: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.18a)]
// CHECK:STDOUT:   %Core.import_ref.39b: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.937fbb.1)]
// CHECK:STDOUT:   %Core.import_ref.385: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.9ce25c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c0b4fa.1)]
// CHECK:STDOUT:   %Core.import_ref.3f7: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.e31723.1)]
// CHECK:STDOUT:   %Core.import_ref.e7a: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.b87f4b.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.9e1f56.1)]
// CHECK:STDOUT:   %Core.import_ref.447 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.efc = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.e60 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6c6 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc4(%U.loc4_12.2: type, %T.loc4_22.2: @EqWith.loc4.%EqWith.type.loc4_38.1 (%EqWith.type.98ea86.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %EqWith.type.loc4_38.1 [symbolic = %require_complete.loc6 (constants.%require_complete.fb5)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.85b122.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.b82ec7.3)]
// CHECK:STDOUT:   %.loc6_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc4_22.1 [symbolic = %.loc6_5.2 (constants.%.74c)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_22.1, @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc6_5.2: @EqWith.loc4.%.loc6_5.2 (%.74c) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.cab)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_5.2: <specific function> = specific_impl_function %impl.elem0.loc6_5.2, @EqWith.WithSelf.Equal(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.67a)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element1, imports.%Core.import_ref.03f [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.c5b45a.3)]
// CHECK:STDOUT:   %.loc7_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc4_22.1 [symbolic = %.loc7_5.2 (constants.%.8ed)]
// CHECK:STDOUT:   %impl.elem1.loc7_5.2: @EqWith.loc4.%.loc7_5.2 (%.8ed) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.dab)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_5.2: <specific function> = specific_impl_function %impl.elem1.loc7_5.2, @EqWith.WithSelf.NotEqual(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.81f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.93d), %y.param: @EqWith.loc4.%U.loc4_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc6: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.93d) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc6: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc6: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_38.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc6_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.725, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref Equal, %.loc6_5.1 [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %impl.elem0.loc6_5.1: @EqWith.loc4.%.loc6_5.2 (%.74c) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.cab)]
// CHECK:STDOUT:     %bound_method.loc6_5.1: <bound method> = bound_method %x.ref.loc6, %impl.elem0.loc6_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_5.1: <specific function> = specific_impl_function %impl.elem0.loc6_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T.d94) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.67a)]
// CHECK:STDOUT:     %bound_method.loc6_5.2: <bound method> = bound_method %x.ref.loc6, %specific_impl_fn.loc6_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc6_5.2(%x.ref.loc6, %y.ref.loc6)
// CHECK:STDOUT:     %x.ref.loc7: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.93d) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc7: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc7: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_38.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc7_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.8d7, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref NotEqual, %.loc7_5.1 [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:     %impl.elem1.loc7_5.1: @EqWith.loc4.%.loc7_5.2 (%.8ed) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.dab)]
// CHECK:STDOUT:     %bound_method.loc7_5.1: <bound method> = bound_method %x.ref.loc7, %impl.elem1.loc7_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_5.1: <specific function> = specific_impl_function %impl.elem1.loc7_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T.d94) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.81f)]
// CHECK:STDOUT:     %bound_method.loc7_5.2: <bound method> = bound_method %x.ref.loc7, %specific_impl_fn.loc7_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc7_5.2(%x.ref.loc7, %y.ref.loc7)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc11(%U.loc11_17.2: type, %T.loc11_27.2: @OrderedWith.loc11.%OrderedWith.type.loc11_48.1 (%OrderedWith.type.32ae87.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc11_48.1 [symbolic = %require_complete.loc13 (constants.%require_complete.3c0)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.1d2e6f.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.b87f4b.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc11_27.1 [symbolic = %.loc13_5.2 (constants.%.6ec)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @OrderedWith.loc11.%.loc13_5.2 (%.6ec) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.6b1)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @OrderedWith.WithSelf.Less(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.527)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element1, imports.%Core.import_ref.3f7 [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc11_27.1 [symbolic = %.loc14_5.2 (constants.%.02d)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @OrderedWith.loc11.%.loc14_5.2 (%.02d) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.ff9)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.001)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element2, imports.%Core.import_ref.385 [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.9ce25c.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc11_27.1 [symbolic = %.loc15_5.2 (constants.%.666)]
// CHECK:STDOUT:   %impl.elem2.loc15_5.2: @OrderedWith.loc11.%.loc15_5.2 (%.666) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem2.loc15_5.2, @OrderedWith.WithSelf.Greater(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.735)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element3, imports.%Core.import_ref.39b [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc11_27.1 [symbolic = %.loc16_5.2 (constants.%.73b)]
// CHECK:STDOUT:   %impl.elem3.loc16_5.2: @OrderedWith.loc11.%.loc16_5.2 (%.73b) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem3.loc16_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.262), %y.param: @OrderedWith.loc11.%U.loc11_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.262) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc13: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc13_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.ef7, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref Less, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @OrderedWith.loc11.%.loc13_5.2 (%.6ec) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.6b1)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T.7df) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.527)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.262) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc14: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc14_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.878, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref LessOrEquivalent, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @OrderedWith.loc11.%.loc14_5.2 (%.02d) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.ff9)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T.7df) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.001)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.262) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.d4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref Greater, %.loc15_5.1 [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:     %impl.elem2.loc15_5.1: @OrderedWith.loc11.%.loc15_5.2 (%.666) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem2.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem2.loc15_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T.7df) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.735)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.262) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.b87, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref GreaterOrEquivalent, %.loc16_5.1 [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:     %impl.elem3.loc16_5.1: @OrderedWith.loc11.%.loc16_5.2 (%.73b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem3.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem3.loc16_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T.7df) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%U, constants.%T.d94) {
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc4_38.1 => constants.%EqWith.type.98ea86.2
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T.d94
// CHECK:STDOUT:   %pattern_type.loc4_22 => constants.%pattern_type.3d4
// CHECK:STDOUT:   %T.as_type.loc4_44.1 => constants.%T.as_type.93d
// CHECK:STDOUT:   %pattern_type.loc4_42 => constants.%pattern_type.e6374a.2
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%U, constants.%T.7df) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.32ae87.2
// CHECK:STDOUT:   %T.loc11_27.1 => constants.%T.7df
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.ebc
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => constants.%T.as_type.262
// CHECK:STDOUT:   %pattern_type.loc11_52 => constants.%pattern_type.f6ec4f.2
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %EqWith.type.loc4_38.1 => constants.%EqWith.type.bf2
// CHECK:STDOUT:   %T.loc4_22.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_22 => constants.%pattern_type.062
// CHECK:STDOUT:   %T.as_type.loc4_44.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_42 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.33f
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.7f2
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsStrongOrdering, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsStrongOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.0fb
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.623
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsPartialOrdering, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsPartialOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.5f1
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.53d
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.9ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeRhs, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeRhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.8cb
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.935
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.aac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeLhs, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeLhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.c6c
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.b4e
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.eb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
