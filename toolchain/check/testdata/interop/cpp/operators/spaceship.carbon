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
// CHECK:STDOUT:   %EqWith.type.57515b.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.942b9d.1: %EqWith.type.57515b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.fe0e34.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.72b34c.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.942b9d.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.f43749.1: %EqWith.WithSelf.NotEqual.type.72b34c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.fc90e3.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.942b9d.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.1f678f.1: %EqWith.WithSelf.Equal.type.fc90e3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.57515b.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.154: type = pattern_type %EqWith.type.57515b.2 [symbolic]
// CHECK:STDOUT:   %T.942: %EqWith.type.57515b.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.fe0e34.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.2c2a05.2: %EqWith.assoc_type.fe0e34.2 = assoc_entity element0, imports.%Core.import_ref.600 [symbolic]
// CHECK:STDOUT:   %assoc1.58e750.2: %EqWith.assoc_type.fe0e34.2 = assoc_entity element1, imports.%Core.import_ref.11d [symbolic]
// CHECK:STDOUT:   %T.as_type.cae: type = facet_access_type %T.942 [symbolic]
// CHECK:STDOUT:   %pattern_type.23a293.2: type = pattern_type %T.as_type.cae [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.9a9: <witness> = require_complete_type %EqWith.type.57515b.2 [symbolic]
// CHECK:STDOUT:   %assoc0.89a: %EqWith.assoc_type.fe0e34.1 = assoc_entity element0, imports.%Core.import_ref.3f4 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.942, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.fc90e3.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T.942) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.72b34c.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T.942) [symbolic]
// CHECK:STDOUT:   %.587: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.fc90e3.3, %T.942 [symbolic]
// CHECK:STDOUT:   %impl.elem0.13e: %.587 = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a0b: <specific function> = specific_impl_function %impl.elem0.13e, @EqWith.WithSelf.Equal(%U, %T.942) [symbolic]
// CHECK:STDOUT:   %assoc1.7c8: %EqWith.assoc_type.fe0e34.1 = assoc_entity element1, imports.%Core.import_ref.48d [symbolic]
// CHECK:STDOUT:   %.a96: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.72b34c.3, %T.942 [symbolic]
// CHECK:STDOUT:   %impl.elem1.521: %.a96 = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cb7: <specific function> = specific_impl_function %impl.elem1.521, @EqWith.WithSelf.NotEqual(%U, %T.942) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c171ee.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.33ce9e.1: %OrderedWith.type.c171ee.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eba71.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.4b6b94.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c74212.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c73b41.1: %OrderedWith.WithSelf.Greater.type.c74212.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.1dd275.1: %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123104.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.cefdc0.1: %OrderedWith.WithSelf.Less.type.123104.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c171ee.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.e98: type = pattern_type %OrderedWith.type.c171ee.2 [symbolic]
// CHECK:STDOUT:   %T.33c: %OrderedWith.type.c171ee.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eba71.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.a764f9.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic]
// CHECK:STDOUT:   %assoc1.05b62d.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element1, imports.%Core.import_ref.6b8 [symbolic]
// CHECK:STDOUT:   %assoc2.19f303.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element2, imports.%Core.import_ref.f8b [symbolic]
// CHECK:STDOUT:   %assoc3.3683e0.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element3, imports.%Core.import_ref.095 [symbolic]
// CHECK:STDOUT:   %T.as_type.381: type = facet_access_type %T.33c [symbolic]
// CHECK:STDOUT:   %pattern_type.0cd5d4.2: type = pattern_type %T.as_type.381 [symbolic]
// CHECK:STDOUT:   %require_complete.600: <witness> = require_complete_type %OrderedWith.type.c171ee.2 [symbolic]
// CHECK:STDOUT:   %assoc0.e6a: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element0, imports.%Core.import_ref.52f [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.33c, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123104.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c74212.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %.c4c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.123104.3, %T.33c [symbolic]
// CHECK:STDOUT:   %impl.elem0.f46: %.c4c = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5e3: <specific function> = specific_impl_function %impl.elem0.f46, @OrderedWith.WithSelf.Less(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %assoc1.224: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element1, imports.%Core.import_ref.f4c [symbolic]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3, %T.33c [symbolic]
// CHECK:STDOUT:   %impl.elem1.d07: %.3d5 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.27d: <specific function> = specific_impl_function %impl.elem1.d07, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %assoc2.2dc: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element2, imports.%Core.import_ref.f2c [symbolic]
// CHECK:STDOUT:   %.7e7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.c74212.3, %T.33c [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.7e7 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b70: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %assoc3.34a: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element3, imports.%Core.import_ref.209 [symbolic]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3, %T.33c [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.71b = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f8: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T.33c) [symbolic]
// CHECK:STDOUT:   %DefaultSpaceshipOnly: type = class_type @DefaultSpaceshipOnly [concrete]
// CHECK:STDOUT:   %pattern_type.5c6: type = pattern_type %DefaultSpaceshipOnly [concrete]
// CHECK:STDOUT:   %ReturnsStrongOrdering: type = class_type @ReturnsStrongOrdering [concrete]
// CHECK:STDOUT:   %pattern_type.75f: type = pattern_type %ReturnsStrongOrdering [concrete]
// CHECK:STDOUT:   %ReturnsPartialOrdering: type = class_type @ReturnsPartialOrdering [concrete]
// CHECK:STDOUT:   %pattern_type.4ad: type = pattern_type %ReturnsPartialOrdering [concrete]
// CHECK:STDOUT:   %CrossTypeLhs: type = class_type @CrossTypeLhs [concrete]
// CHECK:STDOUT:   %pattern_type.6b3: type = pattern_type %CrossTypeLhs [concrete]
// CHECK:STDOUT:   %CrossTypeRhs: type = class_type @CrossTypeRhs [concrete]
// CHECK:STDOUT:   %pattern_type.54f: type = pattern_type %CrossTypeRhs [concrete]
// CHECK:STDOUT:   %EqWith.type.f47: type = facet_type <@EqWith.1, @EqWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.7ed: type = pattern_type %EqWith.type.f47 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b92: type = facet_type <@OrderedWith.1, @OrderedWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.5ed: type = pattern_type %OrderedWith.type.b92 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.1df: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsStrongOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.16c: type = pattern_type %OrderedWith.type.1df [concrete]
// CHECK:STDOUT:   %OrderedWith.type.728: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsPartialOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.fc4: type = pattern_type %OrderedWith.type.728 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.66d: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeRhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.107: type = pattern_type %OrderedWith.type.66d [concrete]
// CHECK:STDOUT:   %OrderedWith.type.242: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeLhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.37e: type = pattern_type %OrderedWith.type.242 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.633: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.89a)]
// CHECK:STDOUT:   %Core.import_ref.5ef: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.7c8)]
// CHECK:STDOUT:   %Core.import_ref.11d: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.72b34c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.f43749.1)]
// CHECK:STDOUT:   %Core.import_ref.600: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.fc90e3.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.1f678f.1)]
// CHECK:STDOUT:   %Core.import_ref.3f4 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.48d = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.246: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.e6a)]
// CHECK:STDOUT:   %Core.import_ref.8a6: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.224)]
// CHECK:STDOUT:   %Core.import_ref.f62: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.2dc)]
// CHECK:STDOUT:   %Core.import_ref.9ce: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.34a)]
// CHECK:STDOUT:   %Core.import_ref.095: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.4b6b94.1)]
// CHECK:STDOUT:   %Core.import_ref.f8b: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.c74212.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c73b41.1)]
// CHECK:STDOUT:   %Core.import_ref.6b8: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.1dd275.1)]
// CHECK:STDOUT:   %Core.import_ref.9d2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.123104.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.cefdc0.1)]
// CHECK:STDOUT:   %Core.import_ref.52f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f4c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f2c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.209 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc4(%U.loc4_12.2: type, %T.loc4_22.2: @EqWith.loc4.%EqWith.type.loc4_38.1 (%EqWith.type.57515b.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %EqWith.type.loc4_38.1 [symbolic = %require_complete.loc6 (constants.%require_complete.9a9)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.fe0e34.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.2) = assoc_entity element0, imports.%Core.import_ref.600 [symbolic = %assoc0 (constants.%assoc0.2c2a05.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.fc90e3.3)]
// CHECK:STDOUT:   %.loc6_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc4_22.1 [symbolic = %.loc6_5.2 (constants.%.587)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_22.1, @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc6_5.2: @EqWith.loc4.%.loc6_5.2 (%.587) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.13e)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_5.2: <specific function> = specific_impl_function %impl.elem0.loc6_5.2, @EqWith.WithSelf.Equal(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.a0b)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.2) = assoc_entity element1, imports.%Core.import_ref.11d [symbolic = %assoc1 (constants.%assoc1.58e750.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.72b34c.3)]
// CHECK:STDOUT:   %.loc7_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc4_22.1 [symbolic = %.loc7_5.2 (constants.%.a96)]
// CHECK:STDOUT:   %impl.elem1.loc7_5.2: @EqWith.loc4.%.loc7_5.2 (%.a96) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.521)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_5.2: <specific function> = specific_impl_function %impl.elem1.loc7_5.2, @EqWith.WithSelf.NotEqual(%U.loc4_12.1, %T.loc4_22.1) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.cb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.cae), %y.param: @EqWith.loc4.%U.loc4_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc6: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.cae) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc6: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc6: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_38.1 (constants.%EqWith.type.57515b.2)]
// CHECK:STDOUT:     %.loc6_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.2) = specific_constant imports.%Core.import_ref.633, @EqWith.WithSelf(constants.%U, constants.%Self.942b9d.1) [symbolic = %assoc0 (constants.%assoc0.2c2a05.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.2) = name_ref Equal, %.loc6_5.1 [symbolic = %assoc0 (constants.%assoc0.2c2a05.2)]
// CHECK:STDOUT:     %impl.elem0.loc6_5.1: @EqWith.loc4.%.loc6_5.2 (%.587) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.13e)]
// CHECK:STDOUT:     %bound_method.loc6_5.1: <bound method> = bound_method %x.ref.loc6, %impl.elem0.loc6_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_5.1: <specific function> = specific_impl_function %impl.elem0.loc6_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T.942) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.a0b)]
// CHECK:STDOUT:     %bound_method.loc6_5.2: <bound method> = bound_method %x.ref.loc6, %specific_impl_fn.loc6_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc6_5.2(%x.ref.loc6, %y.ref.loc6)
// CHECK:STDOUT:     %x.ref.loc7: @EqWith.loc4.%T.as_type.loc4_44.1 (%T.as_type.cae) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc7: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc7: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_38.1 (constants.%EqWith.type.57515b.2)]
// CHECK:STDOUT:     %.loc7_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.2) = specific_constant imports.%Core.import_ref.5ef, @EqWith.WithSelf(constants.%U, constants.%Self.942b9d.1) [symbolic = %assoc1 (constants.%assoc1.58e750.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.fe0e34.2) = name_ref NotEqual, %.loc7_5.1 [symbolic = %assoc1 (constants.%assoc1.58e750.2)]
// CHECK:STDOUT:     %impl.elem1.loc7_5.1: @EqWith.loc4.%.loc7_5.2 (%.a96) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.521)]
// CHECK:STDOUT:     %bound_method.loc7_5.1: <bound method> = bound_method %x.ref.loc7, %impl.elem1.loc7_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_5.1: <specific function> = specific_impl_function %impl.elem1.loc7_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T.942) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.cb7)]
// CHECK:STDOUT:     %bound_method.loc7_5.2: <bound method> = bound_method %x.ref.loc7, %specific_impl_fn.loc7_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc7_5.2(%x.ref.loc7, %y.ref.loc7)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc11(%U.loc11_17.2: type, %T.loc11_27.2: @OrderedWith.loc11.%OrderedWith.type.loc11_48.1 (%OrderedWith.type.c171ee.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc11_48.1 [symbolic = %require_complete.loc13 (constants.%require_complete.600)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2eba71.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.123104.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc11_27.1 [symbolic = %.loc13_5.2 (constants.%.c4c)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @OrderedWith.loc11.%.loc13_5.2 (%.c4c) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.f46)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @OrderedWith.WithSelf.Less(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element1, imports.%Core.import_ref.6b8 [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc11_27.1 [symbolic = %.loc14_5.2 (constants.%.3d5)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @OrderedWith.loc11.%.loc14_5.2 (%.3d5) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.d07)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.27d)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element2, imports.%Core.import_ref.f8b [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.c74212.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc11_27.1 [symbolic = %.loc15_5.2 (constants.%.7e7)]
// CHECK:STDOUT:   %impl.elem2.loc15_5.2: @OrderedWith.loc11.%.loc15_5.2 (%.7e7) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem2.loc15_5.2, @OrderedWith.WithSelf.Greater(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element3, imports.%Core.import_ref.095 [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc11_27.1 [symbolic = %.loc16_5.2 (constants.%.71b)]
// CHECK:STDOUT:   %impl.elem3.loc16_5.2: @OrderedWith.loc11.%.loc16_5.2 (%.71b) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem3.loc16_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc11_17.1, %T.loc11_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.381), %y.param: @OrderedWith.loc11.%U.loc11_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.381) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc13: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc13_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.246, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref Less, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @OrderedWith.loc11.%.loc13_5.2 (%.c4c) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.f46)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T.33c) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.381) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc14: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc14_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.8a6, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref LessOrEquivalent, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @OrderedWith.loc11.%.loc14_5.2 (%.3d5) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.d07)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T.33c) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.27d)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.381) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.f62, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref Greater, %.loc15_5.1 [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:     %impl.elem2.loc15_5.1: @OrderedWith.loc11.%.loc15_5.2 (%.7e7) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem2.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem2.loc15_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T.33c) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc11.%T.as_type.loc11_54.1 (%T.as_type.381) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.9ce, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref GreaterOrEquivalent, %.loc16_5.1 [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:     %impl.elem3.loc16_5.1: @OrderedWith.loc11.%.loc16_5.2 (%.71b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem3.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem3.loc16_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T.33c) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%U, constants.%T.942) {
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc4_38.1 => constants.%EqWith.type.57515b.2
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T.942
// CHECK:STDOUT:   %pattern_type.loc4_22 => constants.%pattern_type.154
// CHECK:STDOUT:   %T.as_type.loc4_44.1 => constants.%T.as_type.cae
// CHECK:STDOUT:   %pattern_type.loc4_42 => constants.%pattern_type.23a293.2
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%U, constants.%T.33c) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.c171ee.2
// CHECK:STDOUT:   %T.loc11_27.1 => constants.%T.33c
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.e98
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => constants.%T.as_type.381
// CHECK:STDOUT:   %pattern_type.loc11_52 => constants.%pattern_type.0cd5d4.2
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %EqWith.type.loc4_38.1 => constants.%EqWith.type.f47
// CHECK:STDOUT:   %T.loc4_22.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_22 => constants.%pattern_type.7ed
// CHECK:STDOUT:   %T.as_type.loc4_44.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_42 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.5c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.b92
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.5ed
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.5c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsStrongOrdering, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsStrongOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.1df
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.16c
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.75f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsPartialOrdering, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsPartialOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.728
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.fc4
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.4ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeRhs, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeRhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.66d
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.107
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.54f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeLhs, <error>) {
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeLhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_48.1 => constants.%OrderedWith.type.242
// CHECK:STDOUT:   %T.loc11_27.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_27 => constants.%pattern_type.37e
// CHECK:STDOUT:   %T.as_type.loc11_54.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_52 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_58 => constants.%pattern_type.6b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
