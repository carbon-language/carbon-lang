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

fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x == y;
  x != y;
  //@dump-sem-ir-end
}

fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
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
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(default_spaceship_only, default_spaceship_only);

  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-27]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_partial_ordering, returns_partial_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_partial_ordering, returns_partial_ordering);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(cross_type_lhs, cross_type_rhs);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-51]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(cross_type_lhs, cross_type_rhs);
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE+11]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(cross_type_rhs, cross_type_lhs);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_spaceship_rewrites.carbon:[[@LINE-59]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

fn EqWith[U: type, T: Core.EqWith(U)](unused x: T, unused y: U) {}

fn Test(returns_strong_ordering: Cpp.ReturnsStrongOrdering) {
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.ReturnsStrongOrdering` into type implementing `Core.EqWith(Cpp.ReturnsStrongOrdering)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](unused x: T, unused y: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(returns_strong_ordering, returns_strong_ordering);
}

// CHECK:STDOUT: --- fail_todo_spaceship_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.686248.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.bd18d5.1: %EqWith.type.686248.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.92e927.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0896a8.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.bd18d5.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.cb1497.1: %EqWith.WithSelf.NotEqual.type.0896a8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e5264.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.bd18d5.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.74d211.1: %EqWith.WithSelf.Equal.type.4e5264.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.686248.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.295: type = pattern_type %EqWith.type.686248.2 [symbolic]
// CHECK:STDOUT:   %T.patt.738: %pattern_type.295 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.bd1: %EqWith.type.686248.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.b9f: type = facet_access_type %T.bd1 [symbolic]
// CHECK:STDOUT:   %pattern_type.d13aea.2: type = pattern_type %T.as_type.b9f [symbolic]
// CHECK:STDOUT:   %x.param_patt.1bf: %pattern_type.d13aea.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.c79: %pattern_type.d13aea.2 = wrapper_binding_pattern x, %x.param_patt.1bf [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d1c4.2 = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.92e927.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.5a05b6.2: %EqWith.assoc_type.92e927.2 = assoc_entity element0, imports.%Core.import_ref.def [symbolic]
// CHECK:STDOUT:   %assoc1.09a898.2: %EqWith.assoc_type.92e927.2 = assoc_entity element1, imports.%Core.import_ref.6f1 [symbolic]
// CHECK:STDOUT:   %require_complete.8a6cdd.2: <witness> = require_complete_type %EqWith.type.686248.2 [symbolic]
// CHECK:STDOUT:   %assoc0.485f43.3: %EqWith.assoc_type.92e927.1 = assoc_entity element0, imports.%Core.import_ref.a05e82.3 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness.ae2c3a.2: <witness> = lookup_impl_witness %T.bd1, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e5264.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T.bd1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0896a8.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T.bd1) [symbolic]
// CHECK:STDOUT:   %.161b4c.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.4e5264.3, %T.bd1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.52b4ea.2: %.161b4c.2 = impl_witness_access %EqWith.lookup_impl_witness.ae2c3a.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2d3bf0.2: <specific function> = specific_impl_function %impl.elem0.52b4ea.2, @EqWith.WithSelf.Equal(%U, %T.bd1) [symbolic]
// CHECK:STDOUT:   %assoc1.0bbec6.2: %EqWith.assoc_type.92e927.1 = assoc_entity element1, imports.%Core.import_ref.7aa136.2 [symbolic]
// CHECK:STDOUT:   %.814: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0896a8.3, %T.bd1 [symbolic]
// CHECK:STDOUT:   %impl.elem1.eb2: %.814 = impl_witness_access %EqWith.lookup_impl_witness.ae2c3a.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0b5: <specific function> = specific_impl_function %impl.elem1.eb2, @EqWith.WithSelf.NotEqual(%U, %T.bd1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507fad.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.635006.1: %OrderedWith.type.507fad.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a530.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.e10b91.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.947541.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.bb34ec.1: %OrderedWith.WithSelf.Greater.type.947541.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.036982.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.ac2301.1: %OrderedWith.WithSelf.LessOrEquivalent.type.036982.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104f17.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.51e833.1: %OrderedWith.WithSelf.Less.type.104f17.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507fad.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.88a: type = pattern_type %OrderedWith.type.507fad.2 [symbolic]
// CHECK:STDOUT:   %T.patt.b1d: %pattern_type.88a = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.635: %OrderedWith.type.507fad.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.62a: type = facet_access_type %T.635 [symbolic]
// CHECK:STDOUT:   %pattern_type.f17d83.2: type = pattern_type %T.as_type.62a [symbolic]
// CHECK:STDOUT:   %x.param_patt.1a7: %pattern_type.f17d83.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.708: %pattern_type.f17d83.2 = wrapper_binding_pattern x, %x.param_patt.1a7 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a530.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.10837a.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic]
// CHECK:STDOUT:   %assoc1.4f4378.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element1, imports.%Core.import_ref.ffd [symbolic]
// CHECK:STDOUT:   %assoc2.60074a.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element2, imports.%Core.import_ref.ecd [symbolic]
// CHECK:STDOUT:   %assoc3.4e66f6.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element3, imports.%Core.import_ref.a69 [symbolic]
// CHECK:STDOUT:   %require_complete.f6d: <witness> = require_complete_type %OrderedWith.type.507fad.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d91: %OrderedWith.assoc_type.06a530.1 = assoc_entity element0, imports.%Core.import_ref.97f [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.635, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104f17.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.036982.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.947541.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %.236: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.104f17.3, %T.635 [symbolic]
// CHECK:STDOUT:   %impl.elem0.935: %.236 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.15a: <specific function> = specific_impl_function %impl.elem0.935, @OrderedWith.WithSelf.Less(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %assoc1.2e2: %OrderedWith.assoc_type.06a530.1 = assoc_entity element1, imports.%Core.import_ref.80a [symbolic]
// CHECK:STDOUT:   %.0de: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.036982.3, %T.635 [symbolic]
// CHECK:STDOUT:   %impl.elem1.19b: %.0de = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1d5: <specific function> = specific_impl_function %impl.elem1.19b, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %assoc2.935: %OrderedWith.assoc_type.06a530.1 = assoc_entity element2, imports.%Core.import_ref.fe3 [symbolic]
// CHECK:STDOUT:   %.5e7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.947541.3, %T.635 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.5e7 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.913: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T.635) [symbolic]
// CHECK:STDOUT:   %assoc3.dfc: %OrderedWith.assoc_type.06a530.1 = assoc_entity element3, imports.%Core.import_ref.1f1 [symbolic]
// CHECK:STDOUT:   %.489: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3, %T.635 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.489 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e09: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T.635) [symbolic]
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
// CHECK:STDOUT:   %EqWith.type.3fd: type = facet_type <@EqWith.1, @EqWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.74e: type = pattern_type %EqWith.type.3fd [concrete]
// CHECK:STDOUT:   %T.patt.af3: %pattern_type.74e = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.62b: %pattern_type.80f = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.80c: %pattern_type.80f = wrapper_binding_pattern y, %y.param_patt.62b [concrete]
// CHECK:STDOUT:   %OrderedWith.type.43a: type = facet_type <@OrderedWith.1, @OrderedWith.1(%DefaultSpaceshipOnly)> [concrete]
// CHECK:STDOUT:   %pattern_type.8cb: type = pattern_type %OrderedWith.type.43a [concrete]
// CHECK:STDOUT:   %T.patt.374: %pattern_type.8cb = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c97: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsStrongOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.d33: type = pattern_type %OrderedWith.type.c97 [concrete]
// CHECK:STDOUT:   %T.patt.9ea: %pattern_type.d33 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.794: %pattern_type.37f = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.d59: %pattern_type.37f = wrapper_binding_pattern y, %y.param_patt.794 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a20: type = facet_type <@OrderedWith.1, @OrderedWith.1(%ReturnsPartialOrdering)> [concrete]
// CHECK:STDOUT:   %pattern_type.98ad: type = pattern_type %OrderedWith.type.a20 [concrete]
// CHECK:STDOUT:   %T.patt.d22: %pattern_type.98ad = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.301: %pattern_type.9ae = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.f84: %pattern_type.9ae = wrapper_binding_pattern y, %y.param_patt.301 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a97: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeRhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.7af: type = pattern_type %OrderedWith.type.a97 [concrete]
// CHECK:STDOUT:   %T.patt.775: %pattern_type.7af = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.da2: %pattern_type.aac = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.b85: %pattern_type.aac = wrapper_binding_pattern y, %y.param_patt.da2 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.527: type = facet_type <@OrderedWith.1, @OrderedWith.1(%CrossTypeLhs)> [concrete]
// CHECK:STDOUT:   %pattern_type.89f: type = pattern_type %OrderedWith.type.527 [concrete]
// CHECK:STDOUT:   %T.patt.f7b: %pattern_type.89f = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.65d: %pattern_type.eb5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.80b: %pattern_type.eb5 = wrapper_binding_pattern y, %y.param_patt.65d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.f86: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.92e927.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.485f43.3)]
// CHECK:STDOUT:   %Core.import_ref.91b: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.92e927.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.0bbec6.2)]
// CHECK:STDOUT:   %Core.import_ref.6f1: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.0896a8.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.cb1497.1)]
// CHECK:STDOUT:   %Core.import_ref.def: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.4e5264.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.74d211.1)]
// CHECK:STDOUT:   %Core.import_ref.a05e82.3 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7aa136.2 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8cd: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.d91)]
// CHECK:STDOUT:   %Core.import_ref.010: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.2e2)]
// CHECK:STDOUT:   %Core.import_ref.16b: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.935)]
// CHECK:STDOUT:   %Core.import_ref.d00: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.dfc)]
// CHECK:STDOUT:   %Core.import_ref.a69: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.e10b91.1)]
// CHECK:STDOUT:   %Core.import_ref.ecd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.947541.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.bb34ec.1)]
// CHECK:STDOUT:   %Core.import_ref.ffd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.036982.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.ac2301.1)]
// CHECK:STDOUT:   %Core.import_ref.b46: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.104f17.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.51e833.1)]
// CHECK:STDOUT:   %Core.import_ref.97f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.80a = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fe3 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1f1 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc4(%U.loc4_12.2: type, %T.loc4_21.2: @EqWith.loc4.%EqWith.type.loc4_36.1 (%EqWith.type.686248.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %EqWith.type.loc4_36.1 [symbolic = %require_complete.loc6 (constants.%require_complete.8a6cdd.2)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.92e927.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = assoc_entity element0, imports.%Core.import_ref.def [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_21.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.4e5264.3)]
// CHECK:STDOUT:   %.loc6_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc4_21.1 [symbolic = %.loc6_5.2 (constants.%.161b4c.2)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_21.1, @EqWith.1, @EqWith.1(%U.loc4_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness.ae2c3a.2)]
// CHECK:STDOUT:   %impl.elem0.loc6_5.2: @EqWith.loc4.%.loc6_5.2 (%.161b4c.2) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.52b4ea.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_5.2: <specific function> = specific_impl_function %impl.elem0.loc6_5.2, @EqWith.WithSelf.Equal(%U.loc4_12.1, %T.loc4_21.1) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.2d3bf0.2)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = assoc_entity element1, imports.%Core.import_ref.6f1 [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc4_12.1, %T.loc4_21.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.0896a8.3)]
// CHECK:STDOUT:   %.loc7_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc4_21.1 [symbolic = %.loc7_5.2 (constants.%.814)]
// CHECK:STDOUT:   %impl.elem1.loc7_5.2: @EqWith.loc4.%.loc7_5.2 (%.814) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.eb2)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_5.2: <specific function> = specific_impl_function %impl.elem1.loc7_5.2, @EqWith.WithSelf.NotEqual(%U.loc4_12.1, %T.loc4_21.1) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.0b5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc4.%T.as_type.loc4_42.1 (%T.as_type.b9f), %y.param: @EqWith.loc4.%U.loc4_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc6: @EqWith.loc4.%T.as_type.loc4_42.1 (%T.as_type.b9f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc6: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc6: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_36.1 (constants.%EqWith.type.686248.2)]
// CHECK:STDOUT:     %.loc6_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = specific_constant imports.%Core.import_ref.f86, @EqWith.WithSelf(constants.%U, constants.%Self.bd18d5.1) [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = name_ref Equal, %.loc6_5.1 [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:     %impl.elem0.loc6_5.1: @EqWith.loc4.%.loc6_5.2 (%.161b4c.2) = impl_witness_access constants.%EqWith.lookup_impl_witness.ae2c3a.2, element0 [symbolic = %impl.elem0.loc6_5.2 (constants.%impl.elem0.52b4ea.2)]
// CHECK:STDOUT:     %bound_method.loc6_5.1: <bound method> = bound_method %x.ref.loc6, %impl.elem0.loc6_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_5.1: <specific function> = specific_impl_function %impl.elem0.loc6_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T.bd1) [symbolic = %specific_impl_fn.loc6_5.2 (constants.%specific_impl_fn.2d3bf0.2)]
// CHECK:STDOUT:     %bound_method.loc6_5.2: <bound method> = bound_method %x.ref.loc6, %specific_impl_fn.loc6_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc6_5.2(%x.ref.loc6, %y.ref.loc6)
// CHECK:STDOUT:     %x.ref.loc7: @EqWith.loc4.%T.as_type.loc4_42.1 (%T.as_type.b9f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc7: @EqWith.loc4.%U.loc4_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc7: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc4_36.1 (constants.%EqWith.type.686248.2)]
// CHECK:STDOUT:     %.loc7_5.1: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = specific_constant imports.%Core.import_ref.91b, @EqWith.WithSelf(constants.%U, constants.%Self.bd18d5.1) [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc4.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = name_ref NotEqual, %.loc7_5.1 [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:     %impl.elem1.loc7_5.1: @EqWith.loc4.%.loc7_5.2 (%.814) = impl_witness_access constants.%EqWith.lookup_impl_witness.ae2c3a.2, element1 [symbolic = %impl.elem1.loc7_5.2 (constants.%impl.elem1.eb2)]
// CHECK:STDOUT:     %bound_method.loc7_5.1: <bound method> = bound_method %x.ref.loc7, %impl.elem1.loc7_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_5.1: <specific function> = specific_impl_function %impl.elem1.loc7_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T.bd1) [symbolic = %specific_impl_fn.loc7_5.2 (constants.%specific_impl_fn.0b5)]
// CHECK:STDOUT:     %bound_method.loc7_5.2: <bound method> = bound_method %x.ref.loc7, %specific_impl_fn.loc7_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc7_5.2(%x.ref.loc7, %y.ref.loc7)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc11(%U.loc11_17.2: type, %T.loc11_26.2: @OrderedWith.loc11.%OrderedWith.type.loc11_46.1 (%OrderedWith.type.507fad.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc11_46.1 [symbolic = %require_complete.loc13 (constants.%require_complete.f6d)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.06a530.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.104f17.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc11_26.1 [symbolic = %.loc13_5.2 (constants.%.236)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_26.1, @OrderedWith.1, @OrderedWith.1(%U.loc11_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @OrderedWith.loc11.%.loc13_5.2 (%.236) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.935)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @OrderedWith.WithSelf.Less(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.15a)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element1, imports.%Core.import_ref.ffd [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.036982.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc11_26.1 [symbolic = %.loc14_5.2 (constants.%.0de)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @OrderedWith.loc11.%.loc14_5.2 (%.0de) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.19b)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.1d5)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element2, imports.%Core.import_ref.ecd [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.947541.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc11_26.1 [symbolic = %.loc15_5.2 (constants.%.5e7)]
// CHECK:STDOUT:   %impl.elem2.loc15_5.2: @OrderedWith.loc11.%.loc15_5.2 (%.5e7) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem2.loc15_5.2, @OrderedWith.WithSelf.Greater(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.913)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element3, imports.%Core.import_ref.a69 [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc11_26.1 [symbolic = %.loc16_5.2 (constants.%.489)]
// CHECK:STDOUT:   %impl.elem3.loc16_5.2: @OrderedWith.loc11.%.loc16_5.2 (%.489) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem3.loc16_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc11_17.1, %T.loc11_26.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.e09)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc11.%T.as_type.loc11_52.1 (%T.as_type.62a), %y.param: @OrderedWith.loc11.%U.loc11_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @OrderedWith.loc11.%T.as_type.loc11_52.1 (%T.as_type.62a) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc13: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc13_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.8cd, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref Less, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @OrderedWith.loc11.%.loc13_5.2 (%.236) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.935)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T.635) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.15a)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @OrderedWith.loc11.%T.as_type.loc11_52.1 (%T.as_type.62a) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc14: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc14_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.010, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref LessOrEquivalent, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @OrderedWith.loc11.%.loc14_5.2 (%.0de) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1.19b)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T.635) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.1d5)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc11.%T.as_type.loc11_52.1 (%T.as_type.62a) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.16b, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref Greater, %.loc15_5.1 [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:     %impl.elem2.loc15_5.1: @OrderedWith.loc11.%.loc15_5.2 (%.5e7) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem2.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem2.loc15_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T.635) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.913)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc11.%T.as_type.loc11_52.1 (%T.as_type.62a) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc11.%U.loc11_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc11_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.d00, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref GreaterOrEquivalent, %.loc16_5.1 [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:     %impl.elem3.loc16_5.1: @OrderedWith.loc11.%.loc16_5.2 (%.489) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc16_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem3.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem3.loc16_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T.635) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.e09)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%U, constants.%T.bd1) {
// CHECK:STDOUT:   %U.patt.loc4_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc4_36.1 => constants.%EqWith.type.686248.2
// CHECK:STDOUT:   %pattern_type.loc4_21 => constants.%pattern_type.295
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T.patt.738
// CHECK:STDOUT:   %T.loc4_21.1 => constants.%T.bd1
// CHECK:STDOUT:   %T.as_type.loc4_42.1 => constants.%T.as_type.b9f
// CHECK:STDOUT:   %pattern_type.loc4_40 => constants.%pattern_type.d13aea.2
// CHECK:STDOUT:   %x.param_patt.loc4_40.2 => constants.%x.param_patt.1bf
// CHECK:STDOUT:   %x.patt.loc4_40.2 => constants.%x.patt.c79
// CHECK:STDOUT:   %pattern_type.loc4_46 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc4_46.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc4_46.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%U, constants.%T.635) {
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc11_46.1 => constants.%OrderedWith.type.507fad.2
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.88a
// CHECK:STDOUT:   %T.patt.loc11_26.2 => constants.%T.patt.b1d
// CHECK:STDOUT:   %T.loc11_26.1 => constants.%T.635
// CHECK:STDOUT:   %T.as_type.loc11_52.1 => constants.%T.as_type.62a
// CHECK:STDOUT:   %pattern_type.loc11_50 => constants.%pattern_type.f17d83.2
// CHECK:STDOUT:   %x.param_patt.loc11_50.2 => constants.%x.param_patt.1a7
// CHECK:STDOUT:   %x.patt.loc11_50.2 => constants.%x.patt.708
// CHECK:STDOUT:   %pattern_type.loc11_56 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc11_56.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc11_56.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc4(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.patt.loc4_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc4_12.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %EqWith.type.loc4_36.1 => constants.%EqWith.type.3fd
// CHECK:STDOUT:   %pattern_type.loc4_21 => constants.%pattern_type.74e
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T.patt.af3
// CHECK:STDOUT:   %T.loc4_21.1 => <error>
// CHECK:STDOUT:   %T.as_type.loc4_42.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_40 => <error>
// CHECK:STDOUT:   %x.param_patt.loc4_40.2 => <error>
// CHECK:STDOUT:   %x.patt.loc4_40.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc4_46 => constants.%pattern_type.80f
// CHECK:STDOUT:   %y.param_patt.loc4_46.2 => constants.%y.param_patt.62b
// CHECK:STDOUT:   %y.patt.loc4_46.2 => constants.%y.patt.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%DefaultSpaceshipOnly, <error>) {
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%DefaultSpaceshipOnly
// CHECK:STDOUT:   %OrderedWith.type.loc11_46.1 => constants.%OrderedWith.type.43a
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.8cb
// CHECK:STDOUT:   %T.patt.loc11_26.2 => constants.%T.patt.374
// CHECK:STDOUT:   %T.loc11_26.1 => <error>
// CHECK:STDOUT:   %T.as_type.loc11_52.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_50 => <error>
// CHECK:STDOUT:   %x.param_patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %x.patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_56 => constants.%pattern_type.80f
// CHECK:STDOUT:   %y.param_patt.loc11_56.2 => constants.%y.param_patt.62b
// CHECK:STDOUT:   %y.patt.loc11_56.2 => constants.%y.patt.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsStrongOrdering, <error>) {
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsStrongOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_46.1 => constants.%OrderedWith.type.c97
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.d33
// CHECK:STDOUT:   %T.patt.loc11_26.2 => constants.%T.patt.9ea
// CHECK:STDOUT:   %T.loc11_26.1 => <error>
// CHECK:STDOUT:   %T.as_type.loc11_52.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_50 => <error>
// CHECK:STDOUT:   %x.param_patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %x.patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_56 => constants.%pattern_type.37f
// CHECK:STDOUT:   %y.param_patt.loc11_56.2 => constants.%y.param_patt.794
// CHECK:STDOUT:   %y.patt.loc11_56.2 => constants.%y.patt.d59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%ReturnsPartialOrdering, <error>) {
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%ReturnsPartialOrdering
// CHECK:STDOUT:   %OrderedWith.type.loc11_46.1 => constants.%OrderedWith.type.a20
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.98ad
// CHECK:STDOUT:   %T.patt.loc11_26.2 => constants.%T.patt.d22
// CHECK:STDOUT:   %T.loc11_26.1 => <error>
// CHECK:STDOUT:   %T.as_type.loc11_52.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_50 => <error>
// CHECK:STDOUT:   %x.param_patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %x.patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_56 => constants.%pattern_type.9ae
// CHECK:STDOUT:   %y.param_patt.loc11_56.2 => constants.%y.param_patt.301
// CHECK:STDOUT:   %y.patt.loc11_56.2 => constants.%y.patt.f84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeRhs, <error>) {
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeRhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_46.1 => constants.%OrderedWith.type.a97
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.7af
// CHECK:STDOUT:   %T.patt.loc11_26.2 => constants.%T.patt.775
// CHECK:STDOUT:   %T.loc11_26.1 => <error>
// CHECK:STDOUT:   %T.as_type.loc11_52.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_50 => <error>
// CHECK:STDOUT:   %x.param_patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %x.patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_56 => constants.%pattern_type.aac
// CHECK:STDOUT:   %y.param_patt.loc11_56.2 => constants.%y.param_patt.da2
// CHECK:STDOUT:   %y.patt.loc11_56.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc11(constants.%CrossTypeLhs, <error>) {
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_17.1 => constants.%CrossTypeLhs
// CHECK:STDOUT:   %OrderedWith.type.loc11_46.1 => constants.%OrderedWith.type.527
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.89f
// CHECK:STDOUT:   %T.patt.loc11_26.2 => constants.%T.patt.f7b
// CHECK:STDOUT:   %T.loc11_26.1 => <error>
// CHECK:STDOUT:   %T.as_type.loc11_52.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_50 => <error>
// CHECK:STDOUT:   %x.param_patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %x.patt.loc11_50.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc11_56 => constants.%pattern_type.eb5
// CHECK:STDOUT:   %y.param_patt.loc11_56.2 => constants.%y.param_patt.65d
// CHECK:STDOUT:   %y.patt.loc11_56.2 => constants.%y.patt.80b
// CHECK:STDOUT: }
// CHECK:STDOUT:
