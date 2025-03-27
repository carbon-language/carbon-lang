// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/symbolic_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/symbolic_lookup.carbon

// --- self_type_facet_value_of_facet_access_type.carbon
library "[[@TEST_NAME]]";

interface X {}
interface Y {}
interface Z {}

impl forall [T:! Y] T as X {}
impl forall [T:! Z] T as X {}

class C(T:! X) {}

// The `adapt C(T)` line produces a symbolic impl lookup that `T` impls `X`:
// - At first the impl lookup query in the eval block has self as
//   `BindSymbolicName(T:! Y)`.
//
// The call to construct `D` from in `Test()` makes a specific for `D`:
// - The value of `T` in `D` is deduced to be a `FacetValue` of type `Y & Z`
//   pointing to the `BindSymbolicName(T:! Y & Z)` from `Test()`. But
//   `FacetValue` needs an instruction of type `TypeType` so the
//   `BindSymbolicName` is wrapped in `FacetAccessType`, meaning it's
//   `FacetValue(FacetAccessType(BindSymbolicName(T:! Y & Z)))`.
// - That facet value is written to the specific of `D` constructed from
//   `Test()`.
// - The eval block of `D` is run with the above specific. This runs the
//   symbolic impl lookup that `T` impls `X`. But with the `T` rewritten by the
//   specific to the `FacetValue(FacetAccessType(BindSymbolicName(T:! Y & Z)))`.
//
// The impl lookup unwraps the `FacetValue` to get the underlying type as it may
// have access to a more constrained FacetType (with access to more interfaces)
// than the `FacetValue` itself.
//
// In this case, the unwrap finds a `FacetAccessType`. Impl lookup must handle
// the presence or absence of `FacetAccessType` in the query self type
// consistently, while comparing the self type (after deduction) to the impl's
// self type. This verifies we handle the case of the query not having a
// `FacetAccessType` in the initial query that makes the symbolic lookup
// instruction (`LookupImplWitness`), but then the self type being replaced
// with a `FacetAccessType` when evaluating the instruction.
class D(T:! Y) { adapt C(T); }

fn Test(T:! Y & Z, d: D(T)) {}
