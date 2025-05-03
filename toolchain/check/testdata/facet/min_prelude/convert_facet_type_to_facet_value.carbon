// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_type_to_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_type_to_facet_value.carbon

// --- convert_facet_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}

// TODO: This may be rejected in the future.
// https://github.com/carbon-language/carbon-lang/issues/4853
impl Animal as Eats {}

fn Feed(e:! Eats) {}

fn F() {
  Feed(Animal);
}

// --- fail_facet_value_to_facet_type.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}

// TODO: This may be rejected in the future.
// https://github.com/carbon-language/carbon-lang/issues/4853
impl Animal as Eats {}

fn Feed(e:! Eats) {}

class Goat {}
impl Goat as Animal {}

fn F() {
  // CHECK:STDERR: fail_facet_value_to_facet_type.carbon:[[@LINE+7]]:3: error: cannot convert type `Goat as Animal` that implements `Animal` into type implementing `Eats` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   Feed(Goat as Animal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_facet_value_to_facet_type.carbon:[[@LINE-9]]:9: note: initializing generic parameter `e` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Feed(e:! Eats) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  Feed(Goat as Animal);
}

// --- fail_convert_multi_interface_facet_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}
interface Climbs {}

// TODO: This may be rejected in the future.
// https://github.com/carbon-language/carbon-lang/issues/4853
impl Animal as Eats {}

fn Feed(e:! Eats) {}

class Goat {}
impl Goat as Animal {}
impl Goat as Climbs {}

// These are expected to fail:
// https://github.com/carbon-language/carbon-lang/issues/4853#issuecomment-2707673344
fn F() {
  // CHECK:STDERR: fail_convert_multi_interface_facet_value_to_facet_value.carbon:[[@LINE+7]]:3: error: cannot convert type `Goat as Animal & Climbs` that implements `Animal & Climbs` into type implementing `Eats` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   Feed(Goat as (Animal & Climbs));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_convert_multi_interface_facet_value_to_facet_value.carbon:[[@LINE-12]]:9: note: initializing generic parameter `e` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Feed(e:! Eats) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  Feed(Goat as (Animal & Climbs));

  // CHECK:STDERR: fail_convert_multi_interface_facet_value_to_facet_value.carbon:[[@LINE+7]]:3: error: cannot convert type `Animal & Climbs` into type implementing `Eats` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Feed(Animal & Climbs);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_convert_multi_interface_facet_value_to_facet_value.carbon:[[@LINE-21]]:9: note: initializing generic parameter `e` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Feed(e:! Eats) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  Feed(Animal & Climbs);
}
