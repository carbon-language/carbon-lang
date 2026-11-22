// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/extendable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/extendable.carbon

// --- base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

fn Test() {
  B as Core.Extendable;
}

// --- abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class A {}

fn Test() {
  A as Core.Extendable;
}

// --- generic_base_class.carbon
library "[[@TEST_NAME]]";

base class B(unused T: type) {}

fn Test() {
  B(()) as Core.Extendable;
}

fn TestSymbolic[T: type]() {
  B(T) as Core.Extendable;
}

// --- fail_class.carbon
library "[[@TEST_NAME]]";

class C {}

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   C as Core.Extendable;
  // CHECK:STDERR:        ·   ━━┳━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot convert type `C` into type implementing `Core.Extendable`
  // CHECK:STDERR:        ·        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  C as Core.Extendable;
}

// --- fail_derived_from_base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

class C {
  extend base: B;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_derived_from_base_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   C as Core.Extendable;
  // CHECK:STDERR:        ·   ━━┳━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot convert type `C` into type implementing `Core.Extendable`
  // CHECK:STDERR:        ·        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  C as Core.Extendable;
}

// --- fail_partial_class.carbon
library "[[@TEST_NAME]]";

base class B {}

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_partial_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   partial B as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `partial B` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  partial B as Core.Extendable;
}

// --- fail_tuple.carbon
library "[[@TEST_NAME]]";

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_tuple.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   ((), ()) as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `((), ())` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  ((), ()) as Core.Extendable;
}

// --- fail_struct.carbon
library "[[@TEST_NAME]]";

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_struct.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   {.a: ()} as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `{.a: ()}` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  {.a: ()} as Core.Extendable;
}

// --- fail_symbolic.carbon
library "[[@TEST_NAME]]";

fn Test[T: type]() {
  // CHECK:STDERR:        ╭─┤ fail_symbolic.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   T as Core.Extendable;
  // CHECK:STDERR:        ·   ━━┳━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot convert type `T` into type implementing `Core.Extendable`
  // CHECK:STDERR:        ·        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  T as Core.Extendable;
}

// --- fail_impl_base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

// CHECK:STDERR:        ╭─┤ fail_impl_base_class.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ impl B as Core.Extendable {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: explicit impl of `Core.Extendable`; implemented automatically for
// CHECK:STDERR:        ·    ┃        `base` and `abstract` classes
// CHECK:STDERR:        ·    ┃        [ImplOfExtendable]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl B as Core.Extendable {}

// --- fail_impl_final_class.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR:        ╭─┤ fail_impl_final_class.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ impl C as Core.Extendable {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: explicit impl of `Core.Extendable`; implemented automatically for
// CHECK:STDERR:        ·    ┃        `base` and `abstract` classes
// CHECK:STDERR:        ·    ┃        [ImplOfExtendable]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C as Core.Extendable {}

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn G(ref c: C) {
  // CHECK:STDERR:        ╭─┤ fail_impl_final_class.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn F[T: Core.Extendable](ref t: T) -> partial T* {
  // CHECK:STDERR:        ·    ┬
  // CHECK:STDERR:        ·    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·       │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(ref c);
  // CHECK:STDERR:        ·   ━━┳━━━━━
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot convert type `C` into type implementing `Core.Extendable`
  // CHECK:STDERR:        ·        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(ref c);
}

// --- fail_impl_base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

// CHECK:STDERR:        ╭─┤ fail_impl_base_class.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ impl B as Core.Extendable {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: explicit impl of `Core.Extendable`; implemented automatically for
// CHECK:STDERR:        ·    ┃        `base` and `abstract` classes
// CHECK:STDERR:        ·    ┃        [ImplOfExtendable]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl B as Core.Extendable {}

// --- fail_impl_final_class.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR:        ╭─┤ fail_impl_final_class.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ impl C as Core.Extendable {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: explicit impl of `Core.Extendable`; implemented automatically for
// CHECK:STDERR:        ·    ┃        `base` and `abstract` classes
// CHECK:STDERR:        ·    ┃        [ImplOfExtendable]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C as Core.Extendable {}

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn G(ref c: C) {
  // CHECK:STDERR:        ╭─┤ fail_impl_final_class.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn F[T: Core.Extendable](ref t: T) -> partial T* {
  // CHECK:STDERR:        ·    ┬
  // CHECK:STDERR:        ·    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·       │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(ref c);
  // CHECK:STDERR:        ·   ━━┳━━━━━
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot convert type `C` into type implementing `Core.Extendable`
  // CHECK:STDERR:        ·        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(ref c);
}

