// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/impl_overlap_wrapped_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/impl_overlap_wrapped_types.carbon

// --- wrapped_types.carbon
library "[[@TEST_NAME]]";

base class C {}
base class D {}
interface I {}

impl C as I {}
impl const C as I {}
impl Core.MaybeUnformed(C) as I {}
impl partial C as I {}
impl C* as I {}

impl D as I {}
impl const D as I {}
impl Core.MaybeUnformed(D) as I {}
impl partial D as I {}
impl D* as I {}

// --- fail_const_overlap.carbon
library "[[@TEST_NAME]]";

base class C {}
interface I {}

impl C as I {}
impl const C as I {}
impl Core.MaybeUnformed(C) as I {}
impl partial C as I {}
impl C* as I {}

// CHECK:STDERR: fail_const_overlap.carbon:[[@LINE+7]]:1: error: redefinition of `impl const C as I` [ImplRedefinition]
// CHECK:STDERR: impl const C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_const_overlap.carbon:[[@LINE-8]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl const C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl const C as I {}

// --- fail_maybe_unformed_overlap.carbon
library "[[@TEST_NAME]]";

base class C {}
interface I {}

impl C as I {}
impl const C as I {}
impl Core.MaybeUnformed(C) as I {}
impl partial C as I {}
impl C* as I {}

// CHECK:STDERR: fail_maybe_unformed_overlap.carbon:[[@LINE+7]]:1: error: redefinition of `impl Core.MaybeUnformed(C) as I` [ImplRedefinition]
// CHECK:STDERR: impl Core.MaybeUnformed(C) as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_maybe_unformed_overlap.carbon:[[@LINE-7]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl Core.MaybeUnformed(C) as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl Core.MaybeUnformed(C) as I {}

// --- fail_partial_overlap.carbon
library "[[@TEST_NAME]]";

base class C {}
interface I {}

impl C as I {}
impl const C as I {}
impl Core.MaybeUnformed(C) as I {}
impl partial C as I {}
impl C* as I {}

// CHECK:STDERR: fail_partial_overlap.carbon:[[@LINE+7]]:1: error: redefinition of `impl partial C as I` [ImplRedefinition]
// CHECK:STDERR: impl partial C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_partial_overlap.carbon:[[@LINE-6]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl partial C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl partial C as I {}

// --- fail_pointer_overlap.carbon
library "[[@TEST_NAME]]";

base class C {}
interface I {}

impl C as I {}
impl const C as I {}
impl Core.MaybeUnformed(C) as I {}
impl partial C as I {}
impl C* as I {}

// CHECK:STDERR: fail_pointer_overlap.carbon:[[@LINE+7]]:1: error: redefinition of `impl C* as I` [ImplRedefinition]
// CHECK:STDERR: impl C* as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_pointer_overlap.carbon:[[@LINE-5]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl C* as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl C* as I {}
