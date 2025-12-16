// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/orphan.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/orphan.carbon

// --- imports.carbon
library "[[@TEST_NAME]]";

class ImportedC {}
class ImportedD(T:! type) {}

interface ImportedY {}
interface ImportedZ(T:! type) {}

constraint ImportedN(T:! type) {}

class ImportedAnyParam[T:! type](X:! T) {}

// --- fail_class_definition_missing.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C;

// CHECK:STDERR: fail_class_definition_missing.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl C as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as ImportedY {}

// --- fail_class_definition_missing_for_self_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C;

// CHECK:STDERR: fail_class_definition_missing_for_self_specific.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedD(C) as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedD(C) as ImportedY {}

// --- fail_class_definition_missing_for_interface_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C;

// CHECK:STDERR: fail_class_definition_missing_for_interface_specific.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedC as ImportedZ(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedC as ImportedZ(C) {}

// --- fail_imported_class.carbon
library "[[@TEST_NAME]]";

import library "imports";

// CHECK:STDERR: fail_imported_class.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedC as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedC as ImportedY {}

// --- fail_imported_class_for_self_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

// CHECK:STDERR: fail_imported_class_for_self_specific.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedD(ImportedC) as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedD(ImportedC) as ImportedY {}

// --- fail_imported_class_for_interface_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

// CHECK:STDERR: fail_imported_class_for_interface_specific.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedC as ImportedZ(ImportedC) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedC as ImportedZ(ImportedC) {}

// --- class_definition_after.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C;

impl C as ImportedY {}

class C {}

// --- class_definition_after_for_self_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C;

impl ImportedD(C) as ImportedY {}

class C {}

// --- class_definition_after_for_interface_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C;

impl ImportedC as ImportedZ(C) {}

class C {}

// --- class_definition_before.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C {}

impl C as ImportedY {}

// --- class_definition_before_for_self_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C {}

impl ImportedD(C) as ImportedY {}

// --- class_definition_before_for_interface_specific.carbon
library "[[@TEST_NAME]]";

import library "imports";

class C {}

impl ImportedC as ImportedZ(C) {}

// --- fail_generic_class_from_import.carbon
library "[[@TEST_NAME]]";

import library "imports";

// CHECK:STDERR: fail_generic_class_from_import.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedAnyParam(ImportedD) as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedAnyParam(ImportedD) as ImportedY {}

// --- generic_class.carbon
library "[[@TEST_NAME]]";

import library "imports";

class D(T:! type) {}

impl ImportedAnyParam(D) as ImportedY {}

// --- fail_generic_interface_from_import.carbon
library "[[@TEST_NAME]]";

import library "imports";

// CHECK:STDERR: fail_generic_interface_from_import.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedAnyParam(ImportedZ) as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedAnyParam(ImportedZ) as ImportedY {}

// --- generic_interface.carbon
library "[[@TEST_NAME]]";

import library "imports";

interface Z(T:! type) {}

impl ImportedAnyParam(Z) as ImportedY {}

// --- fail_generic_named_constraint_from_import.carbon
library "[[@TEST_NAME]]";

import library "imports";

// CHECK:STDERR: fail_generic_named_constraint_from_import.carbon:[[@LINE+4]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl ImportedAnyParam(ImportedN) as ImportedY {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ImportedAnyParam(ImportedN) as ImportedY {}

// --- generic_named_constraint.carbon
library "[[@TEST_NAME]]";

import library "imports";

constraint N(T:! type) {}

impl ImportedAnyParam(N) as ImportedY {}
