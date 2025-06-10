// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/impl_overlap.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/impl_overlap.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- interface_z.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

// --- interface_z_with_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

final impl forall [T:! type] T as Z(T) {}

// --- type_d.carbon
library "[[@TEST_NAME]]";

class D(T:! type) {}

// ============================================================================
// Test files
// ============================================================================

// --- final_impl_with_interface_generic_self.carbon
library "[[@TEST_NAME]]";

interface Z {}

final impl forall [T:! type] T as Z {}

// --- final_impl_with_interface_concrete_self_same_file.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;

final impl C as Z {}

// --- final_impl_with_interface_concrete_self_from_other_file.carbon
library "[[@TEST_NAME]]";

import library "type_d";

interface Z {}

// The final impl is for a root self type from another file, but the interface
// is in the same file, so this is valid.
final impl D(()) as Z {}

// --- final_impl_with_interface_symbolic_self_from_other_file.carbon
library "[[@TEST_NAME]]";

import library "type_d";

interface Z {}

// The root self type is from another file, but the interface is from this file.
// This tests the self type being a symbolic type due to the type parameter T.
final impl forall [T:! type] D(T) as Z {}

// --- fail_multiple_finals_overlap_with_interface.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

final impl forall [T:! type] T as Z(T) {}

// The final impl here overlaps with the final impl above, but is not in a
// match_first.
//
// CHECK:STDERR: fail_multiple_finals_overlap_with_interface.carbon:[[@LINE+7]]:1: error: `final impl` overlaps with `final impl` from same file outside a `match_first` block [FinalImplOverlapsSameFile]
// CHECK:STDERR: final impl forall [T:! type] T as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_multiple_finals_overlap_with_interface.carbon:[[@LINE-8]]:1: note: other `final impl` here [FinalImplOverlapsSameFileNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z(()) {}

class C;

// The final impl here overlaps with the first final impl above, but is not in
// a match_first.
//
// CHECK:STDERR: fail_multiple_finals_overlap_with_interface.carbon:[[@LINE+7]]:1: error: `final impl` overlaps with `final impl` from same file outside a `match_first` block [FinalImplOverlapsSameFile]
// CHECK:STDERR: final impl C as Z({}) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_multiple_finals_overlap_with_interface.carbon:[[@LINE-22]]:1: note: other `final impl` here [FinalImplOverlapsSameFileNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl C as Z({}) {}

// --- fail_multiple_finals_overlap_with_self_type.carbon
library "[[@TEST_NAME]]";

import library "interface_z";

class C;

final impl C as Z(()) {}

// The final impl here overlaps with the first final impl above, but is not in
// a match_first.
//
// CHECK:STDERR: fail_multiple_finals_overlap_with_self_type.carbon:[[@LINE+7]]:1: error: `final impl` overlaps with `final impl` from same file outside a `match_first` block [FinalImplOverlapsSameFile]
// CHECK:STDERR: final impl forall [T:! type] C as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_multiple_finals_overlap_with_self_type.carbon:[[@LINE-8]]:1: note: other `final impl` here [FinalImplOverlapsSameFileNote]
// CHECK:STDERR: final impl C as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] C as Z(T) {}

// --- multiple_finals_with_nonoverlapping_with_interface.carbon
library "[[@TEST_NAME]]";

import library "type_d";

interface Z {}

// Two final impls for D as Z, but they are not overlapping so they are allowed
// outside of match_first.
final impl D(()) as Z {}
final impl D({}) as Z {}

// --- multiple_finals_with_nonoverlapping_with_self_type.carbon
library "[[@TEST_NAME]]";

import library "interface_z";

class C;

// Two final impls for C as Z, but they are not overlapping so they are allowed
// outside of match_first.
final impl C as Z(()) {}
final impl C as Z({}) {}

// --- final_impl_with_root_self_type.carbon
library "[[@TEST_NAME]]";

import library "interface_z";

class C(T:! type);

// Can provide a specialized final blanket impl for a type defined in the same
// file.
final impl forall [T:! type] C(T) as Z(T) {}

// --- fail_final_impl_with_both_interface_and_self_but_different_files.carbon
library "[[@TEST_NAME]]";

import library "interface_z_with_impl";

class C;

// Can't write a final impl in both the interface's file and the root self
// type's file (when they are different files).
//
// CHECK:STDERR: fail_final_impl_with_both_interface_and_self_but_different_files.carbon:[[@LINE+8]]:1: error: `final impl` overlaps with `final impl` from another file [FinalImplOverlapsDifferentFile]
// CHECK:STDERR: final impl C as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_impl_with_both_interface_and_self_but_different_files.carbon:[[@LINE-10]]:1: in import [InImport]
// CHECK:STDERR: interface_z_with_impl.carbon:5:1: note: imported `final impl` here [FinalImplOverlapsDifferentFileNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl C as Z(()) {}

// --- fail_final_overlaps_final_from_other_file.carbon
library "[[@TEST_NAME]]";

import library "interface_z_with_impl";

class C;

// This final impl is overlapped by a final impl in the interface file, and you
// can't write a final impl in two different files.
//
// CHECK:STDERR: fail_final_overlaps_final_from_other_file.carbon:[[@LINE+8]]:1: error: `final impl` overlaps with `final impl` from another file [FinalImplOverlapsDifferentFile]
// CHECK:STDERR: final impl C as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_final_from_other_file.carbon:[[@LINE-10]]:1: in import [InImport]
// CHECK:STDERR: interface_z_with_impl.carbon:5:1: note: imported `final impl` here [FinalImplOverlapsDifferentFileNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl C as Z(C) {}

// --- fail_final_overlaps_non_final_from_other_file.carbon
library "[[@TEST_NAME]]";

import library "interface_z_with_impl";

class C;

// This non-final impl is subsumed by a final impl in the interface file.
//
// CHECK:STDERR: fail_final_overlaps_non_final_from_other_file.carbon:[[@LINE+8]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: impl C as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_non_final_from_other_file.carbon:[[@LINE-9]]:1: in import [InImport]
// CHECK:STDERR: interface_z_with_impl.carbon:5:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z(C) {}

// --- fail_final_different_file_from_self_and_interface.carbon
library "[[@TEST_NAME]]";

import library "interface_z";
import library "type_d";

class C;

// Can't make a final impl that is not in the same file as the self type nor
// the interface.
//
// CHECK:STDERR: fail_final_different_file_from_self_and_interface.carbon:[[@LINE+4]]:1: error: `final impl` found in file that does not contain the root self type nor the interface definition [FinalImplInvalidFile]
// CHECK:STDERR: final impl D(C) as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl D(C) as Z(()) {}

// --- fail_final_different_file_from_self_and_interface_with_generic_self.carbon
library "[[@TEST_NAME]]";

import library "interface_z";

class C;

// Can't make a final impl that is not in the same file as the self type nor
// the interface.
//
// CHECK:STDERR: fail_final_different_file_from_self_and_interface_with_generic_self.carbon:[[@LINE+4]]:1: error: `final impl` found in file that does not contain the root self type nor the interface definition [FinalImplInvalidFile]
// CHECK:STDERR: final impl forall [T:! type] T as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z(C) {}

// --- fail_final_overlaps_earlier_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

// CHECK:STDERR: fail_final_overlaps_earlier_non_final_impl.carbon:[[@LINE+3]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: impl () as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
impl () as Z(()) {}

// CHECK:STDERR: fail_final_overlaps_earlier_non_final_impl.carbon:[[@LINE+4]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z(T) {}

// --- fail_final_overlaps_later_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

final impl forall [T:! type] T as Z(T) {}

// CHECK:STDERR: fail_final_overlaps_later_non_final_impl.carbon:[[@LINE+7]]:1: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
// CHECK:STDERR: impl () as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_later_non_final_impl.carbon:[[@LINE-5]]:1: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Z(()) {}

// --- fail_final_overlaps_earlier_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

class C;

final impl C as Z(C) {}

// CHECK:STDERR: fail_final_overlaps_earlier_final_impl.carbon:[[@LINE+7]]:1: error: `final impl` overlaps with `final impl` from same file outside a `match_first` block [FinalImplOverlapsSameFile]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_earlier_final_impl.carbon:[[@LINE-5]]:1: note: other `final impl` here [FinalImplOverlapsSameFileNote]
// CHECK:STDERR: final impl C as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z(T) {}

// --- fail_final_overlaps_later_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

class C;

final impl forall [T:! type] T as Z(T) {}

// CHECK:STDERR: fail_final_overlaps_later_final_impl.carbon:[[@LINE+7]]:1: error: `final impl` overlaps with `final impl` from same file outside a `match_first` block [FinalImplOverlapsSameFile]
// CHECK:STDERR: final impl C as Z(C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlaps_later_final_impl.carbon:[[@LINE-5]]:1: note: other `final impl` here [FinalImplOverlapsSameFileNote]
// CHECK:STDERR: final impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl C as Z(C) {}

// --- fail_non_final_type_structure_matches_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}
interface Y {}

impl forall [T:! Y] T as Z(T) {}

class C(T:! type);

// No diagnosis here as the type structure is unique.
impl forall [T:! type] T as Z(C(T)) {}

// CHECK:STDERR: fail_non_final_type_structure_matches_non_final_impl.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_non_final_type_structure_matches_non_final_impl.carbon:[[@LINE-10]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl forall [T:! Y] T as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) {}

// --- non_final_type_structure_same_shape_but_different_concrete_types.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;
class D;

impl C as Z {}
impl D as Z {}

// --- partial_overlap_type_structure_of_non_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

impl forall [T:! type] T as Z(T) {}

class C;
// Partially overlaps the blanket impl, which is fine.
final impl C as Z(C) {}

class D;
// Partially overlaps the blanket impl, which is fine.
impl D as Z(D) {}

// --- fail_final_overlap_where_each_is_more_specific_than_the_other.carbon

interface Z(T:! type) {}

class C(T:! type) {}

// This should be diagnosed as overlapping final impls outside a `match_first`.
// The first impl is more specific in the interface constraint. The second is
// more specific in the self type. They overlap for the query `C(()) as Z(())`
// but neither impl is completely more specific than the other.
final impl forall [T:! type] C(T) as Z(()) {}
// CHECK:STDERR: fail_final_overlap_where_each_is_more_specific_than_the_other.carbon:[[@LINE+7]]:1: error: `final impl` overlaps with `final impl` from same file outside a `match_first` block [FinalImplOverlapsSameFile]
// CHECK:STDERR: final impl forall [T:! type] C(()) as Z(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_overlap_where_each_is_more_specific_than_the_other.carbon:[[@LINE-4]]:1: note: other `final impl` here [FinalImplOverlapsSameFileNote]
// CHECK:STDERR: final impl forall [T:! type] C(T) as Z(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] C(()) as Z(T) {}
