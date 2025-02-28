// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/impl.carbon

// --- impl.carbon

impl package Geometry;

// --- impl_disambiguate.carbon

impl package DisambiguateImpl;

impl i32 as Interface {}

// --- impl_package_disambiguate.carbon

impl package DisambiguateImplPackage;

impl package.Name as package.Interface {}

// CHECK:STDOUT: - filename: impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Geometry'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: impl_disambiguate.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'DisambiguateImpl'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: impl_package_disambiguate.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'DisambiguateImplPackage'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:             {kind: 'PackageExpr', text: 'package'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'Name'},
// CHECK:STDOUT:           {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PackageExpr', text: 'package'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Interface'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
