// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/template_template_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/template_template_param.carbon

// --- templates.h

template<typename> struct A {};
template<typename> struct B {};
template<typename, typename> struct C {};

template<template<typename> typename, template<typename> typename>
struct TwoTemplates {};

// --- valid.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
var x: Cpp.TwoTemplates(Cpp.A, Cpp.B);
//@dump-sem-ir-end

// --- fail_template_mismatch.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_template_mismatch.carbon:[[@LINE+15]]:35: error: too few template arguments for class template 'C' [CppInteropParseError]
// CHECK:STDERR:    22 | var x: Cpp.TwoTemplates(Cpp.A, Cpp.C);
// CHECK:STDERR:       |                                   ^
// CHECK:STDERR: fail_template_mismatch.carbon:[[@LINE+12]]:35: note: template template argument has different template parameters than its corresponding template template parameter [CppInteropParseNote]
// CHECK:STDERR:    22 | var x: Cpp.TwoTemplates(Cpp.A, Cpp.C);
// CHECK:STDERR:       |                                   ^
// CHECK:STDERR: fail_template_mismatch.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./templates.h:6:66: note: previous template template parameter is here [CppInteropParseNote]
// CHECK:STDERR:     6 | template<template<typename> typename, template<typename> typename>
// CHECK:STDERR:       |                                       ~~~~~~~~~~~~~~~~~~         ^
// CHECK:STDERR: fail_template_mismatch.carbon:[[@LINE-13]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./templates.h:4:37: note: template is declared here [CppInteropParseNote]
// CHECK:STDERR:     4 | template<typename, typename> struct C {};
// CHECK:STDERR:       | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~        ^
// CHECK:STDERR:
var x: Cpp.TwoTemplates(Cpp.A, Cpp.C);
//@dump-sem-ir-end

// --- fail_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

class X {}

// CHECK:STDERR: fail_type_argument.carbon:[[@LINE+4]]:32: error: template argument for template template parameter must be a class template or type alias template [CppInteropParseError]
// CHECK:STDERR:    12 | var x: Cpp.TwoTemplates(Cpp.A, X);
// CHECK:STDERR:       |                                ^
// CHECK:STDERR:
var x: Cpp.TwoTemplates(Cpp.A, X);

// --- fail_non_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_non_type_argument.carbon:[[@LINE+4]]:32: error: template argument for template template parameter must be a class template or type alias template [CppInteropParseError]
// CHECK:STDERR:    10 | var x: Cpp.TwoTemplates(Cpp.A, true);
// CHECK:STDERR:       |                                ^
// CHECK:STDERR:
var x: Cpp.TwoTemplates(Cpp.A, true);

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TwoTemplates: type = class_type @TwoTemplates [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %TwoTemplates [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTemplates = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTemplates = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_template_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: <error> = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref <error> = var %x.var_patt [concrete = <error>]
// CHECK:STDOUT:   %x: ref <error> = ref_binding x, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
