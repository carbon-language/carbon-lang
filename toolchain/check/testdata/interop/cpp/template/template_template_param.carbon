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
// CHECK:STDOUT:   %pattern_type.2b7: type = pattern_type %TwoTemplates [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.50c: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.84b: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.50c) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.fcf: %T.as_type.as.DefaultOrUnformed.impl.Op.type.84b = struct_value () [symbolic]
// CHECK:STDOUT:   %TwoTemplates.Op.type: type = fn_type @TwoTemplates.Op [concrete]
// CHECK:STDOUT:   %TwoTemplates.Op: %TwoTemplates.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e5: <witness> = custom_witness (%TwoTemplates.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.90a: %Default.type = facet_value %TwoTemplates, (%custom_witness.6e5) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.7cb: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.16f, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.90a) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %TwoTemplates, (%DefaultOrUnformed.impl_witness.7cb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.627: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.84b) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.fcf)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.16f = impl_witness_table (%Core.import_ref.627), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.2b7 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.2b7 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTemplates = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTemplates = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%TwoTemplates, (constants.%DefaultOrUnformed.impl_witness.7cb) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc7_38.1: %DefaultOrUnformed.type = converted constants.%TwoTemplates, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_38.1 [concrete = constants.%TwoTemplates]
// CHECK:STDOUT:   %.loc7_38.2: type = converted %.loc7_38.1, %as_type [concrete = constants.%TwoTemplates]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7_1: ref %TwoTemplates = splice_block file.%x.var [concrete = file.%x.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %TwoTemplates to %.loc7_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
