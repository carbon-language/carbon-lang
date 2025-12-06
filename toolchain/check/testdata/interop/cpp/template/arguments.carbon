// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/arguments.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/arguments.carbon

// --- templates.h

struct A {};
struct B {};

template<typename T, typename U> struct TwoTypes {};

// --- valid.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
var x: Cpp.TwoTypes(Cpp.A, Cpp.B);
//@dump-sem-ir-end

// --- fail_too_few_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_too_few_arguments.carbon:[[@LINE+8]]:26: error: too few template arguments for class template 'TwoTypes' [CppInteropParseError]
// CHECK:STDERR:    14 | var x: Cpp.TwoTypes(Cpp.A);
// CHECK:STDERR:       |                          ^
// CHECK:STDERR: fail_too_few_arguments.carbon:[[@LINE-5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./templates.h:5:41: note: template is declared here [CppInteropParseNote]
// CHECK:STDERR:     5 | template<typename T, typename U> struct TwoTypes {};
// CHECK:STDERR:       | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        ^
// CHECK:STDERR:
var x: Cpp.TwoTypes(Cpp.A);

// --- fail_too_many_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_too_many_arguments.carbon:[[@LINE+4]]:8: error: semantics TODO: `diagnose too many arguments in call to template` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoTypes(Cpp.A, Cpp.A, Cpp.A);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var x: Cpp.TwoTypes(Cpp.A, Cpp.A, Cpp.A);

// --- fail_non_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_non_type_argument.carbon:[[@LINE+7]]:28: error: cannot implicitly convert non-type value of type `bool` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var x: Cpp.TwoTypes(Cpp.A, true);
// CHECK:STDERR:                            ^~~~
// CHECK:STDERR: fail_non_type_argument.carbon:[[@LINE+4]]:28: note: type `bool` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: Cpp.TwoTypes(Cpp.A, true);
// CHECK:STDERR:                            ^~~~
// CHECK:STDERR:
var x: Cpp.TwoTypes(Cpp.A, true);

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TwoTypes: type = class_type @TwoTypes [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %TwoTypes [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTypes = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTypes = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
