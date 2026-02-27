// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/argument_count.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/argument_count.carbon

// --- templates.h

struct A {};
struct B {};

template<typename T, typename U> struct TwoTypes {};

template<typename ...T> struct TypePack {};

template<typename, typename> concept True = true;

// FixedSizePack<T1, T2, ..., Tn>::Inner has a template parameter pack of fixed
// size `n`, because it's expanded to an arity of exactly `n` when instantiating
// the enclosing template, because the inner pack mentions the outer one.
template<typename ...T> struct FixedSizePack {
  template<True<T> ...U> struct Inner {};
};

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

// CHECK:STDERR: fail_too_many_arguments.carbon:[[@LINE+8]]:40: error: too many template arguments for class template 'TwoTypes' [CppInteropParseError]
// CHECK:STDERR:    14 | var x: Cpp.TwoTypes(Cpp.A, Cpp.A, Cpp.A);
// CHECK:STDERR:       |                                      ~~^
// CHECK:STDERR: fail_too_many_arguments.carbon:[[@LINE-5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./templates.h:5:41: note: template is declared here [CppInteropParseNote]
// CHECK:STDERR:     5 | template<typename T, typename U> struct TwoTypes {};
// CHECK:STDERR:       | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        ^
// CHECK:STDERR:
var x: Cpp.TwoTypes(Cpp.A, Cpp.A, Cpp.A);

// --- pack.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
var a0: Cpp.TypePack();
var a1: Cpp.TypePack(Cpp.A);
var a2: Cpp.TypePack(Cpp.A, Cpp.A);
var a3: Cpp.TypePack(Cpp.A, Cpp.A, Cpp.A);
//@dump-sem-ir-end

// --- fixed_size_pack.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var x: Cpp.FixedSizePack(Cpp.A, Cpp.A).Inner(Cpp.B, Cpp.B);

// --- fail_fixed_size_pack_wrong_size.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_fixed_size_pack_wrong_size.carbon:[[@LINE+8]]:57: error: too few template arguments for class template 'Inner' [CppInteropParseError]
// CHECK:STDERR:    14 | var too_few: Cpp.FixedSizePack(Cpp.A, Cpp.A).Inner(Cpp.B);
// CHECK:STDERR:       |                                                         ^
// CHECK:STDERR: fail_fixed_size_pack_wrong_size.carbon:[[@LINE-5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./templates.h:15:33: note: template is declared here [CppInteropParseNote]
// CHECK:STDERR:    15 |   template<True<T> ...U> struct Inner {};
// CHECK:STDERR:       |   ~~~~~~~~~~~~~~~~~~~~~~        ^
// CHECK:STDERR:
var too_few: Cpp.FixedSizePack(Cpp.A, Cpp.A).Inner(Cpp.B);

// CHECK:STDERR: fail_fixed_size_pack_wrong_size.carbon:[[@LINE+8]]:72: error: too many template arguments for class template 'Inner' [CppInteropParseError]
// CHECK:STDERR:    24 | var too_many: Cpp.FixedSizePack(Cpp.A, Cpp.A).Inner(Cpp.B, Cpp.B, Cpp.B);
// CHECK:STDERR:       |                                                                      ~~^
// CHECK:STDERR: fail_fixed_size_pack_wrong_size.carbon:[[@LINE-15]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./templates.h:15:33: note: template is declared here [CppInteropParseNote]
// CHECK:STDERR:    15 |   template<True<T> ...U> struct Inner {};
// CHECK:STDERR:       |   ~~~~~~~~~~~~~~~~~~~~~~        ^
// CHECK:STDERR:
var too_many: Cpp.FixedSizePack(Cpp.A, Cpp.A).Inner(Cpp.B, Cpp.B, Cpp.B);

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TwoTypes: type = class_type @TwoTypes [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %TwoTypes [concrete]
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
// CHECK:STDOUT: --- pack.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TypePack.49edc4.1: type = class_type @TypePack.1 [concrete]
// CHECK:STDOUT:   %pattern_type.9db866.1: type = pattern_type %TypePack.49edc4.1 [concrete]
// CHECK:STDOUT:   %TypePack.49edc4.2: type = class_type @TypePack.2 [concrete]
// CHECK:STDOUT:   %pattern_type.9db866.2: type = pattern_type %TypePack.49edc4.2 [concrete]
// CHECK:STDOUT:   %TypePack.49edc4.3: type = class_type @TypePack.3 [concrete]
// CHECK:STDOUT:   %pattern_type.9db866.3: type = pattern_type %TypePack.49edc4.3 [concrete]
// CHECK:STDOUT:   %TypePack.49edc4.4: type = class_type @TypePack.4 [concrete]
// CHECK:STDOUT:   %pattern_type.9db866.4: type = pattern_type %TypePack.49edc4.4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a0.patt: %pattern_type.9db866.1 = ref_binding_pattern a0 [concrete]
// CHECK:STDOUT:     %a0.var_patt: %pattern_type.9db866.1 = var_pattern %a0.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a0.var: ref %TypePack.49edc4.1 = var %a0.var_patt [concrete]
// CHECK:STDOUT:   %a0: ref %TypePack.49edc4.1 = ref_binding a0, %a0.var [concrete = %a0.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a1.patt: %pattern_type.9db866.2 = ref_binding_pattern a1 [concrete]
// CHECK:STDOUT:     %a1.var_patt: %pattern_type.9db866.2 = var_pattern %a1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a1.var: ref %TypePack.49edc4.2 = var %a1.var_patt [concrete]
// CHECK:STDOUT:   %a1: ref %TypePack.49edc4.2 = ref_binding a1, %a1.var [concrete = %a1.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a2.patt: %pattern_type.9db866.3 = ref_binding_pattern a2 [concrete]
// CHECK:STDOUT:     %a2.var_patt: %pattern_type.9db866.3 = var_pattern %a2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a2.var: ref %TypePack.49edc4.3 = var %a2.var_patt [concrete]
// CHECK:STDOUT:   %a2: ref %TypePack.49edc4.3 = ref_binding a2, %a2.var [concrete = %a2.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a3.patt: %pattern_type.9db866.4 = ref_binding_pattern a3 [concrete]
// CHECK:STDOUT:     %a3.var_patt: %pattern_type.9db866.4 = var_pattern %a3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a3.var: ref %TypePack.49edc4.4 = var %a3.var_patt [concrete]
// CHECK:STDOUT:   %a3: ref %TypePack.49edc4.4 = ref_binding a3, %a3.var [concrete = %a3.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
