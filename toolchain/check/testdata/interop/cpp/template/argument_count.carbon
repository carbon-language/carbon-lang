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
// CHECK:STDOUT:   %pattern_type.3df: type = pattern_type %TwoTypes [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.4ca: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.4ca) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.25a: %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c = struct_value () [symbolic]
// CHECK:STDOUT:   %TwoTypes.Op.type: type = fn_type @TwoTypes.Op [concrete]
// CHECK:STDOUT:   %TwoTypes.Op: %TwoTypes.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.213: <witness> = custom_witness (%TwoTypes.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.ba6: %Default.type = facet_value %TwoTypes, (%custom_witness.213) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.462: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.ba6) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %TwoTypes, (%DefaultOrUnformed.impl_witness.462) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dfa: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.47c) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.25a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.b72 = impl_witness_table (%Core.import_ref.dfa), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.3df = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.3df = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTypes = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTypes = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%TwoTypes, (constants.%DefaultOrUnformed.impl_witness.462) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc7_34.1: %DefaultOrUnformed.type = converted constants.%TwoTypes, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_34.1 [concrete = constants.%TwoTypes]
// CHECK:STDOUT:   %.loc7_34.2: type = converted %.loc7_34.1, %as_type [concrete = constants.%TwoTypes]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7_1: ref %TwoTypes = splice_block file.%x.var [concrete = file.%x.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %TwoTypes to %.loc7_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- pack.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TypePack.ef9a30.1: type = class_type @TypePack.1 [concrete]
// CHECK:STDOUT:   %pattern_type.9115dc.1: type = pattern_type %TypePack.ef9a30.1 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.4ca: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.4ca) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.25a: %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c = struct_value () [symbolic]
// CHECK:STDOUT:   %TypePack.Op.type.43d658.1: type = fn_type @TypePack.Op.1 [concrete]
// CHECK:STDOUT:   %TypePack.Op.cc70d5.1: %TypePack.Op.type.43d658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2132a4.1: <witness> = custom_witness (%TypePack.Op.cc70d5.1), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.5ed9a2.1: %Default.type = facet_value %TypePack.ef9a30.1, (%custom_witness.2132a4.1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.bf0e48.1: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.5ed9a2.1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.86aa39.1: %DefaultOrUnformed.type = facet_value %TypePack.ef9a30.1, (%DefaultOrUnformed.impl_witness.bf0e48.1) [concrete]
// CHECK:STDOUT:   %TypePack.ef9a30.2: type = class_type @TypePack.2 [concrete]
// CHECK:STDOUT:   %pattern_type.9115dc.2: type = pattern_type %TypePack.ef9a30.2 [concrete]
// CHECK:STDOUT:   %TypePack.Op.type.43d658.2: type = fn_type @TypePack.Op.2 [concrete]
// CHECK:STDOUT:   %TypePack.Op.cc70d5.2: %TypePack.Op.type.43d658.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2132a4.2: <witness> = custom_witness (%TypePack.Op.cc70d5.2), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.5ed9a2.2: %Default.type = facet_value %TypePack.ef9a30.2, (%custom_witness.2132a4.2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.bf0e48.2: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.5ed9a2.2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.86aa39.2: %DefaultOrUnformed.type = facet_value %TypePack.ef9a30.2, (%DefaultOrUnformed.impl_witness.bf0e48.2) [concrete]
// CHECK:STDOUT:   %TypePack.ef9a30.3: type = class_type @TypePack.3 [concrete]
// CHECK:STDOUT:   %pattern_type.9115dc.3: type = pattern_type %TypePack.ef9a30.3 [concrete]
// CHECK:STDOUT:   %TypePack.Op.type.43d658.3: type = fn_type @TypePack.Op.3 [concrete]
// CHECK:STDOUT:   %TypePack.Op.cc70d5.3: %TypePack.Op.type.43d658.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2132a4.3: <witness> = custom_witness (%TypePack.Op.cc70d5.3), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.5ed9a2.3: %Default.type = facet_value %TypePack.ef9a30.3, (%custom_witness.2132a4.3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.bf0e48.3: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.5ed9a2.3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.86aa39.3: %DefaultOrUnformed.type = facet_value %TypePack.ef9a30.3, (%DefaultOrUnformed.impl_witness.bf0e48.3) [concrete]
// CHECK:STDOUT:   %TypePack.ef9a30.4: type = class_type @TypePack.4 [concrete]
// CHECK:STDOUT:   %pattern_type.9115dc.4: type = pattern_type %TypePack.ef9a30.4 [concrete]
// CHECK:STDOUT:   %TypePack.Op.type.43d658.4: type = fn_type @TypePack.Op.4 [concrete]
// CHECK:STDOUT:   %TypePack.Op.cc70d5.4: %TypePack.Op.type.43d658.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2132a4.4: <witness> = custom_witness (%TypePack.Op.cc70d5.4), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.5ed9a2.4: %Default.type = facet_value %TypePack.ef9a30.4, (%custom_witness.2132a4.4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.bf0e48.4: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.5ed9a2.4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.86aa39.4: %DefaultOrUnformed.type = facet_value %TypePack.ef9a30.4, (%DefaultOrUnformed.impl_witness.bf0e48.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dfa: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.47c) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.25a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.b72 = impl_witness_table (%Core.import_ref.dfa), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a0.patt: %pattern_type.9115dc.1 = ref_binding_pattern a0 [concrete]
// CHECK:STDOUT:     %a0.var_patt: %pattern_type.9115dc.1 = var_pattern %a0.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a0.var: ref %TypePack.ef9a30.1 = var %a0.var_patt [concrete]
// CHECK:STDOUT:   %a0: ref %TypePack.ef9a30.1 = ref_binding a0, %a0.var [concrete = %a0.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a1.patt: %pattern_type.9115dc.2 = ref_binding_pattern a1 [concrete]
// CHECK:STDOUT:     %a1.var_patt: %pattern_type.9115dc.2 = var_pattern %a1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a1.var: ref %TypePack.ef9a30.2 = var %a1.var_patt [concrete]
// CHECK:STDOUT:   %a1: ref %TypePack.ef9a30.2 = ref_binding a1, %a1.var [concrete = %a1.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a2.patt: %pattern_type.9115dc.3 = ref_binding_pattern a2 [concrete]
// CHECK:STDOUT:     %a2.var_patt: %pattern_type.9115dc.3 = var_pattern %a2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a2.var: ref %TypePack.ef9a30.3 = var %a2.var_patt [concrete]
// CHECK:STDOUT:   %a2: ref %TypePack.ef9a30.3 = ref_binding a2, %a2.var [concrete = %a2.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a3.patt: %pattern_type.9115dc.4 = ref_binding_pattern a3 [concrete]
// CHECK:STDOUT:     %a3.var_patt: %pattern_type.9115dc.4 = var_pattern %a3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a3.var: ref %TypePack.ef9a30.4 = var %a3.var_patt [concrete]
// CHECK:STDOUT:   %a3: ref %TypePack.ef9a30.4 = ref_binding a3, %a3.var [concrete = %a3.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7: %DefaultOrUnformed.type = facet_value constants.%TypePack.ef9a30.1, (constants.%DefaultOrUnformed.impl_witness.bf0e48.1) [concrete = constants.%DefaultOrUnformed.facet.86aa39.1]
// CHECK:STDOUT:   %.loc7_23.1: %DefaultOrUnformed.type = converted constants.%TypePack.ef9a30.1, %DefaultOrUnformed.facet.loc7 [concrete = constants.%DefaultOrUnformed.facet.86aa39.1]
// CHECK:STDOUT:   %as_type.loc7: type = facet_access_type %.loc7_23.1 [concrete = constants.%TypePack.ef9a30.1]
// CHECK:STDOUT:   %.loc7_23.2: type = converted %.loc7_23.1, %as_type.loc7 [concrete = constants.%TypePack.ef9a30.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7_1: ref %TypePack.ef9a30.1 = splice_block file.%a0.var [concrete = file.%a0.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc7: init %TypePack.ef9a30.1 to %.loc7_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%a0.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc7
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8: %DefaultOrUnformed.type = facet_value constants.%TypePack.ef9a30.2, (constants.%DefaultOrUnformed.impl_witness.bf0e48.2) [concrete = constants.%DefaultOrUnformed.facet.86aa39.2]
// CHECK:STDOUT:   %.loc8_28.1: %DefaultOrUnformed.type = converted constants.%TypePack.ef9a30.2, %DefaultOrUnformed.facet.loc8 [concrete = constants.%DefaultOrUnformed.facet.86aa39.2]
// CHECK:STDOUT:   %as_type.loc8: type = facet_access_type %.loc8_28.1 [concrete = constants.%TypePack.ef9a30.2]
// CHECK:STDOUT:   %.loc8_28.2: type = converted %.loc8_28.1, %as_type.loc8 [concrete = constants.%TypePack.ef9a30.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_1: ref %TypePack.ef9a30.2 = splice_block file.%a1.var [concrete = file.%a1.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8: init %TypePack.ef9a30.2 to %.loc8_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%a1.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9: %DefaultOrUnformed.type = facet_value constants.%TypePack.ef9a30.3, (constants.%DefaultOrUnformed.impl_witness.bf0e48.3) [concrete = constants.%DefaultOrUnformed.facet.86aa39.3]
// CHECK:STDOUT:   %.loc9_35.1: %DefaultOrUnformed.type = converted constants.%TypePack.ef9a30.3, %DefaultOrUnformed.facet.loc9 [concrete = constants.%DefaultOrUnformed.facet.86aa39.3]
// CHECK:STDOUT:   %as_type.loc9: type = facet_access_type %.loc9_35.1 [concrete = constants.%TypePack.ef9a30.3]
// CHECK:STDOUT:   %.loc9_35.2: type = converted %.loc9_35.1, %as_type.loc9 [concrete = constants.%TypePack.ef9a30.3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_1: ref %TypePack.ef9a30.3 = splice_block file.%a2.var [concrete = file.%a2.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc9: init %TypePack.ef9a30.3 to %.loc9_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign file.%a2.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc9
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc10: %DefaultOrUnformed.type = facet_value constants.%TypePack.ef9a30.4, (constants.%DefaultOrUnformed.impl_witness.bf0e48.4) [concrete = constants.%DefaultOrUnformed.facet.86aa39.4]
// CHECK:STDOUT:   %.loc10_42.1: %DefaultOrUnformed.type = converted constants.%TypePack.ef9a30.4, %DefaultOrUnformed.facet.loc10 [concrete = constants.%DefaultOrUnformed.facet.86aa39.4]
// CHECK:STDOUT:   %as_type.loc10: type = facet_access_type %.loc10_42.1 [concrete = constants.%TypePack.ef9a30.4]
// CHECK:STDOUT:   %.loc10_42.2: type = converted %.loc10_42.1, %as_type.loc10 [concrete = constants.%TypePack.ef9a30.4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_1: ref %TypePack.ef9a30.4 = splice_block file.%a3.var [concrete = file.%a3.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc10: init %TypePack.ef9a30.4 to %.loc10_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.4()
// CHECK:STDOUT:   assign file.%a3.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
