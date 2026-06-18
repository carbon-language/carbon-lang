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
// CHECK:STDOUT:   %pattern_type.83a: type = pattern_type %TwoTypes [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.83a = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.83a = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %TwoTypes.Op.type: type = fn_type @TwoTypes.Op [concrete]
// CHECK:STDOUT:   %TwoTypes.Op: %TwoTypes.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%TwoTypes.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.49c: %Default.type = facet_value %TwoTypes, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.07d: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.670, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.49c) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %TwoTypes, (%DefaultOrUnformed.impl_witness.07d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.670 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.83a = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.83a = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTypes = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTypes = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%TwoTypes, (constants.%DefaultOrUnformed.impl_witness.07d) [concrete = constants.%DefaultOrUnformed.facet]
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
// CHECK:STDOUT:   %TypePack.78f: type = class_type @TypePack.1 [concrete]
// CHECK:STDOUT:   %pattern_type.a64: type = pattern_type %TypePack.78f [concrete]
// CHECK:STDOUT:   %a0.patt: %pattern_type.a64 = ref_binding_pattern a0 [concrete]
// CHECK:STDOUT:   %a0.var_patt: %pattern_type.a64 = var_pattern %a0.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %TypePack.Op.type.83af6e.1: type = fn_type @TypePack.Op.1 [concrete]
// CHECK:STDOUT:   %TypePack.Op.4dca31.1: %TypePack.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b40fb.1: <witness> = custom_witness (%TypePack.Op.4dca31.1), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.7a3: %Default.type = facet_value %TypePack.78f, (%custom_witness.9b40fb.1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.6f4: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.670, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.7a3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.be2: %DefaultOrUnformed.type = facet_value %TypePack.78f, (%DefaultOrUnformed.impl_witness.6f4) [concrete]
// CHECK:STDOUT:   %TypePack.0bc: type = class_type @TypePack.2 [concrete]
// CHECK:STDOUT:   %pattern_type.20e: type = pattern_type %TypePack.0bc [concrete]
// CHECK:STDOUT:   %a1.patt: %pattern_type.20e = ref_binding_pattern a1 [concrete]
// CHECK:STDOUT:   %a1.var_patt: %pattern_type.20e = var_pattern %a1.patt [concrete]
// CHECK:STDOUT:   %TypePack.Op.type.83af6e.2: type = fn_type @TypePack.Op.2 [concrete]
// CHECK:STDOUT:   %TypePack.Op.4dca31.2: %TypePack.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b40fb.2: <witness> = custom_witness (%TypePack.Op.4dca31.2), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.2dd: %Default.type = facet_value %TypePack.0bc, (%custom_witness.9b40fb.2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.8e2: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.670, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.2dd) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.777: %DefaultOrUnformed.type = facet_value %TypePack.0bc, (%DefaultOrUnformed.impl_witness.8e2) [concrete]
// CHECK:STDOUT:   %TypePack.0ec: type = class_type @TypePack.3 [concrete]
// CHECK:STDOUT:   %pattern_type.c25: type = pattern_type %TypePack.0ec [concrete]
// CHECK:STDOUT:   %a2.patt: %pattern_type.c25 = ref_binding_pattern a2 [concrete]
// CHECK:STDOUT:   %a2.var_patt: %pattern_type.c25 = var_pattern %a2.patt [concrete]
// CHECK:STDOUT:   %TypePack.Op.type.83af6e.3: type = fn_type @TypePack.Op.3 [concrete]
// CHECK:STDOUT:   %TypePack.Op.4dca31.3: %TypePack.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b40fb.3: <witness> = custom_witness (%TypePack.Op.4dca31.3), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.3ab: %Default.type = facet_value %TypePack.0ec, (%custom_witness.9b40fb.3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.873: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.670, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.3ab) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.967: %DefaultOrUnformed.type = facet_value %TypePack.0ec, (%DefaultOrUnformed.impl_witness.873) [concrete]
// CHECK:STDOUT:   %TypePack.186: type = class_type @TypePack.4 [concrete]
// CHECK:STDOUT:   %pattern_type.678: type = pattern_type %TypePack.186 [concrete]
// CHECK:STDOUT:   %a3.patt: %pattern_type.678 = ref_binding_pattern a3 [concrete]
// CHECK:STDOUT:   %a3.var_patt: %pattern_type.678 = var_pattern %a3.patt [concrete]
// CHECK:STDOUT:   %TypePack.Op.type.83af6e.4: type = fn_type @TypePack.Op.4 [concrete]
// CHECK:STDOUT:   %TypePack.Op.4dca31.4: %TypePack.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b40fb.4: <witness> = custom_witness (%TypePack.Op.4dca31.4), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.3c1: %Default.type = facet_value %TypePack.186, (%custom_witness.9b40fb.4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.3f9: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.670, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.3c1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.59c: %DefaultOrUnformed.type = facet_value %TypePack.186, (%DefaultOrUnformed.impl_witness.3f9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.670 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a0.patt: %pattern_type.a64 = ref_binding_pattern a0 [concrete = constants.%a0.patt]
// CHECK:STDOUT:     %a0.var_patt: %pattern_type.a64 = var_pattern %a0.patt [concrete = constants.%a0.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a0.var: ref %TypePack.78f = var %a0.var_patt [concrete]
// CHECK:STDOUT:   %a0: ref %TypePack.78f = ref_binding a0, %a0.var [concrete = %a0.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a1.patt: %pattern_type.20e = ref_binding_pattern a1 [concrete = constants.%a1.patt]
// CHECK:STDOUT:     %a1.var_patt: %pattern_type.20e = var_pattern %a1.patt [concrete = constants.%a1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a1.var: ref %TypePack.0bc = var %a1.var_patt [concrete]
// CHECK:STDOUT:   %a1: ref %TypePack.0bc = ref_binding a1, %a1.var [concrete = %a1.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a2.patt: %pattern_type.c25 = ref_binding_pattern a2 [concrete = constants.%a2.patt]
// CHECK:STDOUT:     %a2.var_patt: %pattern_type.c25 = var_pattern %a2.patt [concrete = constants.%a2.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a2.var: ref %TypePack.0ec = var %a2.var_patt [concrete]
// CHECK:STDOUT:   %a2: ref %TypePack.0ec = ref_binding a2, %a2.var [concrete = %a2.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a3.patt: %pattern_type.678 = ref_binding_pattern a3 [concrete = constants.%a3.patt]
// CHECK:STDOUT:     %a3.var_patt: %pattern_type.678 = var_pattern %a3.patt [concrete = constants.%a3.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a3.var: ref %TypePack.186 = var %a3.var_patt [concrete]
// CHECK:STDOUT:   %a3: ref %TypePack.186 = ref_binding a3, %a3.var [concrete = %a3.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7: %DefaultOrUnformed.type = facet_value constants.%TypePack.78f, (constants.%DefaultOrUnformed.impl_witness.6f4) [concrete = constants.%DefaultOrUnformed.facet.be2]
// CHECK:STDOUT:   %.loc7_23.1: %DefaultOrUnformed.type = converted constants.%TypePack.78f, %DefaultOrUnformed.facet.loc7 [concrete = constants.%DefaultOrUnformed.facet.be2]
// CHECK:STDOUT:   %as_type.loc7: type = facet_access_type %.loc7_23.1 [concrete = constants.%TypePack.78f]
// CHECK:STDOUT:   %.loc7_23.2: type = converted %.loc7_23.1, %as_type.loc7 [concrete = constants.%TypePack.78f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7_1: ref %TypePack.78f = splice_block file.%a0.var [concrete = file.%a0.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc7: init %TypePack.78f to %.loc7_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%a0.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc7
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8: %DefaultOrUnformed.type = facet_value constants.%TypePack.0bc, (constants.%DefaultOrUnformed.impl_witness.8e2) [concrete = constants.%DefaultOrUnformed.facet.777]
// CHECK:STDOUT:   %.loc8_28.1: %DefaultOrUnformed.type = converted constants.%TypePack.0bc, %DefaultOrUnformed.facet.loc8 [concrete = constants.%DefaultOrUnformed.facet.777]
// CHECK:STDOUT:   %as_type.loc8: type = facet_access_type %.loc8_28.1 [concrete = constants.%TypePack.0bc]
// CHECK:STDOUT:   %.loc8_28.2: type = converted %.loc8_28.1, %as_type.loc8 [concrete = constants.%TypePack.0bc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_1: ref %TypePack.0bc = splice_block file.%a1.var [concrete = file.%a1.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8: init %TypePack.0bc to %.loc8_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%a1.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9: %DefaultOrUnformed.type = facet_value constants.%TypePack.0ec, (constants.%DefaultOrUnformed.impl_witness.873) [concrete = constants.%DefaultOrUnformed.facet.967]
// CHECK:STDOUT:   %.loc9_35.1: %DefaultOrUnformed.type = converted constants.%TypePack.0ec, %DefaultOrUnformed.facet.loc9 [concrete = constants.%DefaultOrUnformed.facet.967]
// CHECK:STDOUT:   %as_type.loc9: type = facet_access_type %.loc9_35.1 [concrete = constants.%TypePack.0ec]
// CHECK:STDOUT:   %.loc9_35.2: type = converted %.loc9_35.1, %as_type.loc9 [concrete = constants.%TypePack.0ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_1: ref %TypePack.0ec = splice_block file.%a2.var [concrete = file.%a2.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc9: init %TypePack.0ec to %.loc9_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign file.%a2.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc9
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc10: %DefaultOrUnformed.type = facet_value constants.%TypePack.186, (constants.%DefaultOrUnformed.impl_witness.3f9) [concrete = constants.%DefaultOrUnformed.facet.59c]
// CHECK:STDOUT:   %.loc10_42.1: %DefaultOrUnformed.type = converted constants.%TypePack.186, %DefaultOrUnformed.facet.loc10 [concrete = constants.%DefaultOrUnformed.facet.59c]
// CHECK:STDOUT:   %as_type.loc10: type = facet_access_type %.loc10_42.1 [concrete = constants.%TypePack.186]
// CHECK:STDOUT:   %.loc10_42.2: type = converted %.loc10_42.1, %as_type.loc10 [concrete = constants.%TypePack.186]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_1: ref %TypePack.186 = splice_block file.%a3.var [concrete = file.%a3.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc10: init %TypePack.186 to %.loc10_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.4()
// CHECK:STDOUT:   assign file.%a3.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
