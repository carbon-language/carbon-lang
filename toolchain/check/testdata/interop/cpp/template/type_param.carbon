// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/type_param.carbon

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

// --- char.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
var x: Cpp.TwoTypes(char, char);
//@dump-sem-ir-end

// --- fail_non_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_non_type_argument.carbon:[[@LINE+7]]:28: error: cannot implicitly convert non-type value of type `bool` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var x: Cpp.TwoTypes(Cpp.A, true);
// CHECK:STDERR:                            ^~~~
// CHECK:STDERR: fail_non_type_argument.carbon:[[@LINE+4]]:28: note: type `bool` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var x: Cpp.TwoTypes(Cpp.A, true);
// CHECK:STDERR:                            ^~~~
// CHECK:STDERR:
var x: Cpp.TwoTypes(Cpp.A, true);

// --- fail_unsupported_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_unsupported_type.carbon:[[@LINE+4]]:28: error: semantics TODO: `unsupported type used as template argument` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoTypes(Cpp.A, {});
// CHECK:STDERR:                            ^~
// CHECK:STDERR:
var x: Cpp.TwoTypes(Cpp.A, {});

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TwoTypes: type = class_type @TwoTypes [concrete]
// CHECK:STDOUT:   %pattern_type.bc0: type = pattern_type %TwoTypes [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.e5a: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.e5a) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.eab: %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d = struct_value () [symbolic]
// CHECK:STDOUT:   %TwoTypes.Op.type: type = fn_type @TwoTypes.Op [concrete]
// CHECK:STDOUT:   %TwoTypes.Op: %TwoTypes.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26: <witness> = custom_witness (%TwoTypes.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.3d9: %Default.type = facet_value %TwoTypes, (%custom_witness.b26) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.d7b: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.3d9) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %TwoTypes, (%DefaultOrUnformed.impl_witness.d7b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a03: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.38d) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.eab)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.48a = impl_witness_table (%Core.import_ref.a03), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.bc0 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.bc0 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTypes = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTypes = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%TwoTypes, (constants.%DefaultOrUnformed.impl_witness.d7b) [concrete = constants.%DefaultOrUnformed.facet]
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
// CHECK:STDOUT: --- char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TwoTypes: type = class_type @TwoTypes [concrete]
// CHECK:STDOUT:   %pattern_type.6c8: type = pattern_type %TwoTypes [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.e5a: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.e5a) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.eab: %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d = struct_value () [symbolic]
// CHECK:STDOUT:   %TwoTypes.Op.type: type = fn_type @TwoTypes.Op [concrete]
// CHECK:STDOUT:   %TwoTypes.Op: %TwoTypes.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26: <witness> = custom_witness (%TwoTypes.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.e79: %Default.type = facet_value %TwoTypes, (%custom_witness.b26) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.321: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.e79) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %TwoTypes, (%DefaultOrUnformed.impl_witness.321) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a03: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.38d) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.eab)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.48a = impl_witness_table (%Core.import_ref.a03), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6c8 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6c8 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoTypes = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoTypes = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%TwoTypes, (constants.%DefaultOrUnformed.impl_witness.321) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc7_32.1: %DefaultOrUnformed.type = converted constants.%TwoTypes, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_32.1 [concrete = constants.%TwoTypes]
// CHECK:STDOUT:   %.loc7_32.2: type = converted %.loc7_32.1, %as_type [concrete = constants.%TwoTypes]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7_1: ref %TwoTypes = splice_block file.%x.var [concrete = file.%x.var] {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %TwoTypes to %.loc7_1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
