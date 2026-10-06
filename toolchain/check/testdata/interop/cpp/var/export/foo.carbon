// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/var/export/foo.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/var/export/foo.carbon

import Cpp;

//@dump-sem-ir-begin
var N: i32 = 123 as i32;
//@dump-sem-ir-end

inline Cpp '''
int F() {
    return Carbon::N;
}
''';

// CHECK:STDOUT: --- foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.e89: %pattern_type.6b6 = ref_binding_pattern N [concrete]
// CHECK:STDOUT:   %N.var_patt: %pattern_type.6b6 = var_pattern %N.patt.e89 [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e7f: <witness> = impl_witness imports.%As.impl_witness_table.d22, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.e7f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d02: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.453: type = fn_type_with_self_type %As.WithSelf.Convert.type.d02, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_123.2bf, %Int.as.Copy.impl.Op.4f6 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.05d: <bound method> = bound_method %int_123.2bf, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.d22 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %N.var: ref %i32 = var_storage %N.var_patt [concrete]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %N: ref %i32 = wrapper_binding N, %N.var [concrete = %N.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %N.patt: %pattern_type.6b6 = ref_binding_pattern N [concrete = constants.%N.patt.e89]
// CHECK:STDOUT:     %N.var_patt: %pattern_type.6b6 = var_pattern %N.patt [concrete = constants.%N.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_18.1: %.453 = impl_witness_access constants.%As.impl_witness.e7f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc16_18.1: <bound method> = bound_method %int_123, %impl.elem0.loc16_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc16_18.1: <specific function> = specific_function %impl.elem0.loc16_18.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_18.2: <bound method> = bound_method %int_123, %specific_fn.loc16_18.1 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc16_18.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc16_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc16_18.2: %i32 = converted %int_123, %.loc16_18.1 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %impl.elem0.loc16_18.2: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc16_18.3: <bound method> = bound_method %.loc16_18.2, %impl.elem0.loc16_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc16_18.2: <specific function> = specific_function %impl.elem0.loc16_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_18.4: <bound method> = bound_method %.loc16_18.2, %specific_fn.loc16_18.2 [concrete = constants.%bound_method.05d]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc16_18.4(%.loc16_18.2) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   assign file.%N.var, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
