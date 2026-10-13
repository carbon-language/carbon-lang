// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: {.a: i32} = {.a = 0};
var b_ref: {.a: {.b: i32, .c: (i32,)}, .d: i32} =
    {.a = {.b = 0, .c = (0,)}, .d = 0};

class C(S: {.a: i32, .b: i32}) {}
fn F() -> C({.a = 1, .b = 2});

// --- implicit.impl.carbon

impl package Implicit;

//@dump-sem-ir-begin
var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;
var c: C({.a = 1, .b = 2}) = F();
//@dump-sem-ir-end

// --- fail_bad_type.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: struct `{.a: i32, .b: i32}` has no field named `c` [StructInitUnexpectedFieldInLiteral]
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:1: in import [InImport]
// CHECK:STDERR: implicit.carbon:8:9: note: initializing generic parameter `S` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(S: {.a: i32, .b: i32}) {}
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C({.a = 1, .b = 2})` to `C({.a = 3, .b = 4})` [ConversionFailure]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C({.a = 1, .b = 2})` does not implement interface `Core.ImplicitAs(C({.a = 3, .b = 4}))` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.a = 3, .b = 4}) = F();

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let value: {.x: i32};
}

class C {}

impl C as I where .value = {.x = 1} {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> {.x: i32} {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.8d6: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.8d6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.8d6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.8d6 = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.8d6 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %.7fb: ref %i32 = struct_access imports.%a_ref.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.135: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.58f: %tuple.type.135 = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.489: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.489} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.940: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.a7c: type = pattern_type %struct_type.a.d.940 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.a7c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.a7c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %b_ref.patt: %pattern_type.a7c = ref_binding_pattern b_ref [concrete]
// CHECK:STDOUT:   %b_ref.var_patt: %pattern_type.a7c = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   %.827: ref %struct_type.b.c = struct_access imports.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %.fdf: ref %struct_type.b.c = struct_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %.ddc: ref %i32 = struct_access %.827, element0 [concrete]
// CHECK:STDOUT:   %.fc2: ref %i32 = struct_access %.fdf, element0 [concrete]
// CHECK:STDOUT:   %.237: ref %tuple.type.489 = struct_access %.827, element1 [concrete]
// CHECK:STDOUT:   %.5e0: ref %tuple.type.489 = struct_access %.fdf, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.a88: ref %i32 = tuple_access %.237, element0 [concrete]
// CHECK:STDOUT:   %.6f7: ref %i32 = struct_access imports.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %.809: ref %i32 = struct_access file.%b.var, element1 [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.1e8: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.117: %struct_type.a.b.1e8 = struct_value (%int_1.874, %int_2.92b) [concrete]
// CHECK:STDOUT:   %C.fb5: type = class_type @C, @C(%struct.117) [concrete]
// CHECK:STDOUT:   %pattern_type.ad0: type = pattern_type %C.fb5 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.ad0 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.ad0 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %struct_type.a = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Implicit.b_ref: ref %struct_type.a.d.940 = import_ref Implicit//default, b_ref, loaded [concrete = %b_ref.var]
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Implicit.import_ref.2d4: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Implicit//default, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Implicit.import_ref.2d4), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %struct_type.a = var_storage constants.%a_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %b_ref.var: ref %struct_type.a.d.940 = var_storage constants.%b_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %a.var: ref %struct_type.a = var_storage %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a = wrapper_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8d6 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.8d6 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %struct_type.a.d.940 = var_storage %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_43: type = splice_block %struct_type.a.d [concrete = constants.%struct_type.a.d.940] {
// CHECK:STDOUT:     %i32.loc6_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_32.1: %tuple.type.135 = tuple_literal (%i32.loc6_28) [concrete = constants.%tuple.58f]
// CHECK:STDOUT:     %.loc6_32.2: type = converted %.loc6_32.1, constants.%tuple.type.489 [concrete = constants.%tuple.type.489]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.489} [concrete = constants.%struct_type.b.c]
// CHECK:STDOUT:     %i32.loc6_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.d: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete = constants.%struct_type.a.d.940]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %struct_type.a.d.940 = wrapper_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a7c = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.a7c = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.fb5 = var_storage %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_26.1: type = splice_block %C [concrete = constants.%C.fb5] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc7_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:     %impl.elem0.loc7_25.1: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc7_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_25.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:     %specific_fn.loc7_25.1: <specific function> = specific_function %impl.elem0.loc7_25.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_25.2: <bound method> = bound_method %int_1, %specific_fn.loc7_25.1 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.1: init %i32 = call %bound_method.loc7_25.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:     %.loc7_25.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:     %.loc7_25.3: %i32 = converted %int_1, %.loc7_25.2 [concrete = constants.%int_1.874]
// CHECK:STDOUT:     %impl.elem0.loc7_25.2: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc7_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc7_25.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn.loc7_25.2: <specific function> = specific_function %impl.elem0.loc7_25.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_25.4: <bound method> = bound_method %int_2, %specific_fn.loc7_25.2 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.2: init %i32 = call %bound_method.loc7_25.4(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:     %.loc7_25.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.2 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:     %.loc7_25.5: %i32 = converted %int_2, %.loc7_25.4 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:     %struct: %struct_type.a.b.1e8 = struct_value (%.loc7_25.3, %.loc7_25.5) [concrete = constants.%struct.117]
// CHECK:STDOUT:     %.loc7_26.2: %struct_type.a.b.1e8 = converted %.loc7_25.1, %struct [concrete = constants.%struct.117]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.117) [concrete = constants.%C.fb5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.fb5 = wrapper_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.ad0 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.ad0 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %struct_type.a = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %.loc5_20.1: ref %i32 = struct_access %a_ref.ref, element0 [concrete = constants.%.7fb]
// CHECK:STDOUT:   %.loc5_20.2: %i32 = acquire_value %.loc5_20.1
// CHECK:STDOUT:   %impl.elem0.loc5: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc5_20.1: <bound method> = bound_method %.loc5_20.2, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_20.2: <bound method> = bound_method %.loc5_20.2, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc5: init %i32 = call %bound_method.loc5_20.2(%.loc5_20.2)
// CHECK:STDOUT:   %.loc5_20.3: init %struct_type.a to file.%a.var = struct_init (%Int.as.Copy.impl.Op.call.loc5)
// CHECK:STDOUT:   %.loc5_1: init %struct_type.a = converted %a_ref.ref, %.loc5_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc5_1
// CHECK:STDOUT:   %b_ref.ref: ref %struct_type.a.d.940 = name_ref b_ref, imports.%Implicit.b_ref [concrete = imports.%b_ref.var]
// CHECK:STDOUT:   %.loc6_47.1: ref %struct_type.b.c = struct_access %b_ref.ref, element0 [concrete = constants.%.827]
// CHECK:STDOUT:   %.loc6_47.2: ref %i32 = struct_access %.loc6_47.1, element0 [concrete = constants.%.ddc]
// CHECK:STDOUT:   %.loc6_47.3: %i32 = acquire_value %.loc6_47.2
// CHECK:STDOUT:   %impl.elem0.loc6_47.1: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc6_47.1: <bound method> = bound_method %.loc6_47.3, %impl.elem0.loc6_47.1
// CHECK:STDOUT:   %specific_fn.loc6_47.1: <specific function> = specific_function %impl.elem0.loc6_47.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_47.2: <bound method> = bound_method %.loc6_47.3, %specific_fn.loc6_47.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_47.1: init %i32 = call %bound_method.loc6_47.2(%.loc6_47.3)
// CHECK:STDOUT:   %.loc6_47.4: ref %struct_type.b.c = struct_access file.%b.var, element0 [concrete = constants.%.fdf]
// CHECK:STDOUT:   %.loc6_47.5: ref %i32 = struct_access %.loc6_47.4, element0 [concrete = constants.%.fc2]
// CHECK:STDOUT:   %.loc6_47.6: init %i32 to %.loc6_47.5 = in_place_init %Int.as.Copy.impl.Op.call.loc6_47.1
// CHECK:STDOUT:   %.loc6_47.7: ref %tuple.type.489 = struct_access %.loc6_47.1, element1 [concrete = constants.%.237]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc6_47.7, element0 [concrete = constants.%tuple.elem0.a88]
// CHECK:STDOUT:   %.loc6_47.8: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc6_47.2: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc6_47.3: <bound method> = bound_method %.loc6_47.8, %impl.elem0.loc6_47.2
// CHECK:STDOUT:   %specific_fn.loc6_47.2: <specific function> = specific_function %impl.elem0.loc6_47.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_47.4: <bound method> = bound_method %.loc6_47.8, %specific_fn.loc6_47.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_47.2: init %i32 = call %bound_method.loc6_47.4(%.loc6_47.8)
// CHECK:STDOUT:   %.loc6_47.9: ref %tuple.type.489 = struct_access %.loc6_47.4, element1 [concrete = constants.%.5e0]
// CHECK:STDOUT:   %.loc6_47.10: init %tuple.type.489 to %.loc6_47.9 = tuple_init (%Int.as.Copy.impl.Op.call.loc6_47.2)
// CHECK:STDOUT:   %.loc6_47.11: init %tuple.type.489 = converted %.loc6_47.7, %.loc6_47.10
// CHECK:STDOUT:   %.loc6_47.12: init %tuple.type.489 to %.loc6_47.9 = in_place_init %.loc6_47.11
// CHECK:STDOUT:   %.loc6_47.13: init %struct_type.b.c to %.loc6_47.4 = struct_init (%.loc6_47.6, %.loc6_47.12)
// CHECK:STDOUT:   %.loc6_47.14: init %struct_type.b.c = converted %.loc6_47.1, %.loc6_47.13
// CHECK:STDOUT:   %.loc6_47.15: ref %i32 = struct_access %b_ref.ref, element1 [concrete = constants.%.6f7]
// CHECK:STDOUT:   %.loc6_47.16: %i32 = acquire_value %.loc6_47.15
// CHECK:STDOUT:   %impl.elem0.loc6_47.3: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc6_47.5: <bound method> = bound_method %.loc6_47.16, %impl.elem0.loc6_47.3
// CHECK:STDOUT:   %specific_fn.loc6_47.3: <specific function> = specific_function %impl.elem0.loc6_47.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_47.6: <bound method> = bound_method %.loc6_47.16, %specific_fn.loc6_47.3
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_47.3: init %i32 = call %bound_method.loc6_47.6(%.loc6_47.16)
// CHECK:STDOUT:   %.loc6_47.17: ref %i32 = struct_access file.%b.var, element1 [concrete = constants.%.809]
// CHECK:STDOUT:   %.loc6_47.18: init %i32 to %.loc6_47.17 = in_place_init %Int.as.Copy.impl.Op.call.loc6_47.3
// CHECK:STDOUT:   %.loc6_47.19: init %struct_type.a.d.940 to file.%b.var = struct_init (%.loc6_47.14, %.loc6_47.18)
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.d.940 = converted %b_ref.ref, %.loc6_47.19
// CHECK:STDOUT:   assign file.%b.var, %.loc6_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %C.fb5 = splice_block file.%c.var [concrete = file.%c.var] {}
// CHECK:STDOUT:   %F.call: init %C.fb5 to %.loc7 = call %F.ref()
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.323: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.a6a [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x = struct_value (%int_1) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.175 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a08: %I.assoc_type = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = constants.%assoc0.323]
// CHECK:STDOUT:   %Main.import_ref.4ea: %struct_type.x = import_ref Main//symbolic_decl, loc10_37, loaded [concrete = constants.%struct]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.4ea), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a6a: %struct_type.x = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %struct_type.x = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18.1: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.a08 [concrete = constants.%assoc0.323]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %struct_type.x = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc7_18.2: %i32 = struct_access %impl.elem0.loc7_18.1, element0 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %.loc7_18.2, %impl.elem0.loc7_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %.loc7_18.2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_18.2(%.loc7_18.2) [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc7_18.3: init %struct_type.x = struct_init (%Int.as.Copy.impl.Op.call) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc7_24: init %struct_type.x = converted %impl.elem0.loc7_18.1, %.loc7_18.3 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc7_24
// CHECK:STDOUT: }
// CHECK:STDOUT:
