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

class C(S:! {.a: i32, .b: i32}) {}
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
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:1: in import [InImport]
// CHECK:STDERR: implicit.carbon:8:9: note: initializing generic parameter `S` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(S:! {.a: i32, .b: i32}) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C({.a = 1, .b = 2})` to `C({.a = 3, .b = 4})` [ConversionFailure]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C({.a = 1, .b = 2})` does not implement interface `Core.ImplicitAs(C({.a = 3, .b = 4}))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.a = 3, .b = 4}) = F();

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! {.x: i32};
}

class C {}

impl C as I where .val = {.x = 1} {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> {.x: i32} {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f13: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ec = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.5ca: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.b54: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.c27: ref %i32 = struct_access imports.%a_ref.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ea1: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.d6b: %Int.as.Copy.impl.Op.type.ea1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.d0b: <witness> = impl_witness imports.%Copy.impl_witness_table.608, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.edd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.99d: %Int.as.Copy.impl.Op.type.edd = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.574: %Copy.type = facet_value %i32, (%Copy.impl_witness.d0b) [concrete]
// CHECK:STDOUT:   %.0ec: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.574 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.99d, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.dd4} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.9db: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.020: type = pattern_type %struct_type.a.d.9db [concrete]
// CHECK:STDOUT:   %.769: ref %struct_type.b.c = struct_access imports.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %.92c: ref %struct_type.b.c = struct_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %.d8a: ref %i32 = struct_access %.769, element0 [concrete]
// CHECK:STDOUT:   %.9dd: ref %i32 = struct_access %.92c, element0 [concrete]
// CHECK:STDOUT:   %.e5f: ref %tuple.type.dd4 = struct_access %.769, element1 [concrete]
// CHECK:STDOUT:   %.bbf: ref %tuple.type.dd4 = struct_access %.92c, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.aa5: ref %i32 = tuple_access %.e5f, element0 [concrete]
// CHECK:STDOUT:   %.dfe: ref %i32 = struct_access imports.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %.607: ref %i32 = struct_access file.%b.var, element1 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.774: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.ea0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0a0, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b1d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.774 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.9a4) [concrete]
// CHECK:STDOUT:   %.d32: type = fn_type_with_self_type %ImplicitAs.Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a3e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b1d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.b1d, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.609: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.01f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b1d [concrete]
// CHECK:STDOUT:   %bound_method.ece: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.5ca = struct_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct) [concrete]
// CHECK:STDOUT:   %pattern_type.a5a: type = pattern_type %C.a8a [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %struct_type.a = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Implicit.b_ref: ref %struct_type.a.d.9db = import_ref Implicit//default, b_ref, loaded [concrete = %b_ref.var]
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Implicit.import_ref.9fd: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ec) = import_ref Implicit//default, inst118 [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f13)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0a0 = impl_witness_table (%Implicit.import_ref.9fd), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.b54 = binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.b54 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %struct_type.a = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.d8e: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ea1) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.d6b)]
// CHECK:STDOUT:   %Copy.impl_witness_table.608 = impl_witness_table (%Core.import_ref.d8e), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %b_ref.patt: %pattern_type.020 = binding_pattern b_ref [concrete]
// CHECK:STDOUT:   %b_ref.var_patt: %pattern_type.020 = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   %b_ref.var: ref %struct_type.a.d.9db = var %b_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b54 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b54 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.020 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.020 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %struct_type.a.d.9db = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_43: type = splice_block %struct_type.a.d [concrete = constants.%struct_type.a.d.9db] {
// CHECK:STDOUT:     %int_32.loc6_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_32.1: %tuple.type.85c = tuple_literal (%i32.loc6_28)
// CHECK:STDOUT:     %.loc6_32.2: type = converted %.loc6_32.1, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.dd4} [concrete = constants.%struct_type.b.c]
// CHECK:STDOUT:     %int_32.loc6_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.d: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete = constants.%struct_type.a.d.9db]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %struct_type.a.d.9db = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.a5a = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.a5a = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.a8a = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_26.1: type = splice_block %C [concrete = constants.%C.a8a] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc7_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc7_25.1: %.d32 = impl_witness_access constants.%ImplicitAs.impl_witness.9a4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b1d]
// CHECK:STDOUT:     %bound_method.loc7_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_25.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a3e]
// CHECK:STDOUT:     %specific_fn.loc7_25.1: <specific function> = specific_function %impl.elem0.loc7_25.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_25.2: <bound method> = bound_method %int_1, %specific_fn.loc7_25.1 [concrete = constants.%bound_method.609]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.1: init %i32 = call %bound_method.loc7_25.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc7_25.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc7_25.3: %i32 = converted %int_1, %.loc7_25.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %impl.elem0.loc7_25.2: %.d32 = impl_witness_access constants.%ImplicitAs.impl_witness.9a4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b1d]
// CHECK:STDOUT:     %bound_method.loc7_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc7_25.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.01f]
// CHECK:STDOUT:     %specific_fn.loc7_25.2: <specific function> = specific_function %impl.elem0.loc7_25.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_25.4: <bound method> = bound_method %int_2, %specific_fn.loc7_25.2 [concrete = constants.%bound_method.ece]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.2: init %i32 = call %bound_method.loc7_25.4(%int_2) [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc7_25.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_25.2 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc7_25.5: %i32 = converted %int_2, %.loc7_25.4 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %struct: %struct_type.a.b.5ca = struct_value (%.loc7_25.3, %.loc7_25.5) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc7_26.2: %struct_type.a.b.5ca = converted %.loc7_25.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct) [concrete = constants.%C.a8a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.a8a = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %struct_type.a = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %.loc5_20.1: ref %i32 = struct_access %a_ref.ref, element0 [concrete = constants.%.c27]
// CHECK:STDOUT:   %.loc5_20.2: %i32 = bind_value %.loc5_20.1
// CHECK:STDOUT:   %impl.elem0.loc5: %.0ec = impl_witness_access constants.%Copy.impl_witness.d0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.99d]
// CHECK:STDOUT:   %bound_method.loc5_20.1: <bound method> = bound_method %.loc5_20.2, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_20.2: <bound method> = bound_method %.loc5_20.2, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc5: init %i32 = call %bound_method.loc5_20.2(%.loc5_20.2)
// CHECK:STDOUT:   %.loc5_20.3: init %struct_type.a = struct_init (%Int.as.Copy.impl.Op.call.loc5) to file.%a.var
// CHECK:STDOUT:   %.loc5_1: init %struct_type.a = converted %a_ref.ref, %.loc5_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc5_1
// CHECK:STDOUT:   %b_ref.ref: ref %struct_type.a.d.9db = name_ref b_ref, imports.%Implicit.b_ref [concrete = imports.%b_ref.var]
// CHECK:STDOUT:   %.loc6_47.1: ref %struct_type.b.c = struct_access %b_ref.ref, element0 [concrete = constants.%.769]
// CHECK:STDOUT:   %.loc6_47.2: ref %i32 = struct_access %.loc6_47.1, element0 [concrete = constants.%.d8a]
// CHECK:STDOUT:   %.loc6_47.3: %i32 = bind_value %.loc6_47.2
// CHECK:STDOUT:   %impl.elem0.loc6_47.1: %.0ec = impl_witness_access constants.%Copy.impl_witness.d0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.99d]
// CHECK:STDOUT:   %bound_method.loc6_47.1: <bound method> = bound_method %.loc6_47.3, %impl.elem0.loc6_47.1
// CHECK:STDOUT:   %specific_fn.loc6_47.1: <specific function> = specific_function %impl.elem0.loc6_47.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_47.2: <bound method> = bound_method %.loc6_47.3, %specific_fn.loc6_47.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_47.1: init %i32 = call %bound_method.loc6_47.2(%.loc6_47.3)
// CHECK:STDOUT:   %.loc6_47.4: ref %struct_type.b.c = struct_access file.%b.var, element0 [concrete = constants.%.92c]
// CHECK:STDOUT:   %.loc6_47.5: ref %i32 = struct_access %.loc6_47.4, element0 [concrete = constants.%.9dd]
// CHECK:STDOUT:   %.loc6_47.6: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc6_47.1 to %.loc6_47.5
// CHECK:STDOUT:   %.loc6_47.7: ref %tuple.type.dd4 = struct_access %.loc6_47.1, element1 [concrete = constants.%.e5f]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc6_47.7, element0 [concrete = constants.%tuple.elem0.aa5]
// CHECK:STDOUT:   %.loc6_47.8: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc6_47.2: %.0ec = impl_witness_access constants.%Copy.impl_witness.d0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.99d]
// CHECK:STDOUT:   %bound_method.loc6_47.3: <bound method> = bound_method %.loc6_47.8, %impl.elem0.loc6_47.2
// CHECK:STDOUT:   %specific_fn.loc6_47.2: <specific function> = specific_function %impl.elem0.loc6_47.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_47.4: <bound method> = bound_method %.loc6_47.8, %specific_fn.loc6_47.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_47.2: init %i32 = call %bound_method.loc6_47.4(%.loc6_47.8)
// CHECK:STDOUT:   %.loc6_47.9: ref %tuple.type.dd4 = struct_access %.loc6_47.4, element1 [concrete = constants.%.bbf]
// CHECK:STDOUT:   %.loc6_47.10: init %tuple.type.dd4 = tuple_init (%Int.as.Copy.impl.Op.call.loc6_47.2) to %.loc6_47.9
// CHECK:STDOUT:   %.loc6_47.11: init %tuple.type.dd4 = converted %.loc6_47.7, %.loc6_47.10
// CHECK:STDOUT:   %.loc6_47.12: init %tuple.type.dd4 = initialize_from %.loc6_47.11 to %.loc6_47.9
// CHECK:STDOUT:   %.loc6_47.13: init %struct_type.b.c = struct_init (%.loc6_47.6, %.loc6_47.12) to %.loc6_47.4
// CHECK:STDOUT:   %.loc6_47.14: init %struct_type.b.c = converted %.loc6_47.1, %.loc6_47.13
// CHECK:STDOUT:   %.loc6_47.15: ref %i32 = struct_access %b_ref.ref, element1 [concrete = constants.%.dfe]
// CHECK:STDOUT:   %.loc6_47.16: %i32 = bind_value %.loc6_47.15
// CHECK:STDOUT:   %impl.elem0.loc6_47.3: %.0ec = impl_witness_access constants.%Copy.impl_witness.d0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.99d]
// CHECK:STDOUT:   %bound_method.loc6_47.5: <bound method> = bound_method %.loc6_47.16, %impl.elem0.loc6_47.3
// CHECK:STDOUT:   %specific_fn.loc6_47.3: <specific function> = specific_function %impl.elem0.loc6_47.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_47.6: <bound method> = bound_method %.loc6_47.16, %specific_fn.loc6_47.3
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_47.3: init %i32 = call %bound_method.loc6_47.6(%.loc6_47.16)
// CHECK:STDOUT:   %.loc6_47.17: ref %i32 = struct_access file.%b.var, element1 [concrete = constants.%.607]
// CHECK:STDOUT:   %.loc6_47.18: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc6_47.3 to %.loc6_47.17
// CHECK:STDOUT:   %.loc6_47.19: init %struct_type.a.d.9db = struct_init (%.loc6_47.14, %.loc6_47.18) to file.%b.var
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.d.9db = converted %b_ref.ref, %.loc6_47.19
// CHECK:STDOUT:   assign file.%b.var, %.loc6_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %C.a8a = splice_block file.%c.var [concrete = file.%c.var] {}
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc7
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x = struct_value (%int_1) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.862: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.380 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.f59 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.1fc: %I.assoc_type = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = constants.%assoc0.862]
// CHECK:STDOUT:   %Main.import_ref.1ec: %struct_type.x = import_ref Main//symbolic_decl, loc10_35, loaded [concrete = constants.%struct]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.1ec), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.380: %struct_type.x = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %struct_type.x = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.1fc [concrete = constants.%assoc0.862]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18.1: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %struct_type.x = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc7_18.2: %i32 = struct_access %impl.elem0.loc7_18.1, element0 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %.loc7_18.2, %impl.elem0.loc7_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %.loc7_18.2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_18.2(%.loc7_18.2) [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc7_18.3: init %struct_type.x = struct_init (%Int.as.Copy.impl.Op.call) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc7_22: init %struct_type.x = converted %impl.elem0.loc7_18.1, %.loc7_18.3 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc7_22 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
