// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/init_adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/init_adapt.carbon

// --- init_adapt.carbon

library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: i32;
}

class AdaptC {
  adapt C;
}

let a: C = {.a = 1, .b = 2};

let b: AdaptC = a as AdaptC;

let c: C = b as C;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

var d: AdaptC = MakeC() as AdaptC;

var e: C = MakeAdaptC() as C;

// --- fail_not_implicit.carbon

library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: i32;
}

class AdaptC {
  adapt C;
}

let a: C = {.a = 1, .b = 2};

// Cannot implicitly convert between a type and an adapter for the type.

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:17: error: cannot implicitly convert expression of type `C` to `AdaptC` [ConversionFailure]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR:                 ^
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:17: note: type `C` does not implement interface `Core.ImplicitAs(AdaptC)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
let b: AdaptC = a;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:12: error: cannot implicitly convert expression of type `AdaptC` to `C` [ConversionFailure]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:12: note: type `AdaptC` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR:            ^
// CHECK:STDERR:
let c: C = b;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C` to `AdaptC` [ConversionFailure]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `Core.ImplicitAs(AdaptC)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
var d: AdaptC = MakeC();

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `AdaptC` to `C` [ConversionFailure]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `AdaptC` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
var e: C = MakeAdaptC();

// CHECK:STDOUT: --- init_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b4c: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.7b9: <witness> = complete_type_witness %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt.c32: %pattern_type.98b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %.c76: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %pattern_type.720: type = pattern_type %AdaptC [concrete]
// CHECK:STDOUT:   %b.patt.658: %pattern_type.720 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %.624: Core.Form = init_form %AdaptC [concrete]
// CHECK:STDOUT:   %return.param_patt.788: %pattern_type.720 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.794: %pattern_type.720 = return_slot_pattern %return.param_patt.788, %AdaptC [concrete]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [concrete]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.720 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt: %pattern_type.720 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.98b = ref_binding_pattern e [concrete]
// CHECK:STDOUT:   %e.var_patt: %pattern_type.98b = var_pattern %e.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AdaptC = %AdaptC.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .MakeAdaptC = %MakeAdaptC.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [concrete = constants.%AdaptC] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.98b = value_binding_pattern a [concrete = constants.%a.patt.c32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc13_27.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_27.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc13_27.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc13_27.1: <specific function> = specific_function %impl.elem0.loc13_27.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc13_27.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1: init %i32 = call %bound_method.loc13_27.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_27.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_27.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref %i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init %i32 to %.loc13_27.3 = in_place_init %.loc13_27.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc13_27.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_27.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc13_27.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc13_27.2: <specific function> = specific_function %impl.elem0.loc13_27.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc13_27.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2: init %i32 = call %bound_method.loc13_27.4(@__global_init.%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_27.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_27.6: ref %i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.7: init %i32 to %.loc13_27.6 = in_place_init %.loc13_27.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_27.8: init %C to %.loc13_27.2 = class_init (%.loc13_27.4, %.loc13_27.7) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_27.9: init %C = converted @__global_init.%.loc13, %.loc13_27.8 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_27.10: ref %C = temporary %.loc13_27.2, %.loc13_27.9 [concrete = constants.%.c76]
// CHECK:STDOUT:   %.loc13_27.11: %C = acquire_value %.loc13_27.10 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: %C = wrapper_binding a, %.loc13_27.11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.720 = value_binding_pattern b [concrete = constants.%b.patt.658]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc15: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %b: %AdaptC = wrapper_binding b, @__global_init.%.loc15_19.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, @__global_init.%.loc17_14.2
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc19: Core.Form = init_form %C.ref [concrete = constants.%.a44]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [concrete = constants.%MakeAdaptC] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.720 = out_param_pattern [concrete = constants.%return.param_patt.788]
// CHECK:STDOUT:     %return.patt: %pattern_type.720 = return_slot_pattern %return.param_patt, %AdaptC.ref [concrete = constants.%return.patt.794]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:     %.loc21: Core.Form = init_form %AdaptC.ref [concrete = constants.%.624]
// CHECK:STDOUT:     %return.param: ref %AdaptC = out_param call_param0
// CHECK:STDOUT:     %return: ref %AdaptC = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.720 = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.720 = var_pattern %d.patt [concrete = constants.%d.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptC = var %d.var_patt [concrete]
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %d: ref %AdaptC = wrapper_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.98b = ref_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.98b = var_pattern %e.patt [concrete = constants.%e.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %C = var %e.var_patt [concrete]
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e: ref %C = wrapper_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.loc6: %C.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.b4c [concrete = constants.%complete_type.7b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   adapt_decl %C.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.b4c [concrete = constants.%complete_type.7b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> out %return.param: %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> out %return.param: %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, file.%a
// CHECK:STDOUT:   %AdaptC.ref.loc15: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %.loc15_19.1: %AdaptC = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_19.2: %AdaptC = converted %a.ref, %.loc15_19.1
// CHECK:STDOUT:   %b.ref: %AdaptC = name_ref b, file.%b
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_14.1: %C = as_compatible %b.ref
// CHECK:STDOUT:   %.loc17_14.2: %C = converted %b.ref, %.loc17_14.1
// CHECK:STDOUT:   %MakeC.ref: %MakeC.type = name_ref MakeC, file.%MakeC.decl [concrete = constants.%MakeC]
// CHECK:STDOUT:   %.loc23_1: ref %AdaptC = splice_block file.%d.var [concrete = file.%d.var] {}
// CHECK:STDOUT:   %MakeC.call: init %C to %.loc23_1 = call %MakeC.ref()
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %.loc23_25.1: init %AdaptC = as_compatible %MakeC.call
// CHECK:STDOUT:   %.loc23_25.2: init %AdaptC = converted %MakeC.call, %.loc23_25.1
// CHECK:STDOUT:   assign file.%d.var, %.loc23_25.2
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [concrete = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc25_1: ref %C = splice_block file.%e.var [concrete = file.%e.var] {}
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC to %.loc25_1 = call %MakeAdaptC.ref()
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc25_25.1: init %C = as_compatible %MakeAdaptC.call
// CHECK:STDOUT:   %.loc25_25.2: init %C = converted %MakeAdaptC.call, %.loc25_25.1
// CHECK:STDOUT:   assign file.%e.var, %.loc25_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b4c: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.7b9: <witness> = complete_type_witness %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt.c32: %pattern_type.98b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %.c76: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %pattern_type.720: type = pattern_type %AdaptC [concrete]
// CHECK:STDOUT:   %b.patt.658: %pattern_type.720 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %.624: Core.Form = init_form %AdaptC [concrete]
// CHECK:STDOUT:   %return.param_patt.788: %pattern_type.720 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.794: %pattern_type.720 = return_slot_pattern %return.param_patt.788, %AdaptC [concrete]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [concrete]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.720 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt: %pattern_type.720 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.98b = ref_binding_pattern e [concrete]
// CHECK:STDOUT:   %e.var_patt: %pattern_type.98b = var_pattern %e.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AdaptC = %AdaptC.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .MakeAdaptC = %MakeAdaptC.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [concrete = constants.%AdaptC] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.98b = value_binding_pattern a [concrete = constants.%a.patt.c32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc13_27.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_27.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc13_27.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc13_27.1: <specific function> = specific_function %impl.elem0.loc13_27.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc13_27.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1: init %i32 = call %bound_method.loc13_27.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_27.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_27.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref %i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init %i32 to %.loc13_27.3 = in_place_init %.loc13_27.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc13_27.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_27.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc13_27.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc13_27.2: <specific function> = specific_function %impl.elem0.loc13_27.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc13_27.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2: init %i32 = call %bound_method.loc13_27.4(@__global_init.%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_27.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_27.6: ref %i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.7: init %i32 to %.loc13_27.6 = in_place_init %.loc13_27.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_27.8: init %C to %.loc13_27.2 = class_init (%.loc13_27.4, %.loc13_27.7) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_27.9: init %C = converted @__global_init.%.loc13, %.loc13_27.8 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_27.10: ref %C = temporary %.loc13_27.2, %.loc13_27.9 [concrete = constants.%.c76]
// CHECK:STDOUT:   %.loc13_27.11: %C = acquire_value %.loc13_27.10 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: %C = wrapper_binding a, %.loc13_27.11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.720 = value_binding_pattern b [concrete = constants.%b.patt.658]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24: %AdaptC = converted @__global_init.%a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %AdaptC.ref.loc24: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %b: %AdaptC = wrapper_binding b, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33: %C = converted @__global_init.%b.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %C.ref.loc33: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, <error> [concrete = <error>]
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc35: Core.Form = init_form %C.ref [concrete = constants.%.a44]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [concrete = constants.%MakeAdaptC] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.720 = out_param_pattern [concrete = constants.%return.param_patt.788]
// CHECK:STDOUT:     %return.patt: %pattern_type.720 = return_slot_pattern %return.param_patt, %AdaptC.ref [concrete = constants.%return.patt.794]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:     %.loc37: Core.Form = init_form %AdaptC.ref [concrete = constants.%.624]
// CHECK:STDOUT:     %return.param: ref %AdaptC = out_param call_param0
// CHECK:STDOUT:     %return: ref %AdaptC = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.720 = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.720 = var_pattern %d.patt [concrete = constants.%d.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptC = var %d.var_patt [concrete]
// CHECK:STDOUT:   %AdaptC.ref.loc46: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %d: ref %AdaptC = wrapper_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.98b = ref_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.98b = var_pattern %e.patt [concrete = constants.%e.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %C = var %e.var_patt [concrete]
// CHECK:STDOUT:   %C.ref.loc55: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e: ref %C = wrapper_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.loc6: %C.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.b4c [concrete = constants.%complete_type.7b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   adapt_decl %C.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.b4c [concrete = constants.%complete_type.7b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> out %return.param: %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> out %return.param: %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: %AdaptC = name_ref b, file.%b [concrete = <error>]
// CHECK:STDOUT:   %MakeC.ref: %MakeC.type = name_ref MakeC, file.%MakeC.decl [concrete = constants.%MakeC]
// CHECK:STDOUT:   %.loc46_23: ref %C = temporary_storage
// CHECK:STDOUT:   %MakeC.call: init %C to %.loc46_23 = call %MakeC.ref()
// CHECK:STDOUT:   %.loc46_1: %AdaptC = converted %MakeC.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [concrete = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc55_23: ref %AdaptC = temporary_storage
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC to %.loc55_23 = call %MakeAdaptC.ref()
// CHECK:STDOUT:   %.loc55_1: %C = converted %MakeAdaptC.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%e.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
