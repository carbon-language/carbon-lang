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
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:17: note: type `C` does not implement interface `Core.ImplicitAs(AdaptC)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
let b: AdaptC = a;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:12: error: cannot implicitly convert expression of type `AdaptC` to `C` [ConversionFailure]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:12: note: type `AdaptC` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR:            ^
// CHECK:STDERR:
let c: C = b;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C` to `AdaptC` [ConversionFailure]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `Core.ImplicitAs(AdaptC)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
var d: AdaptC = MakeC();

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `AdaptC` to `C` [ConversionFailure]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `AdaptC` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d92: <witness> = impl_witness @AdaptC.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.b20: type = ptr_type %AdaptC [concrete]
// CHECK:STDOUT:   %pattern_type.d7e: type = pattern_type %ptr.b20 [concrete]
// CHECK:STDOUT:   %AdaptC.as.Destroy.impl.Op.type: type = fn_type @AdaptC.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptC.as.Destroy.impl.Op: %AdaptC.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %pattern_type.a1a: type = pattern_type %AdaptC [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [concrete]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc13_27.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_27.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc13_27.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc13_27.1: <specific function> = specific_function %impl.elem0.loc13_27.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc13_27.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1: init %i32 = call %bound_method.loc13_27.2(@__global_init.%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_27.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_27.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref %i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init %i32 = initialize_from %.loc13_27.1 to %.loc13_27.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc13_27.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_27.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc13_27.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc13_27.2: <specific function> = specific_function %impl.elem0.loc13_27.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc13_27.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2: init %i32 = call %bound_method.loc13_27.4(@__global_init.%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_27.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_27.6: ref %i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.7: init %i32 = initialize_from %.loc13_27.5 to %.loc13_27.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_27.8: init %C = class_init (%.loc13_27.4, %.loc13_27.7), %.loc13_27.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_27.9: ref %C = temporary %.loc13_27.2, %.loc13_27.8
// CHECK:STDOUT:   %.loc13_27.10: ref %C = converted @__global_init.%.loc13, %.loc13_27.9
// CHECK:STDOUT:   %.loc13_27.11: %C = bind_value %.loc13_27.10
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc13_27.11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a1a = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc15: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %b: %AdaptC = bind_name b, @__global_init.%.loc15_19.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = bind_name c, @__global_init.%.loc17_14.2
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [concrete = constants.%MakeAdaptC] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a1a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a1a = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:     %return.param: ref %AdaptC = out_param call_param0
// CHECK:STDOUT:     %return: ref %AdaptC = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.a1a = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.a1a = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptC = var %d.var_patt [concrete]
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %d: ref %AdaptC = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.c48 = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.c48 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %C = var %e.var_patt [concrete]
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e: ref %C = bind_name e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AdaptC.as.Destroy.impl: @AdaptC.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptC.as.Destroy.impl.Op.decl: %AdaptC.as.Destroy.impl.Op.type = fn_decl @AdaptC.as.Destroy.impl.Op [concrete = constants.%AdaptC.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.d7e = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d7e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.b20 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptC [concrete = constants.%AdaptC]
// CHECK:STDOUT:     %self: %ptr.b20 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptC.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptC.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %C.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.501 [concrete = constants.%complete_type.705]
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptC [concrete = constants.%AdaptC]
// CHECK:STDOUT:   impl_decl @AdaptC.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptC.as.Destroy.impl.%AdaptC.as.Destroy.impl.Op.decl), @AdaptC.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.d92]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.501 [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptC.as.Destroy.impl.Op(%self.param: %ptr.b20) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %return.param: %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> %return.param: %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
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
// CHECK:STDOUT:   %MakeC.call: init %C = call %MakeC.ref() to %.loc23_1
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %.loc23_25.1: init %AdaptC = as_compatible %MakeC.call
// CHECK:STDOUT:   %.loc23_25.2: init %AdaptC = converted %MakeC.call, %.loc23_25.1
// CHECK:STDOUT:   assign file.%d.var, %.loc23_25.2
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [concrete = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc25_1: ref %C = splice_block file.%e.var [concrete = file.%e.var] {}
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC = call %MakeAdaptC.ref() to %.loc25_1
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d92: <witness> = impl_witness @AdaptC.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.b20: type = ptr_type %AdaptC [concrete]
// CHECK:STDOUT:   %pattern_type.d7e: type = pattern_type %ptr.b20 [concrete]
// CHECK:STDOUT:   %AdaptC.as.Destroy.impl.Op.type: type = fn_type @AdaptC.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptC.as.Destroy.impl.Op: %AdaptC.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %pattern_type.a1a: type = pattern_type %AdaptC [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [concrete]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc13_27.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_27.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc13_27.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc13_27.1: <specific function> = specific_function %impl.elem0.loc13_27.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc13_27.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1: init %i32 = call %bound_method.loc13_27.2(@__global_init.%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_27.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_27.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref %i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init %i32 = initialize_from %.loc13_27.1 to %.loc13_27.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc13_27.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_27.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc13_27.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc13_27.2: <specific function> = specific_function %impl.elem0.loc13_27.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_27.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc13_27.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2: init %i32 = call %bound_method.loc13_27.4(@__global_init.%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_27.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_27.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_27.6: ref %i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.7: init %i32 = initialize_from %.loc13_27.5 to %.loc13_27.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_27.8: init %C = class_init (%.loc13_27.4, %.loc13_27.7), %.loc13_27.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_27.9: ref %C = temporary %.loc13_27.2, %.loc13_27.8
// CHECK:STDOUT:   %.loc13_27.10: ref %C = converted @__global_init.%.loc13, %.loc13_27.9
// CHECK:STDOUT:   %.loc13_27.11: %C = bind_value %.loc13_27.10
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc13_27.11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a1a = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc24: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %.loc24: %AdaptC = converted @__global_init.%a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %b: %AdaptC = bind_name b, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc33: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc33: %C = converted @__global_init.%b.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %c: %C = bind_name c, <error> [concrete = <error>]
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [concrete = constants.%MakeAdaptC] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a1a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a1a = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:     %return.param: ref %AdaptC = out_param call_param0
// CHECK:STDOUT:     %return: ref %AdaptC = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.a1a = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.a1a = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptC = var %d.var_patt [concrete]
// CHECK:STDOUT:   %AdaptC.ref.loc46: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %d: ref %AdaptC = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.c48 = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.c48 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %C = var %e.var_patt [concrete]
// CHECK:STDOUT:   %C.ref.loc55: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e: ref %C = bind_name e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AdaptC.as.Destroy.impl: @AdaptC.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptC.as.Destroy.impl.Op.decl: %AdaptC.as.Destroy.impl.Op.type = fn_decl @AdaptC.as.Destroy.impl.Op [concrete = constants.%AdaptC.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.d7e = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d7e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.b20 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptC [concrete = constants.%AdaptC]
// CHECK:STDOUT:     %self: %ptr.b20 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptC.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptC.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %C.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.501 [concrete = constants.%complete_type.705]
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptC [concrete = constants.%AdaptC]
// CHECK:STDOUT:   impl_decl @AdaptC.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptC.as.Destroy.impl.%AdaptC.as.Destroy.impl.Op.decl), @AdaptC.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.d92]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.501 [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptC.as.Destroy.impl.Op(%self.param: %ptr.b20) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %return.param: %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> %return.param: %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %a.ref: %C = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: %AdaptC = name_ref b, file.%b [concrete = <error>]
// CHECK:STDOUT:   %MakeC.ref: %MakeC.type = name_ref MakeC, file.%MakeC.decl [concrete = constants.%MakeC]
// CHECK:STDOUT:   %.loc46_23: ref %C = temporary_storage
// CHECK:STDOUT:   %MakeC.call: init %C = call %MakeC.ref() to %.loc46_23
// CHECK:STDOUT:   %.loc46_1: %AdaptC = converted %MakeC.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [concrete = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc55_23: ref %AdaptC = temporary_storage
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC = call %MakeAdaptC.ref() to %.loc55_23
// CHECK:STDOUT:   %.loc55_1: %C = converted %MakeAdaptC.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%e.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
