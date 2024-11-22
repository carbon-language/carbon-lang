// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_adapt.carbon

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

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `C` to `AdaptC` [ImplicitAsConversionFailure]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `ImplicitAs(AdaptC)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: AdaptC = a;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `AdaptC` to `C` [ImplicitAsConversionFailure]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `AdaptC` does not implement interface `ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
let c: C = b;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `C` to `AdaptC` [ImplicitAsConversionFailure]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `ImplicitAs(AdaptC)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var d: AdaptC = MakeC();

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `AdaptC` to `C` [ImplicitAsConversionFailure]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+3]]:1: note: type `AdaptC` does not implement interface `ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
var e: C = MakeAdaptC();

// CHECK:STDOUT: --- init_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %C [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.36, %.39) [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [template]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AdaptC = %AdaptC.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .MakeAdaptC = %MakeAdaptC.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [template = constants.%AdaptC] {} {}
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %AdaptC.ref.loc15: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [template = constants.%MakeC] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [template = constants.%MakeAdaptC] {
// CHECK:STDOUT:     %return.patt: %AdaptC = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptC = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:     %return.param: ref %AdaptC = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %AdaptC = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %d.var: ref %AdaptC = var d
// CHECK:STDOUT:   %d: ref %AdaptC = bind_name d, %d.var
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %e.var: ref %C = var e
// CHECK:STDOUT:   %e: ref %C = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.3: type = converted %int.make_type_signed.loc5, %.loc5_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.3: type = converted %int.make_type_signed.loc6, %.loc6_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl %C
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %C [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: Core.IntLiteral = int_value 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_26: Core.IntLiteral = int_value 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_27.1: %.9 = struct_literal (%.loc13_18, %.loc13_26)
// CHECK:STDOUT:   %.loc13_27.2: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_27.3: <bound method> = bound_method %.loc13_18, %.loc13_27.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_27.4: <specific function> = specific_function %.loc13_27.3, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc13_27.1: init %i32 = call %.loc13_27.4(%.loc13_18) [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_27.5: init %i32 = converted %.loc13_18, %int.convert_checked.loc13_27.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_27.6: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.7: ref %i32 = class_element_access %.loc13_27.6, element0
// CHECK:STDOUT:   %.loc13_27.8: init %i32 = initialize_from %.loc13_27.5 to %.loc13_27.7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_27.9: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_27.10: <bound method> = bound_method %.loc13_26, %.loc13_27.9 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_27.11: <specific function> = specific_function %.loc13_27.10, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc13_27.2: init %i32 = call %.loc13_27.11(%.loc13_26) [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_27.12: init %i32 = converted %.loc13_26, %int.convert_checked.loc13_27.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_27.13: ref %i32 = class_element_access %.loc13_27.6, element1
// CHECK:STDOUT:   %.loc13_27.14: init %i32 = initialize_from %.loc13_27.12 to %.loc13_27.13 [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_27.15: init %C = class_init (%.loc13_27.8, %.loc13_27.14), %.loc13_27.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_27.16: ref %C = temporary %.loc13_27.6, %.loc13_27.15
// CHECK:STDOUT:   %.loc13_28.1: ref %C = converted %.loc13_27.1, %.loc13_27.16
// CHECK:STDOUT:   %.loc13_28.2: %C = bind_value %.loc13_28.1
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc13_28.2
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %AdaptC.ref.loc15: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %.loc15_19.1: %AdaptC = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_19.2: %AdaptC = converted %a.ref, %.loc15_19.1
// CHECK:STDOUT:   %b: %AdaptC = bind_name b, %.loc15_19.2
// CHECK:STDOUT:   %b.ref: %AdaptC = name_ref b, %b
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc17_14.1: %C = as_compatible %b.ref
// CHECK:STDOUT:   %.loc17_14.2: %C = converted %b.ref, %.loc17_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc17_14.2
// CHECK:STDOUT:   %MakeC.ref: %MakeC.type = name_ref MakeC, file.%MakeC.decl [template = constants.%MakeC]
// CHECK:STDOUT:   %.loc23_5: ref %AdaptC = splice_block file.%d.var {}
// CHECK:STDOUT:   %MakeC.call: init %C = call %MakeC.ref() to %.loc23_5
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %.loc23_25.1: init %AdaptC = as_compatible %MakeC.call
// CHECK:STDOUT:   %.loc23_25.2: init %AdaptC = converted %MakeC.call, %.loc23_25.1
// CHECK:STDOUT:   assign file.%d.var, %.loc23_25.2
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [template = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc25_5: ref %C = splice_block file.%e.var {}
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC = call %MakeAdaptC.ref() to %.loc25_5
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc25_25.1: init %C = as_compatible %MakeAdaptC.call
// CHECK:STDOUT:   %.loc25_25.2: init %C = converted %MakeAdaptC.call, %.loc25_25.1
// CHECK:STDOUT:   assign file.%e.var, %.loc25_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %C [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.36, %.39) [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [template]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AdaptC = %AdaptC.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .MakeAdaptC = %MakeAdaptC.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [template = constants.%AdaptC] {} {}
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %AdaptC.ref.loc24: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %C.ref.loc33: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [template = constants.%MakeC] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [template = constants.%MakeAdaptC] {
// CHECK:STDOUT:     %return.patt: %AdaptC = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptC = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:     %return.param: ref %AdaptC = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %AdaptC = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc46: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %d.var: ref %AdaptC = var d
// CHECK:STDOUT:   %d: ref %AdaptC = bind_name d, %d.var
// CHECK:STDOUT:   %C.ref.loc54: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %e.var: ref %C = var e
// CHECK:STDOUT:   %e: ref %C = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.3: type = converted %int.make_type_signed.loc5, %.loc5_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.3: type = converted %int.make_type_signed.loc6, %.loc6_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl %C
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %C [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: Core.IntLiteral = int_value 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_26: Core.IntLiteral = int_value 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_27.1: %.9 = struct_literal (%.loc13_18, %.loc13_26)
// CHECK:STDOUT:   %.loc13_27.2: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_27.3: <bound method> = bound_method %.loc13_18, %.loc13_27.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_27.4: <specific function> = specific_function %.loc13_27.3, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc13_27.1: init %i32 = call %.loc13_27.4(%.loc13_18) [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_27.5: init %i32 = converted %.loc13_18, %int.convert_checked.loc13_27.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_27.6: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.7: ref %i32 = class_element_access %.loc13_27.6, element0
// CHECK:STDOUT:   %.loc13_27.8: init %i32 = initialize_from %.loc13_27.5 to %.loc13_27.7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_27.9: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_27.10: <bound method> = bound_method %.loc13_26, %.loc13_27.9 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_27.11: <specific function> = specific_function %.loc13_27.10, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc13_27.2: init %i32 = call %.loc13_27.11(%.loc13_26) [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_27.12: init %i32 = converted %.loc13_26, %int.convert_checked.loc13_27.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_27.13: ref %i32 = class_element_access %.loc13_27.6, element1
// CHECK:STDOUT:   %.loc13_27.14: init %i32 = initialize_from %.loc13_27.12 to %.loc13_27.13 [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_27.15: init %C = class_init (%.loc13_27.8, %.loc13_27.14), %.loc13_27.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_27.16: ref %C = temporary %.loc13_27.6, %.loc13_27.15
// CHECK:STDOUT:   %.loc13_28.1: ref %C = converted %.loc13_27.1, %.loc13_27.16
// CHECK:STDOUT:   %.loc13_28.2: %C = bind_value %.loc13_28.1
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc13_28.2
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %.loc24: %AdaptC = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %b: %AdaptC = bind_name b, <error>
// CHECK:STDOUT:   %b.ref: %AdaptC = name_ref b, %b
// CHECK:STDOUT:   %.loc33: %C = converted %b.ref, <error> [template = <error>]
// CHECK:STDOUT:   %c: %C = bind_name c, <error>
// CHECK:STDOUT:   %MakeC.ref: %MakeC.type = name_ref MakeC, file.%MakeC.decl [template = constants.%MakeC]
// CHECK:STDOUT:   %.loc46_22: ref %C = temporary_storage
// CHECK:STDOUT:   %MakeC.call: init %C = call %MakeC.ref() to %.loc46_22
// CHECK:STDOUT:   %.loc46_24: %AdaptC = converted %MakeC.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [template = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc54_22: ref %AdaptC = temporary_storage
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC = call %MakeAdaptC.ref() to %.loc54_22
// CHECK:STDOUT:   %.loc54_24: %C = converted %MakeAdaptC.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%e.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
