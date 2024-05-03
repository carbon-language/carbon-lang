// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- init_adapt.carbon

library "init_adapt" api;

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

library "fail_not_implicit" api;

class C {
  var a: i32;
  var b: i32;
}

class AdaptC {
  adapt C;
}

let a: C = {.a = 1, .b = 2};

// Cannot implicitly convert between a type and an adapter for the type.

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: ERROR: Cannot implicitly convert from `C` to `AdaptC`.
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: AdaptC = a;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: ERROR: Cannot implicitly convert from `AdaptC` to `C`.
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
let c: C = b;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: ERROR: Cannot implicitly convert from `C` to `AdaptC`.
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var d: AdaptC = MakeC();

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `AdaptC` to `C`.
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
var e: C = MakeAdaptC();

// CHECK:STDOUT: --- init_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.1: C = struct_value (%.4, %.5) [template]
// CHECK:STDOUT:   %MakeC: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.2: MakeC = struct_value () [template]
// CHECK:STDOUT:   %MakeAdaptC: type = fn_type @MakeAdaptC [template]
// CHECK:STDOUT:   %struct.3: MakeAdaptC = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AdaptC = %AdaptC.decl
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .MakeAdaptC = %MakeAdaptC.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [template = constants.%AdaptC] {}
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_18: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_26: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_27.1: {.a: i32, .b: i32} = struct_literal (%.loc13_18, %.loc13_26)
// CHECK:STDOUT:   %.loc13_27.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init i32 = initialize_from %.loc13_18 to %.loc13_27.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_27.5: ref i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.6: init i32 = initialize_from %.loc13_26 to %.loc13_27.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_27.7: init C = class_init (%.loc13_27.4, %.loc13_27.6), %.loc13_27.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_27.8: ref C = temporary %.loc13_27.2, %.loc13_27.7
// CHECK:STDOUT:   %.loc13_28.1: ref C = converted %.loc13_27.1, %.loc13_27.8
// CHECK:STDOUT:   %.loc13_28.2: C = bind_value %.loc13_28.1
// CHECK:STDOUT:   %a: C = bind_name a, %.loc13_28.2
// CHECK:STDOUT:   %AdaptC.ref.loc15_8: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %AdaptC.ref.loc15_22: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %.loc15_19.1: AdaptC = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_19.2: AdaptC = converted %a.ref, %.loc15_19.1
// CHECK:STDOUT:   %b: AdaptC = bind_name b, %.loc15_19.2
// CHECK:STDOUT:   %C.ref.loc17_8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %b.ref: AdaptC = name_ref b, %b
// CHECK:STDOUT:   %C.ref.loc17_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc17_14.1: C = as_compatible %b.ref
// CHECK:STDOUT:   %.loc17_14.2: C = converted %b.ref, %.loc17_14.1
// CHECK:STDOUT:   %c: C = bind_name c, %.loc17_14.2
// CHECK:STDOUT:   %MakeC.decl: MakeC = fn_decl @MakeC [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @MakeC.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: MakeAdaptC = fn_decl @MakeAdaptC [template = constants.%struct.3] {
// CHECK:STDOUT:     %AdaptC.ref.loc21: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:     @MakeAdaptC.%return: ref AdaptC = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc23: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %d.var: ref AdaptC = var d
// CHECK:STDOUT:   %d: ref AdaptC = bind_name d, %d.var
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %e.var: ref C = var e
// CHECK:STDOUT:   %e: ref C = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: <unbound element of class C> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6: <unbound element of class C> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl C
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeC.ref: MakeC = name_ref MakeC, file.%MakeC.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc23_5: ref AdaptC = splice_block file.%d.var {}
// CHECK:STDOUT:   %MakeC.call: init C = call %MakeC.ref() to %.loc23_5
// CHECK:STDOUT:   %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %.loc23_25.1: init AdaptC = as_compatible %MakeC.call
// CHECK:STDOUT:   %.loc23_25.2: init AdaptC = converted %MakeC.call, %.loc23_25.1
// CHECK:STDOUT:   assign file.%d.var, %.loc23_25.2
// CHECK:STDOUT:   %MakeAdaptC.ref: MakeAdaptC = name_ref MakeAdaptC, file.%MakeAdaptC.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc25_5: ref C = splice_block file.%e.var {}
// CHECK:STDOUT:   %MakeAdaptC.call: init AdaptC = call %MakeAdaptC.ref() to %.loc25_5
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc25_25.1: init C = as_compatible %MakeAdaptC.call
// CHECK:STDOUT:   %.loc25_25.2: init C = converted %MakeAdaptC.call, %.loc25_25.1
// CHECK:STDOUT:   assign file.%e.var, %.loc25_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.1: C = struct_value (%.4, %.5) [template]
// CHECK:STDOUT:   %MakeC: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.2: MakeC = struct_value () [template]
// CHECK:STDOUT:   %MakeAdaptC: type = fn_type @MakeAdaptC [template]
// CHECK:STDOUT:   %struct.3: MakeAdaptC = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AdaptC = %AdaptC.decl
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .MakeAdaptC = %MakeAdaptC.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [template = constants.%AdaptC] {}
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_18: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_26: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_27.1: {.a: i32, .b: i32} = struct_literal (%.loc13_18, %.loc13_26)
// CHECK:STDOUT:   %.loc13_27.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init i32 = initialize_from %.loc13_18 to %.loc13_27.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_27.5: ref i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.6: init i32 = initialize_from %.loc13_26 to %.loc13_27.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_27.7: init C = class_init (%.loc13_27.4, %.loc13_27.6), %.loc13_27.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_27.8: ref C = temporary %.loc13_27.2, %.loc13_27.7
// CHECK:STDOUT:   %.loc13_28.1: ref C = converted %.loc13_27.1, %.loc13_27.8
// CHECK:STDOUT:   %.loc13_28.2: C = bind_value %.loc13_28.1
// CHECK:STDOUT:   %a: C = bind_name a, %.loc13_28.2
// CHECK:STDOUT:   %AdaptC.ref.loc21: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b: AdaptC = bind_name b, <error>
// CHECK:STDOUT:   %C.ref.loc27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %b.ref: AdaptC = name_ref b, %b
// CHECK:STDOUT:   %c: C = bind_name c, <error>
// CHECK:STDOUT:   %MakeC.decl: MakeC = fn_decl @MakeC [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc29: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @MakeC.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: MakeAdaptC = fn_decl @MakeAdaptC [template = constants.%struct.3] {
// CHECK:STDOUT:     %AdaptC.ref.loc31: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:     @MakeAdaptC.%return: ref AdaptC = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc37: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %d.var: ref AdaptC = var d
// CHECK:STDOUT:   %d: ref AdaptC = bind_name d, %d.var
// CHECK:STDOUT:   %C.ref.loc42: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %e.var: ref C = var e
// CHECK:STDOUT:   %e: ref C = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: <unbound element of class C> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6: <unbound element of class C> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl C
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeC.ref: MakeC = name_ref MakeC, file.%MakeC.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc37: ref C = temporary_storage
// CHECK:STDOUT:   %MakeC.call: init C = call %MakeC.ref() to %.loc37
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   %MakeAdaptC.ref: MakeAdaptC = name_ref MakeAdaptC, file.%MakeAdaptC.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc42: ref AdaptC = temporary_storage
// CHECK:STDOUT:   %MakeAdaptC.call: init AdaptC = call %MakeAdaptC.ref() to %.loc42
// CHECK:STDOUT:   assign file.%e.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
