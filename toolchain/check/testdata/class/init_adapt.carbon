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

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `C` to `AdaptC`
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `ImplicitAs`
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: AdaptC = a;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `AdaptC` to `C`
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `<error>` does not implement interface `ImplicitAs`
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
let c: C = b;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `C` to `AdaptC`
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `ImplicitAs`
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var d: AdaptC = MakeC();

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `AdaptC` to `C`
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+3]]:1: note: type `AdaptC` does not implement interface `ImplicitAs`
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
var e: C = MakeAdaptC();

// CHECK:STDOUT: --- init_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %C [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.7, %.8) [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [template]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [template = constants.%MakeC] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [template = constants.%MakeAdaptC] {} {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:     %return: ref %AdaptC = var <return slot>
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
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32.loc5, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_32.loc6, %.loc6_10.1 [template = i32]
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_26: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_27.1: %.3 = struct_literal (%.loc13_18, %.loc13_26)
// CHECK:STDOUT:   %.loc13_27.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init i32 = initialize_from %.loc13_18 to %.loc13_27.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_27.5: ref i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.6: init i32 = initialize_from %.loc13_26 to %.loc13_27.5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_27.7: init %C = class_init (%.loc13_27.4, %.loc13_27.6), %.loc13_27.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_27.8: ref %C = temporary %.loc13_27.2, %.loc13_27.7
// CHECK:STDOUT:   %.loc13_28.1: ref %C = converted %.loc13_27.1, %.loc13_27.8
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %C [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.7, %.8) [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%AdaptC) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%AdaptC) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.13: %.9 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = interface_type @ImplicitAs, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.3 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %MakeAdaptC.type: type = fn_type @MakeAdaptC [template]
// CHECK:STDOUT:   %MakeAdaptC: %MakeAdaptC.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.9) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+56, unloaded
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
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [template = constants.%MakeC] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeAdaptC.decl: %MakeAdaptC.type = fn_decl @MakeAdaptC [template = constants.%MakeAdaptC] {} {
// CHECK:STDOUT:     %AdaptC.ref: type = name_ref AdaptC, file.%AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:     %return: ref %AdaptC = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc46: type = name_ref AdaptC, %AdaptC.decl [template = constants.%AdaptC]
// CHECK:STDOUT:   %d.var: ref %AdaptC = var d
// CHECK:STDOUT:   %d: ref %AdaptC = bind_name d, %d.var
// CHECK:STDOUT:   %C.ref.loc54: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %e.var: ref %C = var e
// CHECK:STDOUT:   %e: ref %C = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.9) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32.loc5, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_32.loc6, %.loc6_10.1 [template = i32]
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeAdaptC() -> %AdaptC;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_26: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_27.1: %.3 = struct_literal (%.loc13_18, %.loc13_26)
// CHECK:STDOUT:   %.loc13_27.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_27.3: ref i32 = class_element_access %.loc13_27.2, element0
// CHECK:STDOUT:   %.loc13_27.4: init i32 = initialize_from %.loc13_18 to %.loc13_27.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_27.5: ref i32 = class_element_access %.loc13_27.2, element1
// CHECK:STDOUT:   %.loc13_27.6: init i32 = initialize_from %.loc13_26 to %.loc13_27.5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_27.7: init %C = class_init (%.loc13_27.4, %.loc13_27.6), %.loc13_27.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_27.8: ref %C = temporary %.loc13_27.2, %.loc13_27.7
// CHECK:STDOUT:   %.loc13_28.1: ref %C = converted %.loc13_27.1, %.loc13_27.8
// CHECK:STDOUT:   %.loc13_28.2: %C = bind_value %.loc13_28.1
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc13_28.2
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %ImplicitAs.type.loc24: type = interface_type @ImplicitAs, @ImplicitAs(constants.%AdaptC) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc24_18.1: %.11 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%AdaptC) [template = constants.%.12]
// CHECK:STDOUT:   %Convert.ref.loc24: %.11 = name_ref Convert, %.loc24_18.1 [template = constants.%.12]
// CHECK:STDOUT:   %.loc24_18.2: %AdaptC = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %b: %AdaptC = bind_name b, <error>
// CHECK:STDOUT:   %b.ref: %AdaptC = name_ref b, %b
// CHECK:STDOUT:   %ImplicitAs.type.loc33: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C) [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %.loc33_13.1: %.14 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%C) [template = constants.%.15]
// CHECK:STDOUT:   %Convert.ref.loc33: %.14 = name_ref Convert, %.loc33_13.1 [template = constants.%.15]
// CHECK:STDOUT:   %.loc33_13.2: %C = converted %b.ref, <error> [template = <error>]
// CHECK:STDOUT:   %c: %C = bind_name c, <error>
// CHECK:STDOUT:   %MakeC.ref: %MakeC.type = name_ref MakeC, file.%MakeC.decl [template = constants.%MakeC]
// CHECK:STDOUT:   %.loc46_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %MakeC.call: init %C = call %MakeC.ref() to %.loc46_22.1
// CHECK:STDOUT:   %ImplicitAs.type.loc46: type = interface_type @ImplicitAs, @ImplicitAs(constants.%AdaptC) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc46_24.1: %.11 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%AdaptC) [template = constants.%.12]
// CHECK:STDOUT:   %Convert.ref.loc46: %.11 = name_ref Convert, %.loc46_24.1 [template = constants.%.12]
// CHECK:STDOUT:   %.loc46_22.2: ref %C = temporary %.loc46_22.1, %MakeC.call
// CHECK:STDOUT:   %.loc46_24.2: %AdaptC = converted %MakeC.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   %MakeAdaptC.ref: %MakeAdaptC.type = name_ref MakeAdaptC, file.%MakeAdaptC.decl [template = constants.%MakeAdaptC]
// CHECK:STDOUT:   %.loc54_22.1: ref %AdaptC = temporary_storage
// CHECK:STDOUT:   %MakeAdaptC.call: init %AdaptC = call %MakeAdaptC.ref() to %.loc54_22.1
// CHECK:STDOUT:   %ImplicitAs.type.loc54: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C) [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %.loc54_24.1: %.14 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%C) [template = constants.%.15]
// CHECK:STDOUT:   %Convert.ref.loc54: %.14 = name_ref Convert, %.loc54_24.1 [template = constants.%.15]
// CHECK:STDOUT:   %.loc54_22.2: ref %AdaptC = temporary %.loc54_22.1, %MakeAdaptC.call
// CHECK:STDOUT:   %.loc54_24.2: %C = converted %MakeAdaptC.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%e.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%AdaptC) {
// CHECK:STDOUT:   %Dest => constants.%AdaptC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%C) {
// CHECK:STDOUT:   %Dest => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
