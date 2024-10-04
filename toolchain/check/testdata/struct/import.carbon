// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;
var c: C({.a = 1, .b = 2}) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: missing value for field `a` in struct initialization
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:6: in import
// CHECK:STDERR: implicit.carbon:8:9: note: initializing generic parameter `S` declared here
// CHECK:STDERR: class C(S:! {.a: i32, .b: i32}) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `C` to `C`
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+3]]:1: note: type `C` does not implement interface `ImplicitAs`
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var c_bad: C({.a = 3, .b = 4}) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %struct.1: %.2 = struct_value (%.3) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: type = struct_type {.b: i32, .c: %.5} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: %.6, .d: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.a: %.8, .d: i32} [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %tuple: %.5 = tuple_value (%.3) [template]
// CHECK:STDOUT:   %struct.2: %.6 = struct_value (%.3, %tuple) [template]
// CHECK:STDOUT:   %struct.3: %.7 = struct_value (%struct.2, %.3) [template]
// CHECK:STDOUT:   %.11: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %S: %.11 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %.12: type = struct_type {} [template]
// CHECK:STDOUT:   %.13: <witness> = complete_type_witness %.12 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %struct.4: %.11 = struct_value (%.14, %.15) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct.4) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc4_17.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_17.2: type = converted %int.make_type_32.loc4, %.loc4_17.1 [template = i32]
// CHECK:STDOUT:   %.loc4_20: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref %.2 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %.2 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %int.make_type_32.loc5_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_22.1: type = value_of_initializer %int.make_type_32.loc5_22 [template = i32]
// CHECK:STDOUT:   %.loc5_22.2: type = converted %int.make_type_32.loc5_22, %.loc5_22.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc5_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_36.1: %.4 = tuple_literal (%int.make_type_32.loc5_32)
// CHECK:STDOUT:   %.loc5_36.2: type = value_of_initializer %int.make_type_32.loc5_32 [template = i32]
// CHECK:STDOUT:   %.loc5_36.3: type = converted %int.make_type_32.loc5_32, %.loc5_36.2 [template = i32]
// CHECK:STDOUT:   %.loc5_36.4: type = converted %.loc5_36.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_37: type = struct_type {.b: i32, .c: %.5} [template = constants.%.6]
// CHECK:STDOUT:   %int.make_type_32.loc5_44: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_44.1: type = value_of_initializer %int.make_type_32.loc5_44 [template = i32]
// CHECK:STDOUT:   %.loc5_44.2: type = converted %int.make_type_32.loc5_44, %.loc5_44.1 [template = i32]
// CHECK:STDOUT:   %.loc5_47: type = struct_type {.a: %.6, .d: i32} [template = constants.%.7]
// CHECK:STDOUT:   %b_ref.var: ref %.7 = var b_ref
// CHECK:STDOUT:   %b_ref: ref %.7 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %S.patt: %.11 = symbolic_binding_pattern S, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc8_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_18.1: type = value_of_initializer %int.make_type_32.loc8_18 [template = i32]
// CHECK:STDOUT:     %.loc8_18.2: type = converted %int.make_type_32.loc8_18, %.loc8_18.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc8_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_27.1: type = value_of_initializer %int.make_type_32.loc8_27 [template = i32]
// CHECK:STDOUT:     %.loc8_27.2: type = converted %int.make_type_32.loc8_27, %.loc8_27.1 [template = i32]
// CHECK:STDOUT:     %.loc8_30: type = struct_type {.a: i32, .b: i32} [template = constants.%.11]
// CHECK:STDOUT:     %S.param: %.11 = param S, runtime_param<invalid>
// CHECK:STDOUT:     %S.loc8: %.11 = bind_symbolic_name S, 0, %S.param [symbolic = %S.1 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc9_19: i32 = int_literal 1 [template = constants.%.14]
// CHECK:STDOUT:     %.loc9_27: i32 = int_literal 2 [template = constants.%.15]
// CHECK:STDOUT:     %.loc9_28: %.11 = struct_literal (%.loc9_19, %.loc9_27)
// CHECK:STDOUT:     %struct: %.11 = struct_value (%.loc9_19, %.loc9_27) [template = constants.%struct.4]
// CHECK:STDOUT:     %.loc9_12: %.11 = converted %.loc9_28, %struct [template = constants.%struct.4]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.4) [template = constants.%C.3]
// CHECK:STDOUT:     %return: ref %C.3 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%S.loc8: %.11) {
// CHECK:STDOUT:   %S.1: %.11 = bind_symbolic_name S, 0 [symbolic = %S.1 (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc8_34: <witness> = complete_type_witness %.12 [template = constants.%.13]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_30: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_31.1: %.2 = struct_literal (%.loc4_30)
// CHECK:STDOUT:   %.loc4_31.2: init %.2 = struct_init (%.loc4_30) to file.%a_ref.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc4_32: init %.2 = converted %.loc4_31.1, %.loc4_31.2 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_32
// CHECK:STDOUT:   %.loc6_17: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_26: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_28.1: %.5 = tuple_literal (%.loc6_26)
// CHECK:STDOUT:   %.loc6_29.1: %.6 = struct_literal (%.loc6_17, %.loc6_28.1)
// CHECK:STDOUT:   %.loc6_37: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_38.1: %.7 = struct_literal (%.loc6_29.1, %.loc6_37)
// CHECK:STDOUT:   %.loc6_38.2: ref %.6 = struct_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc6_29.2: ref i32 = struct_access %.loc6_38.2, element0
// CHECK:STDOUT:   %.loc6_29.3: init i32 = initialize_from %.loc6_17 to %.loc6_29.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_29.4: ref %.5 = struct_access %.loc6_38.2, element1
// CHECK:STDOUT:   %.loc6_28.2: init %.5 = tuple_init (%.loc6_26) to %.loc6_29.4 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.5: init %.5 = converted %.loc6_28.1, %.loc6_28.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.6: init %.5 = initialize_from %.loc6_29.5 to %.loc6_29.4 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.7: init %.6 = struct_init (%.loc6_29.3, %.loc6_29.6) to %.loc6_38.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc6_38.3: init %.6 = converted %.loc6_29.1, %.loc6_29.7 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc6_38.4: ref i32 = struct_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc6_38.5: init i32 = initialize_from %.loc6_37 to %.loc6_38.4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_38.6: init %.7 = struct_init (%.loc6_38.3, %.loc6_38.5) to file.%b_ref.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc6_39: init %.7 = converted %.loc6_38.1, %.loc6_38.6 [template = constants.%struct.3]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc6_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S.1 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.4) {
// CHECK:STDOUT:   %S.1 => constants.%struct.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.5: type = struct_type {.b: i32, .c: %.4} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: %.5, .d: i32} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.a: %.7, .d: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {} [template]
// CHECK:STDOUT:   %.11: <witness> = complete_type_witness %.10 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %S: %.12 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %.13: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.15: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %struct: %.12 = struct_value (%.13, %.14) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.10 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %.2 = import_ref Implicit//default, inst+17, loaded
// CHECK:STDOUT:   %import_ref.2: ref %.6 = import_ref Implicit//default, inst+57, loaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+100, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+123, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.6 = import_ref Implicit//default, inst+104, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc4_13.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_13.2: type = converted %int.make_type_32.loc4, %.loc4_13.1 [template = i32]
// CHECK:STDOUT:   %.loc4_16: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc5_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18.1: type = value_of_initializer %int.make_type_32.loc5_18 [template = i32]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %int.make_type_32.loc5_18, %.loc5_18.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc5_28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_32.1: %.3 = tuple_literal (%int.make_type_32.loc5_28)
// CHECK:STDOUT:   %.loc5_32.2: type = value_of_initializer %int.make_type_32.loc5_28 [template = i32]
// CHECK:STDOUT:   %.loc5_32.3: type = converted %int.make_type_32.loc5_28, %.loc5_32.2 [template = i32]
// CHECK:STDOUT:   %.loc5_32.4: type = converted %.loc5_32.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_33: type = struct_type {.b: i32, .c: %.4} [template = constants.%.5]
// CHECK:STDOUT:   %int.make_type_32.loc5_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_40.1: type = value_of_initializer %int.make_type_32.loc5_40 [template = i32]
// CHECK:STDOUT:   %.loc5_40.2: type = converted %int.make_type_32.loc5_40, %.loc5_40.1 [template = i32]
// CHECK:STDOUT:   %.loc5_43: type = struct_type {.a: %.5, .d: i32} [template = constants.%.6]
// CHECK:STDOUT:   %b.var: ref %.6 = var b
// CHECK:STDOUT:   %b: ref %.6 = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc6_16: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc6_24: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc6_25: %.12 = struct_literal (%.loc6_16, %.loc6_24)
// CHECK:STDOUT:   %struct: %.12 = struct_value (%.loc6_16, %.loc6_24) [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_9: %.12 = converted %.loc6_25, %struct [template = constants.%struct]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%struct) [template = constants.%C.3]
// CHECK:STDOUT:   %c.var: ref %C.3 = var c
// CHECK:STDOUT:   %c: ref %C.3 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %.12) {
// CHECK:STDOUT:   %S: %.12 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %.2 = name_ref a_ref, imports.%import_ref.1
// CHECK:STDOUT:   %.loc4_20.1: ref i32 = struct_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_20.2: i32 = bind_value %.loc4_20.1
// CHECK:STDOUT:   %.loc4_20.3: init %.2 = struct_init (%.loc4_20.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_25: init %.2 = converted %a_ref.ref, %.loc4_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_25
// CHECK:STDOUT:   %b_ref.ref: ref %.6 = name_ref b_ref, imports.%import_ref.2
// CHECK:STDOUT:   %.loc5_47.1: ref %.5 = struct_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_47.2: ref i32 = struct_access %.loc5_47.1, element0
// CHECK:STDOUT:   %.loc5_47.3: i32 = bind_value %.loc5_47.2
// CHECK:STDOUT:   %.loc5_47.4: ref %.5 = struct_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_47.5: ref i32 = struct_access %.loc5_47.4, element0
// CHECK:STDOUT:   %.loc5_47.6: init i32 = initialize_from %.loc5_47.3 to %.loc5_47.5
// CHECK:STDOUT:   %.loc5_47.7: ref %.4 = struct_access %.loc5_47.1, element1
// CHECK:STDOUT:   %.loc5_47.8: ref i32 = tuple_access %.loc5_47.7, element0
// CHECK:STDOUT:   %.loc5_47.9: i32 = bind_value %.loc5_47.8
// CHECK:STDOUT:   %.loc5_47.10: ref %.4 = struct_access %.loc5_47.4, element1
// CHECK:STDOUT:   %.loc5_47.11: init %.4 = tuple_init (%.loc5_47.9) to %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.12: init %.4 = converted %.loc5_47.7, %.loc5_47.11
// CHECK:STDOUT:   %.loc5_47.13: init %.4 = initialize_from %.loc5_47.12 to %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.14: init %.5 = struct_init (%.loc5_47.6, %.loc5_47.13) to %.loc5_47.4
// CHECK:STDOUT:   %.loc5_47.15: init %.5 = converted %.loc5_47.1, %.loc5_47.14
// CHECK:STDOUT:   %.loc5_47.16: ref i32 = struct_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_47.17: i32 = bind_value %.loc5_47.16
// CHECK:STDOUT:   %.loc5_47.18: ref i32 = struct_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_47.19: init i32 = initialize_from %.loc5_47.17 to %.loc5_47.18
// CHECK:STDOUT:   %.loc5_47.20: init %.6 = struct_init (%.loc5_47.15, %.loc5_47.19) to file.%b.var
// CHECK:STDOUT:   %.loc5_52: init %.6 = converted %b_ref.ref, %.loc5_47.20
// CHECK:STDOUT:   assign file.%b.var, %.loc5_52
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.3 = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_type.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %S: %.4 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.c: i32, .d: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %.4 = struct_value (%.5, %.6) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+17, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+57, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+100, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+123, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+104, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_28: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_29: %.7 = struct_literal (%.loc11_20, %.loc11_28)
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %c_bad: ref <error> = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %.4) {
// CHECK:STDOUT:   %S: %.4 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc11: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc11
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %S: %.4 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %.5: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %struct.1: %.4 = struct_value (%.5, %.6) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct.1) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %struct.2: %.4 = struct_value (%.10, %.9) [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%struct.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.10 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%C.3) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%C.3) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %.15: %.11 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+17, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+57, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+100, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+123, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+104, unloaded
// CHECK:STDOUT:   %import_ref.6: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%.1 (%.11) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc9_20: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_28: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_29: %.4 = struct_literal (%.loc9_20, %.loc9_28)
// CHECK:STDOUT:   %struct: %.4 = struct_value (%.loc9_20, %.loc9_28) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc9_13: %.4 = converted %.loc9_29, %struct [template = constants.%struct.1]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%struct.1) [template = constants.%C.3]
// CHECK:STDOUT:   %c_bad.var: ref %C.3 = var c_bad
// CHECK:STDOUT:   %c_bad: ref %C.3 = bind_name c_bad, %c_bad.var
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
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.11) = assoc_entity element0, imports.%import_ref.10 [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Convert = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %.4) {
// CHECK:STDOUT:   %S: %.4 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.4;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc9_35.1: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.4 = call %F.ref() to %.loc9_35.1
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C.3) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc9_37.1: %.13 = specific_constant imports.%import_ref.8, @ImplicitAs(constants.%C.3) [template = constants.%.14]
// CHECK:STDOUT:   %Convert.ref: %.13 = name_ref Convert, %.loc9_37.1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc9_35.2: ref %C.4 = temporary %.loc9_35.1, %F.call
// CHECK:STDOUT:   %.loc9_37.2: %C.3 = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.1) {
// CHECK:STDOUT:   %S => constants.%struct.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.2) {
// CHECK:STDOUT:   %S => constants.%struct.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
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
// CHECK:STDOUT: specific @ImplicitAs(constants.%C.3) {
// CHECK:STDOUT:   %Dest => constants.%C.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %.2 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
