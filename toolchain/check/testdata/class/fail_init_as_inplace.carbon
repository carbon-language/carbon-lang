// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_init_as_inplace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/class/fail_init_as_inplace.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn G(p: Class*);

fn F() {
  // TODO: This case should presumably work: `{...} as Class` should be an
  // initializing expression, not a value expression.
  //
  // CHECK:STDERR: fail_init_as_inplace.carbon:[[@LINE+3]]:18: ERROR: Cannot copy value of type `Class`.
  // CHECK:STDERR:   var c: Class = {.a = 1, .b = 2} as Class;
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~~
  var c: Class = {.a = 1, .b = 2} as Class;
  G(&c);
}

// CHECK:STDOUT: --- fail_init_as_inplace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type Class [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.4: Class = struct_value (%.6, %.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16: type = ptr_type Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc16_6.1: Class* = param p
// CHECK:STDOUT:     @G.%p: Class* = bind_name p, %p.loc16_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: <unbound element of class Class> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: Class*);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc25_10: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %.loc25_24: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc25_32: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc25_33.1: {.a: i32, .b: i32} = struct_literal (%.loc25_24, %.loc25_32)
// CHECK:STDOUT:   %Class.ref.loc25_38: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc25_33.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc25_33.3: ref i32 = class_element_access %.loc25_33.2, element0
// CHECK:STDOUT:   %.loc25_33.4: init i32 = initialize_from %.loc25_24 to %.loc25_33.3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc25_33.5: ref i32 = class_element_access %.loc25_33.2, element1
// CHECK:STDOUT:   %.loc25_33.6: init i32 = initialize_from %.loc25_32 to %.loc25_33.5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc25_33.7: init Class = class_init (%.loc25_33.4, %.loc25_33.6), %.loc25_33.2 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc25_33.8: ref Class = temporary %.loc25_33.2, %.loc25_33.7
// CHECK:STDOUT:   %.loc25_35.1: ref Class = converted %.loc25_33.1, %.loc25_33.8
// CHECK:STDOUT:   %.loc25_35.2: Class = bind_value %.loc25_35.1
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc26: Class* = addr_of %c.ref
// CHECK:STDOUT:   %G.call: init () = call %G.ref(%.loc26)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
