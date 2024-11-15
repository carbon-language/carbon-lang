// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field_access.carbon

class Class {
  var j: i32;
  var k: i32;
}

fn Run() {
  var c: Class;
  c.j = 1;
  c.k = 2;
  var cj: i32 = c.j;
  var ck: i32 = c.k;
}

// CHECK:STDOUT: --- field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.32, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.1 = field_decl j, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.1 = field_decl k, element1 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc12_8
// CHECK:STDOUT:   .k = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref %Class = var c
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc18: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc18: %.1 = name_ref j, @Class.%.loc12_8 [template = @Class.%.loc12_8]
// CHECK:STDOUT:   %.loc18_4: ref i32 = class_element_access %c.ref.loc18, element0
// CHECK:STDOUT:   %.loc18_9: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_7.1: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc18_7.2: <bound method> = bound_method %.loc18_9, %.loc18_7.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc18: init i32 = call %.loc18_7.2(%.loc18_9) [template = constants.%.31]
// CHECK:STDOUT:   %.loc18_7.3: init i32 = converted %.loc18_9, %int.convert_checked.loc18 [template = constants.%.31]
// CHECK:STDOUT:   assign %.loc18_4, %.loc18_7.3
// CHECK:STDOUT:   %c.ref.loc19: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc19: %.1 = name_ref k, @Class.%.loc13_8 [template = @Class.%.loc13_8]
// CHECK:STDOUT:   %.loc19_4: ref i32 = class_element_access %c.ref.loc19, element1
// CHECK:STDOUT:   %.loc19_9: Core.IntLiteral = int_value 2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_7.1: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc19_7.2: <bound method> = bound_method %.loc19_9, %.loc19_7.1 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc19: init i32 = call %.loc19_7.2(%.loc19_9) [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_7.3: init i32 = converted %.loc19_9, %int.convert_checked.loc19 [template = constants.%.34]
// CHECK:STDOUT:   assign %.loc19_4, %.loc19_7.3
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc20_11.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_11.2: type = converted %int.make_type_32.loc20, %.loc20_11.1 [template = i32]
// CHECK:STDOUT:   %cj.var: ref i32 = var cj
// CHECK:STDOUT:   %cj: ref i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   %c.ref.loc20: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc20: %.1 = name_ref j, @Class.%.loc12_8 [template = @Class.%.loc12_8]
// CHECK:STDOUT:   %.loc20_18.1: ref i32 = class_element_access %c.ref.loc20, element0
// CHECK:STDOUT:   %.loc20_18.2: i32 = bind_value %.loc20_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc20_18.2
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_11.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_11.2: type = converted %int.make_type_32.loc21, %.loc21_11.1 [template = i32]
// CHECK:STDOUT:   %ck.var: ref i32 = var ck
// CHECK:STDOUT:   %ck: ref i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %c.ref.loc21: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc21: %.1 = name_ref k, @Class.%.loc13_8 [template = @Class.%.loc13_8]
// CHECK:STDOUT:   %.loc21_18.1: ref i32 = class_element_access %c.ref.loc21, element1
// CHECK:STDOUT:   %.loc21_18.2: i32 = bind_value %.loc21_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc21_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
