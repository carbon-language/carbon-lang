// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field_access_in_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field_access_in_value.carbon

class Class {
  var j: i32;
  var k: i32;
}

fn Test() {
  var cv: Class;
  cv.j = 1;
  cv.k = 2;
  let c: Class = cv;
  var cj: i32 = c.j;
  var ck: i32 = c.k;
}

// CHECK:STDOUT: --- field_access_in_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.j.k [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %Class.elem = field_decl j, element0 [template]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_signed.loc13, %.loc13_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8: %Class.elem = field_decl k, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.j.k [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc12_8
// CHECK:STDOUT:   .k = %.loc13_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc17: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %cv.var: ref %Class = var cv
// CHECK:STDOUT:   %cv: ref %Class = bind_name cv, %cv.var
// CHECK:STDOUT:   %cv.ref.loc18: ref %Class = name_ref cv, %cv
// CHECK:STDOUT:   %j.ref.loc18: %Class.elem = name_ref j, @Class.%.loc12_8 [template = @Class.%.loc12_8]
// CHECK:STDOUT:   %.loc18_5: ref %i32 = class_element_access %cv.ref.loc18, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc18: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18: <bound method> = bound_method %int_1, %impl.elem0.loc18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18: <specific function> = specific_function %Convert.bound.loc18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %Convert.specific_fn.loc18(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_8: init %i32 = converted %int_1, %int.convert_checked.loc18 [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %.loc18_5, %.loc18_8
// CHECK:STDOUT:   %cv.ref.loc19: ref %Class = name_ref cv, %cv
// CHECK:STDOUT:   %k.ref.loc19: %Class.elem = name_ref k, @Class.%.loc13_8 [template = @Class.%.loc13_8]
// CHECK:STDOUT:   %.loc19_5: ref %i32 = class_element_access %cv.ref.loc19, element1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc19: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc19: <bound method> = bound_method %int_2, %impl.elem0.loc19 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc19: <specific function> = specific_function %Convert.bound.loc19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %Convert.specific_fn.loc19(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc19_8: init %i32 = converted %int_2, %int.convert_checked.loc19 [template = constants.%int_2.2]
// CHECK:STDOUT:   assign %.loc19_5, %.loc19_8
// CHECK:STDOUT:   %Class.ref.loc20: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %cv.ref.loc20: ref %Class = name_ref cv, %cv
// CHECK:STDOUT:   %.loc20: %Class = bind_value %cv.ref.loc20
// CHECK:STDOUT:   %c: %Class = bind_name c, %.loc20
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int_32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_11.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_11.2: type = converted %int.make_type_signed.loc21, %.loc21_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %cj.var: ref %i32 = var cj
// CHECK:STDOUT:   %cj: ref %i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   %c.ref.loc21: %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc21: %Class.elem = name_ref j, @Class.%.loc12_8 [template = @Class.%.loc12_8]
// CHECK:STDOUT:   %.loc21_18.1: ref %i32 = class_element_access %c.ref.loc21, element0
// CHECK:STDOUT:   %.loc21_18.2: %i32 = bind_value %.loc21_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc21_18.2
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%int_32.loc22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_11.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %int.make_type_signed.loc22, %.loc22_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %ck.var: ref %i32 = var ck
// CHECK:STDOUT:   %ck: ref %i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %c.ref.loc22: %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc22: %Class.elem = name_ref k, @Class.%.loc13_8 [template = @Class.%.loc13_8]
// CHECK:STDOUT:   %.loc22_18.1: ref %i32 = class_element_access %c.ref.loc22, element1
// CHECK:STDOUT:   %.loc22_18.2: %i32 = bind_value %.loc22_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc22_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
