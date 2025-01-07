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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [template]
// CHECK:STDOUT:   %complete_type.f45: <witness> = complete_type_witness %struct_type.j.k [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:   %.loc12: %Class.elem = field_decl j, element0 [template]
// CHECK:STDOUT:   %.loc13: %Class.elem = field_decl k, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.j.k [template = constants.%complete_type.f45]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc12
// CHECK:STDOUT:   .k = %.loc13
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %Class = var c
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc18: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc18: %Class.elem = name_ref j, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc18_4: ref %i32 = class_element_access %c.ref.loc18, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc18: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc18: <bound method> = bound_method %int_1, %impl.elem0.loc18 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc18: <specific function> = specific_function %Convert.bound.loc18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %Convert.specific_fn.loc18(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc18_7: init %i32 = converted %int_1, %int.convert_checked.loc18 [template = constants.%int_1.c60]
// CHECK:STDOUT:   assign %.loc18_4, %.loc18_7
// CHECK:STDOUT:   %c.ref.loc19: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc19: %Class.elem = name_ref k, @Class.%.loc13 [template = @Class.%.loc13]
// CHECK:STDOUT:   %.loc19_4: ref %i32 = class_element_access %c.ref.loc19, element1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc19: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc19: <bound method> = bound_method %int_2, %impl.elem0.loc19 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc19: <specific function> = specific_function %Convert.bound.loc19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %Convert.specific_fn.loc19(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc19_7: init %i32 = converted %int_2, %int.convert_checked.loc19 [template = constants.%int_2.166]
// CHECK:STDOUT:   assign %.loc19_4, %.loc19_7
// CHECK:STDOUT:   %cj.var: ref %i32 = var cj
// CHECK:STDOUT:   %cj: ref %i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   %c.ref.loc20: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc20: %Class.elem = name_ref j, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc20_18.1: ref %i32 = class_element_access %c.ref.loc20, element0
// CHECK:STDOUT:   %.loc20_18.2: %i32 = bind_value %.loc20_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc20_18.2
// CHECK:STDOUT:   %ck.var: ref %i32 = var ck
// CHECK:STDOUT:   %ck: ref %i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %c.ref.loc21: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc21: %Class.elem = name_ref k, @Class.%.loc13 [template = @Class.%.loc13]
// CHECK:STDOUT:   %.loc21_18.1: ref %i32 = class_element_access %c.ref.loc21, element1
// CHECK:STDOUT:   %.loc21_18.2: %i32 = bind_value %.loc21_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc21_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
