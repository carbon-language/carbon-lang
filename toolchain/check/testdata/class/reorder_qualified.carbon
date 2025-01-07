// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/reorder_qualified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/reorder_qualified.carbon

class A {
  class B {
    class C;

    fn BF();
    var b: i32;
  }

  class B.C {
    class D {
      fn F();

      fn DF();
      var d: i32;
    }

    fn D.DF() {
      // A, B, C, and D are complete here.
      var a: A = {.a = 1};
      var b: B = {.b = 2};
      var c: C = {.c = 3};
      var d: D = {.d = 4};

      // Unqualified lookup looks in all of them.
      AF();
      BF();
      CF();
      DF();
    }

    fn CF();
    var c: i32;
  }

  fn AF();
  var a: i32;
}

// CHECK:STDOUT: --- reorder_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %BF.type: type = fn_type @BF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %BF: %BF.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %i32 [template]
// CHECK:STDOUT:   %struct_type.b.fc0: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT:   %complete_type.d62: <witness> = complete_type_witness %struct_type.b.fc0 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %DF.type: type = fn_type @DF [template]
// CHECK:STDOUT:   %DF: %DF.type = struct_value () [template]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [template]
// CHECK:STDOUT:   %struct_type.d.227: type = struct_type {.d: %i32} [template]
// CHECK:STDOUT:   %complete_type.23b: <witness> = complete_type_witness %struct_type.d.227 [template]
// CHECK:STDOUT:   %CF.type: type = fn_type @CF [template]
// CHECK:STDOUT:   %CF: %CF.type = struct_value () [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %struct_type.c.8db: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %complete_type.da4: <witness> = complete_type_witness %struct_type.c.8db [template]
// CHECK:STDOUT:   %AF.type: type = fn_type @AF [template]
// CHECK:STDOUT:   %AF: %AF.type = struct_value () [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.a.7cc: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %complete_type.297: <witness> = complete_type_witness %struct_type.a.7cc [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.c60) [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %B.val: %B = struct_value (%int_2.166) [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %struct_type.c.5b8: type = struct_type {.c: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.bound.0db: <bound method> = bound_method %int_3.1ba, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.456: <specific function> = specific_function %Convert.bound.0db, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25b: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_3.25b) [template]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %struct_type.d.3ea: type = struct_type {.d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.bound.fe5: <bound method> = bound_method %int_4.0c1, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c6f: <specific function> = specific_function %Convert.bound.fe5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2ec: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_4.2ec) [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %AF.decl: %AF.type = fn_decl @AF [template = constants.%AF] {} {}
// CHECK:STDOUT:   %.loc46: %A.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.7cc [template = constants.%complete_type.297]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF.decl
// CHECK:STDOUT:   .a = %.loc46
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %BF.decl: %BF.type = fn_decl @BF [template = constants.%BF] {} {}
// CHECK:STDOUT:   %.loc16: %B.elem = field_decl b, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.b.fc0 [template = constants.%complete_type.d62]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %BF.decl
// CHECK:STDOUT:   .b = %.loc16
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [template = constants.%DF] {} {}
// CHECK:STDOUT:   %CF.decl: %CF.type = fn_decl @CF [template = constants.%CF] {} {}
// CHECK:STDOUT:   %.loc42: %C.elem = field_decl c, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c.8db [template = constants.%complete_type.da4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF.decl
// CHECK:STDOUT:   .c = %.loc42
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [template = constants.%DF] {} {}
// CHECK:STDOUT:   %.loc24: %D.elem = field_decl d, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.d.227 [template = constants.%complete_type.23b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .DF = %DF.decl
// CHECK:STDOUT:   .d = %.loc24
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var a
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc29_25.1: %struct_type.a.a6c = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0.loc29: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc29: <bound method> = bound_method %int_1, %impl.elem0.loc29 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc29: <specific function> = specific_function %Convert.bound.loc29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32 = call %Convert.specific_fn.loc29(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc29_25.2: init %i32 = converted %int_1, %int.convert_checked.loc29 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc29_25.3: ref %i32 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc29_25.4: init %i32 = initialize_from %.loc29_25.2 to %.loc29_25.3 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc29_25.5: init %A = class_init (%.loc29_25.4), %a.var [template = constants.%A.val]
// CHECK:STDOUT:   %.loc29_26: init %A = converted %.loc29_25.1, %.loc29_25.5 [template = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc29_26
// CHECK:STDOUT:   %b.var: ref %B = var b
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc30_25.1: %struct_type.b.a15 = struct_literal (%int_2)
// CHECK:STDOUT:   %impl.elem0.loc30: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc30: <bound method> = bound_method %int_2, %impl.elem0.loc30 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc30: <specific function> = specific_function %Convert.bound.loc30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc30: init %i32 = call %Convert.specific_fn.loc30(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc30_25.2: init %i32 = converted %int_2, %int.convert_checked.loc30 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc30_25.3: ref %i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc30_25.4: init %i32 = initialize_from %.loc30_25.2 to %.loc30_25.3 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc30_25.5: init %B = class_init (%.loc30_25.4), %b.var [template = constants.%B.val]
// CHECK:STDOUT:   %.loc30_26: init %B = converted %.loc30_25.1, %.loc30_25.5 [template = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc30_26
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc31_25.1: %struct_type.c.5b8 = struct_literal (%int_3)
// CHECK:STDOUT:   %impl.elem0.loc31: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc31: <bound method> = bound_method %int_3, %impl.elem0.loc31 [template = constants.%Convert.bound.0db]
// CHECK:STDOUT:   %Convert.specific_fn.loc31: <specific function> = specific_function %Convert.bound.loc31, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.456]
// CHECK:STDOUT:   %int.convert_checked.loc31: init %i32 = call %Convert.specific_fn.loc31(%int_3) [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc31_25.2: init %i32 = converted %int_3, %int.convert_checked.loc31 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc31_25.3: ref %i32 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc31_25.4: init %i32 = initialize_from %.loc31_25.2 to %.loc31_25.3 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc31_25.5: init %C = class_init (%.loc31_25.4), %c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc31_26: init %C = converted %.loc31_25.1, %.loc31_25.5 [template = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc31_26
// CHECK:STDOUT:   %d.var: ref %D = var d
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc32_25.1: %struct_type.d.3ea = struct_literal (%int_4)
// CHECK:STDOUT:   %impl.elem0.loc32: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc32: <bound method> = bound_method %int_4, %impl.elem0.loc32 [template = constants.%Convert.bound.fe5]
// CHECK:STDOUT:   %Convert.specific_fn.loc32: <specific function> = specific_function %Convert.bound.loc32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c6f]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32 = call %Convert.specific_fn.loc32(%int_4) [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc32_25.2: init %i32 = converted %int_4, %int.convert_checked.loc32 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc32_25.3: ref %i32 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc32_25.4: init %i32 = initialize_from %.loc32_25.2 to %.loc32_25.3 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc32_25.5: init %D = class_init (%.loc32_25.4), %d.var [template = constants.%D.val]
// CHECK:STDOUT:   %.loc32_26: init %D = converted %.loc32_25.1, %.loc32_25.5 [template = constants.%D.val]
// CHECK:STDOUT:   assign %d.var, %.loc32_26
// CHECK:STDOUT:   %AF.ref: %AF.type = name_ref AF, @A.%AF.decl [template = constants.%AF]
// CHECK:STDOUT:   %AF.call: init %empty_tuple.type = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: %BF.type = name_ref BF, @B.%BF.decl [template = constants.%BF]
// CHECK:STDOUT:   %BF.call: init %empty_tuple.type = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: %CF.type = name_ref CF, @C.%CF.decl [template = constants.%CF]
// CHECK:STDOUT:   %CF.call: init %empty_tuple.type = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: %DF.type = name_ref DF, @D.%DF.decl [template = constants.%DF]
// CHECK:STDOUT:   %DF.call: init %empty_tuple.type = call %DF.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AF();
// CHECK:STDOUT:
