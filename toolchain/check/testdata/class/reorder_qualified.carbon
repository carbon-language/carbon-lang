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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %B, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: i32} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %DF.type: type = fn_type @DF [template]
// CHECK:STDOUT:   %DF: %DF.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %D, i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.d: i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %CF.type: type = fn_type @CF [template]
// CHECK:STDOUT:   %CF: %CF.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %C, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %AF.type: type = fn_type @AF [template]
// CHECK:STDOUT:   %AF: %AF.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %.11: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.12: <witness> = complete_type_witness %.11 [template]
// CHECK:STDOUT:   %.14: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.15: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.39: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.14, %Convert.15 [template]
// CHECK:STDOUT:   %.41: i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.1: %A = struct_value (%.41) [template]
// CHECK:STDOUT:   %.43: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.44: type = struct_type {.b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.45: <bound method> = bound_method %.43, %Convert.15 [template]
// CHECK:STDOUT:   %.46: i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct.2: %B = struct_value (%.46) [template]
// CHECK:STDOUT:   %.48: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.49: type = struct_type {.c: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.50: <bound method> = bound_method %.48, %Convert.15 [template]
// CHECK:STDOUT:   %.51: i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value (%.51) [template]
// CHECK:STDOUT:   %.53: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.54: type = struct_type {.d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.55: <bound method> = bound_method %.53, %Convert.15 [template]
// CHECK:STDOUT:   %.56: i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct.4: %D = struct_value (%.56) [template]
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
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc46_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc46_10.2: type = converted %int.make_type_32, %.loc46_10.1 [template = i32]
// CHECK:STDOUT:   %.loc46_8: %.10 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc47: <witness> = complete_type_witness %.11 [template = constants.%.12]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF.decl
// CHECK:STDOUT:   .a = %.loc46_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %BF.decl: %BF.type = fn_decl @BF [template = constants.%BF] {} {}
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_12.2: type = converted %int.make_type_32, %.loc16_12.1 [template = i32]
// CHECK:STDOUT:   %.loc16_10: %.1 = field_decl b, element0 [template]
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %BF.decl
// CHECK:STDOUT:   .b = %.loc16_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [template = constants.%DF] {} {}
// CHECK:STDOUT:   %CF.decl: %CF.type = fn_decl @CF [template = constants.%CF] {} {}
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc42_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc42_12.2: type = converted %int.make_type_32, %.loc42_12.1 [template = i32]
// CHECK:STDOUT:   %.loc42_10: %.7 = field_decl c, element0 [template]
// CHECK:STDOUT:   %.loc43: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF.decl
// CHECK:STDOUT:   .c = %.loc42_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [template = constants.%DF] {} {}
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc24_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc24_14.2: type = converted %int.make_type_32, %.loc24_14.1 [template = i32]
// CHECK:STDOUT:   %.loc24_12: %.4 = field_decl d, element0 [template]
// CHECK:STDOUT:   %.loc25: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .DF = %DF.decl
// CHECK:STDOUT:   .d = %.loc24_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %a.var: ref %A = var a
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   %.loc29_24: Core.IntLiteral = int_value 1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc29_25.1: %.15 = struct_literal (%.loc29_24)
// CHECK:STDOUT:   %.loc29_25.2: %Convert.type.2 = interface_witness_access constants.%.39, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc29_25.3: <bound method> = bound_method %.loc29_24, %.loc29_25.2 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc29: init i32 = call %.loc29_25.3(%.loc29_24) [template = constants.%.41]
// CHECK:STDOUT:   %.loc29_25.4: init i32 = converted %.loc29_24, %int.convert_checked.loc29 [template = constants.%.41]
// CHECK:STDOUT:   %.loc29_25.5: ref i32 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc29_25.6: init i32 = initialize_from %.loc29_25.4 to %.loc29_25.5 [template = constants.%.41]
// CHECK:STDOUT:   %.loc29_25.7: init %A = class_init (%.loc29_25.6), %a.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc29_26: init %A = converted %.loc29_25.1, %.loc29_25.7 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %a.var, %.loc29_26
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %b.var: ref %B = var b
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   %.loc30_24: Core.IntLiteral = int_value 2 [template = constants.%.43]
// CHECK:STDOUT:   %.loc30_25.1: %.44 = struct_literal (%.loc30_24)
// CHECK:STDOUT:   %.loc30_25.2: %Convert.type.2 = interface_witness_access constants.%.39, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc30_25.3: <bound method> = bound_method %.loc30_24, %.loc30_25.2 [template = constants.%.45]
// CHECK:STDOUT:   %int.convert_checked.loc30: init i32 = call %.loc30_25.3(%.loc30_24) [template = constants.%.46]
// CHECK:STDOUT:   %.loc30_25.4: init i32 = converted %.loc30_24, %int.convert_checked.loc30 [template = constants.%.46]
// CHECK:STDOUT:   %.loc30_25.5: ref i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc30_25.6: init i32 = initialize_from %.loc30_25.4 to %.loc30_25.5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc30_25.7: init %B = class_init (%.loc30_25.6), %b.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc30_26: init %B = converted %.loc30_25.1, %.loc30_25.7 [template = constants.%struct.2]
// CHECK:STDOUT:   assign %b.var, %.loc30_26
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc31_24: Core.IntLiteral = int_value 3 [template = constants.%.48]
// CHECK:STDOUT:   %.loc31_25.1: %.49 = struct_literal (%.loc31_24)
// CHECK:STDOUT:   %.loc31_25.2: %Convert.type.2 = interface_witness_access constants.%.39, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc31_25.3: <bound method> = bound_method %.loc31_24, %.loc31_25.2 [template = constants.%.50]
// CHECK:STDOUT:   %int.convert_checked.loc31: init i32 = call %.loc31_25.3(%.loc31_24) [template = constants.%.51]
// CHECK:STDOUT:   %.loc31_25.4: init i32 = converted %.loc31_24, %int.convert_checked.loc31 [template = constants.%.51]
// CHECK:STDOUT:   %.loc31_25.5: ref i32 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc31_25.6: init i32 = initialize_from %.loc31_25.4 to %.loc31_25.5 [template = constants.%.51]
// CHECK:STDOUT:   %.loc31_25.7: init %C = class_init (%.loc31_25.6), %c.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc31_26: init %C = converted %.loc31_25.1, %.loc31_25.7 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %c.var, %.loc31_26
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %d.var: ref %D = var d
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   %.loc32_24: Core.IntLiteral = int_value 4 [template = constants.%.53]
// CHECK:STDOUT:   %.loc32_25.1: %.54 = struct_literal (%.loc32_24)
// CHECK:STDOUT:   %.loc32_25.2: %Convert.type.2 = interface_witness_access constants.%.39, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc32_25.3: <bound method> = bound_method %.loc32_24, %.loc32_25.2 [template = constants.%.55]
// CHECK:STDOUT:   %int.convert_checked.loc32: init i32 = call %.loc32_25.3(%.loc32_24) [template = constants.%.56]
// CHECK:STDOUT:   %.loc32_25.4: init i32 = converted %.loc32_24, %int.convert_checked.loc32 [template = constants.%.56]
// CHECK:STDOUT:   %.loc32_25.5: ref i32 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc32_25.6: init i32 = initialize_from %.loc32_25.4 to %.loc32_25.5 [template = constants.%.56]
// CHECK:STDOUT:   %.loc32_25.7: init %D = class_init (%.loc32_25.6), %d.var [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc32_26: init %D = converted %.loc32_25.1, %.loc32_25.7 [template = constants.%struct.4]
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
