// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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
// CHECK:STDOUT:   %BF: type = fn_type @BF [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: BF = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type B, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.b: i32} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %DF: type = fn_type @DF [template]
// CHECK:STDOUT:   %struct.3: DF = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type D, i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.d: i32} [template]
// CHECK:STDOUT:   %CF: type = fn_type @CF [template]
// CHECK:STDOUT:   %struct.4: CF = struct_value () [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %AF: type = fn_type @AF [template]
// CHECK:STDOUT:   %struct.5: AF = struct_value () [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.10: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %struct.6: A = struct_value (%.11) [template]
// CHECK:STDOUT:   %.12: type = ptr_type {.b: i32} [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.7: B = struct_value (%.13) [template]
// CHECK:STDOUT:   %.14: type = ptr_type {.c: i32} [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %struct.8: C = struct_value (%.15) [template]
// CHECK:STDOUT:   %.16: type = ptr_type {.d: i32} [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %struct.9: D = struct_value (%.17) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %AF.decl: AF = fn_decl @AF [template = constants.%struct.5] {}
// CHECK:STDOUT:   %.loc42: <unbound element of class A> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF.decl
// CHECK:STDOUT:   .a = %.loc42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %BF.decl: BF = fn_decl @BF [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = field_decl b, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %BF.decl
// CHECK:STDOUT:   .b = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %DF.decl: DF = fn_decl @DF [template = constants.%struct.3] {}
// CHECK:STDOUT:   %CF.decl: CF = fn_decl @CF [template = constants.%struct.4] {}
// CHECK:STDOUT:   %.loc38: <unbound element of class C> = field_decl c, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF.decl
// CHECK:STDOUT:   .c = %.loc38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {}
// CHECK:STDOUT:   %DF.decl: DF = fn_decl @DF [template = constants.%struct.3] {}
// CHECK:STDOUT:   %.loc20: <unbound element of class D> = field_decl d, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .DF = %DF.decl
// CHECK:STDOUT:   .d = %.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %a.var: ref A = var a
// CHECK:STDOUT:   %a: ref A = bind_name a, %a.var
// CHECK:STDOUT:   %.loc25_24: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc25_25.1: {.a: i32} = struct_literal (%.loc25_24)
// CHECK:STDOUT:   %.loc25_25.2: ref i32 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc25_25.3: init i32 = initialize_from %.loc25_24 to %.loc25_25.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc25_25.4: init A = class_init (%.loc25_25.3), %a.var [template = constants.%struct.6]
// CHECK:STDOUT:   %.loc25_26: init A = converted %.loc25_25.1, %.loc25_25.4 [template = constants.%struct.6]
// CHECK:STDOUT:   assign %a.var, %.loc25_26
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %b.var: ref B = var b
// CHECK:STDOUT:   %b: ref B = bind_name b, %b.var
// CHECK:STDOUT:   %.loc26_24: i32 = int_literal 2 [template = constants.%.13]
// CHECK:STDOUT:   %.loc26_25.1: {.b: i32} = struct_literal (%.loc26_24)
// CHECK:STDOUT:   %.loc26_25.2: ref i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc26_25.3: init i32 = initialize_from %.loc26_24 to %.loc26_25.2 [template = constants.%.13]
// CHECK:STDOUT:   %.loc26_25.4: init B = class_init (%.loc26_25.3), %b.var [template = constants.%struct.7]
// CHECK:STDOUT:   %.loc26_26: init B = converted %.loc26_25.1, %.loc26_25.4 [template = constants.%struct.7]
// CHECK:STDOUT:   assign %b.var, %.loc26_26
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc27_24: i32 = int_literal 3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc27_25.1: {.c: i32} = struct_literal (%.loc27_24)
// CHECK:STDOUT:   %.loc27_25.2: ref i32 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc27_25.3: init i32 = initialize_from %.loc27_24 to %.loc27_25.2 [template = constants.%.15]
// CHECK:STDOUT:   %.loc27_25.4: init C = class_init (%.loc27_25.3), %c.var [template = constants.%struct.8]
// CHECK:STDOUT:   %.loc27_26: init C = converted %.loc27_25.1, %.loc27_25.4 [template = constants.%struct.8]
// CHECK:STDOUT:   assign %c.var, %.loc27_26
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %d.var: ref D = var d
// CHECK:STDOUT:   %d: ref D = bind_name d, %d.var
// CHECK:STDOUT:   %.loc28_24: i32 = int_literal 4 [template = constants.%.17]
// CHECK:STDOUT:   %.loc28_25.1: {.d: i32} = struct_literal (%.loc28_24)
// CHECK:STDOUT:   %.loc28_25.2: ref i32 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc28_25.3: init i32 = initialize_from %.loc28_24 to %.loc28_25.2 [template = constants.%.17]
// CHECK:STDOUT:   %.loc28_25.4: init D = class_init (%.loc28_25.3), %d.var [template = constants.%struct.9]
// CHECK:STDOUT:   %.loc28_26: init D = converted %.loc28_25.1, %.loc28_25.4 [template = constants.%struct.9]
// CHECK:STDOUT:   assign %d.var, %.loc28_26
// CHECK:STDOUT:   %AF.ref: AF = name_ref AF, @A.%AF.decl [template = constants.%struct.5]
// CHECK:STDOUT:   %AF.call: init () = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: BF = name_ref BF, @B.%BF.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %BF.call: init () = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: CF = name_ref CF, @C.%CF.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %CF.call: init () = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: DF = name_ref DF, @D.%DF.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %DF.call: init () = call %DF.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AF();
// CHECK:STDOUT:
