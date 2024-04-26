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
// CHECK:STDOUT:   %.1: type = unbound_element_type B, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: i32} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type D, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.d: i32} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %struct.1: A = struct_value (%.10) [template]
// CHECK:STDOUT:   %.11: type = ptr_type {.b: i32} [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.2: B = struct_value (%.12) [template]
// CHECK:STDOUT:   %.13: type = ptr_type {.c: i32} [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %struct.3: C = struct_value (%.14) [template]
// CHECK:STDOUT:   %.15: type = ptr_type {.d: i32} [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %struct.4: D = struct_value (%.16) [template]
// CHECK:STDOUT:   %.17: type = tuple_type () [template]
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
// CHECK:STDOUT:   %AF: <function> = fn_decl @AF [template] {}
// CHECK:STDOUT:   %.loc42: <unbound element of class A> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF
// CHECK:STDOUT:   .a = %.loc42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %BF: <function> = fn_decl @BF [template] {}
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = field_decl b, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %BF
// CHECK:STDOUT:   .b = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %DF: <function> = fn_decl @DF [template] {}
// CHECK:STDOUT:   %CF: <function> = fn_decl @CF [template] {}
// CHECK:STDOUT:   %.loc38: <unbound element of class C> = field_decl c, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF
// CHECK:STDOUT:   .c = %.loc38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %DF: <function> = fn_decl @DF [template] {}
// CHECK:STDOUT:   %.loc20: <unbound element of class D> = field_decl d, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .DF = %DF
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
// CHECK:STDOUT:   %.loc25_24: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc25_25.1: {.a: i32} = struct_literal (%.loc25_24)
// CHECK:STDOUT:   %.loc25_25.2: ref i32 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc25_25.3: init i32 = initialize_from %.loc25_24 to %.loc25_25.2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc25_25.4: init A = class_init (%.loc25_25.3), %a.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc25_26: init A = converted %.loc25_25.1, %.loc25_25.4 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %a.var, %.loc25_26
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %b.var: ref B = var b
// CHECK:STDOUT:   %b: ref B = bind_name b, %b.var
// CHECK:STDOUT:   %.loc26_24: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc26_25.1: {.b: i32} = struct_literal (%.loc26_24)
// CHECK:STDOUT:   %.loc26_25.2: ref i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc26_25.3: init i32 = initialize_from %.loc26_24 to %.loc26_25.2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc26_25.4: init B = class_init (%.loc26_25.3), %b.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc26_26: init B = converted %.loc26_25.1, %.loc26_25.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign %b.var, %.loc26_26
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc27_24: i32 = int_literal 3 [template = constants.%.14]
// CHECK:STDOUT:   %.loc27_25.1: {.c: i32} = struct_literal (%.loc27_24)
// CHECK:STDOUT:   %.loc27_25.2: ref i32 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc27_25.3: init i32 = initialize_from %.loc27_24 to %.loc27_25.2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc27_25.4: init C = class_init (%.loc27_25.3), %c.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc27_26: init C = converted %.loc27_25.1, %.loc27_25.4 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %c.var, %.loc27_26
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %d.var: ref D = var d
// CHECK:STDOUT:   %d: ref D = bind_name d, %d.var
// CHECK:STDOUT:   %.loc28_24: i32 = int_literal 4 [template = constants.%.16]
// CHECK:STDOUT:   %.loc28_25.1: {.d: i32} = struct_literal (%.loc28_24)
// CHECK:STDOUT:   %.loc28_25.2: ref i32 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc28_25.3: init i32 = initialize_from %.loc28_24 to %.loc28_25.2 [template = constants.%.16]
// CHECK:STDOUT:   %.loc28_25.4: init D = class_init (%.loc28_25.3), %d.var [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc28_26: init D = converted %.loc28_25.1, %.loc28_25.4 [template = constants.%struct.4]
// CHECK:STDOUT:   assign %d.var, %.loc28_26
// CHECK:STDOUT:   %AF.ref: <function> = name_ref AF, @A.%AF [template = @A.%AF]
// CHECK:STDOUT:   %AF.call: init () = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: <function> = name_ref BF, @B.%BF [template = @B.%BF]
// CHECK:STDOUT:   %BF.call: init () = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: <function> = name_ref CF, @C.%CF [template = @C.%CF]
// CHECK:STDOUT:   %CF.call: init () = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: <function> = name_ref DF, @D.%DF [template = @D.%DF]
// CHECK:STDOUT:   %DF.call: init () = call %DF.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AF();
// CHECK:STDOUT:
