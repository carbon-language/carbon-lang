// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %BF.type: type = fn_type @BF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BF: %BF.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.b.0a3: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ba8: <witness> = complete_type_witness %struct_type.b.0a3 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %DF.type: type = fn_type @DF [concrete]
// CHECK:STDOUT:   %DF: %DF.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.d.b7b: type = struct_type {.d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.860: <witness> = complete_type_witness %struct_type.d.b7b [concrete]
// CHECK:STDOUT:   %CF.type: type = fn_type @CF [concrete]
// CHECK:STDOUT:   %CF: %CF.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.c.b66: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.836: <witness> = complete_type_witness %struct_type.c.b66 [concrete]
// CHECK:STDOUT:   %AF.type: type = fn_type @AF [concrete]
// CHECK:STDOUT:   %AF: %AF.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.ba9: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a.ba9 [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %pattern_type.98e: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %pattern_type.6d8: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.5b8: type = struct_type {.c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_3.822) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.d.3ea: type = struct_type {.d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_4.940) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.3, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.102: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%D) [concrete]
// CHECK:STDOUT:   %Op.type.7e3: type = fn_type @Op.3, @impl.49c(%D) [concrete]
// CHECK:STDOUT:   %Op.47c: %Op.type.7e3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.321: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %Destroy.facet.18e: %Destroy.type = facet_value %D, (%Destroy.impl_witness.102) [concrete]
// CHECK:STDOUT:   %.8ce: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.18e [concrete]
// CHECK:STDOUT:   %Op.specific_fn.ceb: <specific function> = specific_function %Op.47c, @Op.3(%D) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c38: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.type.2eb: type = fn_type @Op.3, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.f77: %Op.type.2eb = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.388: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Destroy.facet.8bd: %Destroy.type = facet_value %C, (%Destroy.impl_witness.c38) [concrete]
// CHECK:STDOUT:   %.e53: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.8bd [concrete]
// CHECK:STDOUT:   %Op.specific_fn.c65: <specific function> = specific_function %Op.f77, @Op.3(%C) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.7c6: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%B) [concrete]
// CHECK:STDOUT:   %Op.type.266: type = fn_type @Op.3, @impl.49c(%B) [concrete]
// CHECK:STDOUT:   %Op.4a4: %Op.type.266 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.01b: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %Destroy.facet.422: %Destroy.type = facet_value %B, (%Destroy.impl_witness.7c6) [concrete]
// CHECK:STDOUT:   %.24f: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.422 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.70c: <specific function> = specific_function %Op.4a4, @Op.3(%B) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b6e: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%A) [concrete]
// CHECK:STDOUT:   %Op.type.b96: type = fn_type @Op.3, @impl.49c(%A) [concrete]
// CHECK:STDOUT:   %Op.885: %Op.type.b96 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %Destroy.facet.4a1: %Destroy.type = facet_value %A, (%Destroy.impl_witness.b6e) [concrete]
// CHECK:STDOUT:   %.af1: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.4a1 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.716: <specific function> = specific_function %Op.885, @Op.3(%A) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %AF.decl: %AF.type = fn_decl @AF [concrete = constants.%AF] {} {}
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc50: %A.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.ba9]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF.decl
// CHECK:STDOUT:   .a = %.loc50
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %BF.decl: %BF.type = fn_decl @BF [concrete = constants.%BF] {} {}
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc20: %B.elem = field_decl b, element0 [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %i32} [concrete = constants.%struct_type.b.0a3]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.b [concrete = constants.%complete_type.ba8]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %BF.decl
// CHECK:STDOUT:   .b = %.loc20
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [concrete = constants.%DF] {} {}
// CHECK:STDOUT:   %CF.decl: %CF.type = fn_decl @CF [concrete = constants.%CF] {} {}
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc46: %C.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c.b66]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete = constants.%complete_type.836]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF.decl
// CHECK:STDOUT:   .c = %.loc46
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [concrete = constants.%DF] {} {}
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc28: %D.elem = field_decl d, element0 [concrete]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %i32} [concrete = constants.%struct_type.d.b7b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.d [concrete = constants.%complete_type.860]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .DF = %DF.decl
// CHECK:STDOUT:   .d = %.loc28
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT:   .CF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c10 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc33_25.1: %struct_type.a.a6c = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0.loc33_25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc33_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc33_25 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc33_25: <specific function> = specific_function %impl.elem0.loc33_25, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_25.2: <bound method> = bound_method %int_1, %specific_fn.loc33_25 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc33: init %i32 = call %bound_method.loc33_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_25.2: init %i32 = converted %int_1, %int.convert_checked.loc33 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_25.3: ref %i32 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc33_25.4: init %i32 = initialize_from %.loc33_25.2 to %.loc33_25.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_25.5: init %A = class_init (%.loc33_25.4), %a.var [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc33_7: init %A = converted %.loc33_25.1, %.loc33_25.5 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc33_7
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.98e = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.98e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc34_25.1: %struct_type.b.a15 = struct_literal (%int_2)
// CHECK:STDOUT:   %impl.elem0.loc34_25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc34_25.1: <bound method> = bound_method %int_2, %impl.elem0.loc34_25 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc34_25: <specific function> = specific_function %impl.elem0.loc34_25, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_25.2: <bound method> = bound_method %int_2, %specific_fn.loc34_25 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc34: init %i32 = call %bound_method.loc34_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc34_25.2: init %i32 = converted %int_2, %int.convert_checked.loc34 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc34_25.3: ref %i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc34_25.4: init %i32 = initialize_from %.loc34_25.2 to %.loc34_25.3 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc34_25.5: init %B = class_init (%.loc34_25.4), %b.var [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc34_7: init %B = converted %.loc34_25.1, %.loc34_25.5 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc34_7
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6d8 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.6d8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc35_25.1: %struct_type.c.5b8 = struct_literal (%int_3)
// CHECK:STDOUT:   %impl.elem0.loc35_25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc35_25.1: <bound method> = bound_method %int_3, %impl.elem0.loc35_25 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc35_25: <specific function> = specific_function %impl.elem0.loc35_25, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_25.2: <bound method> = bound_method %int_3, %specific_fn.loc35_25 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc35: init %i32 = call %bound_method.loc35_25.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc35_25.2: init %i32 = converted %int_3, %int.convert_checked.loc35 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc35_25.3: ref %i32 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc35_25.4: init %i32 = initialize_from %.loc35_25.2 to %.loc35_25.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc35_25.5: init %C = class_init (%.loc35_25.4), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc35_7: init %C = converted %.loc35_25.1, %.loc35_25.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc35_7
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.d58 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.d58 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %D = var %d.var_patt
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc36_25.1: %struct_type.d.3ea = struct_literal (%int_4)
// CHECK:STDOUT:   %impl.elem0.loc36_25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc36_25.1: <bound method> = bound_method %int_4, %impl.elem0.loc36_25 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc36_25: <specific function> = specific_function %impl.elem0.loc36_25, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_25.2: <bound method> = bound_method %int_4, %specific_fn.loc36_25 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc36: init %i32 = call %bound_method.loc36_25.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc36_25.2: init %i32 = converted %int_4, %int.convert_checked.loc36 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc36_25.3: ref %i32 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc36_25.4: init %i32 = initialize_from %.loc36_25.2 to %.loc36_25.3 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc36_25.5: init %D = class_init (%.loc36_25.4), %d.var [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc36_7: init %D = converted %.loc36_25.1, %.loc36_25.5 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign %d.var, %.loc36_7
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   %AF.ref: %AF.type = name_ref AF, @A.%AF.decl [concrete = constants.%AF]
// CHECK:STDOUT:   %AF.call: init %empty_tuple.type = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: %BF.type = name_ref BF, @B.%BF.decl [concrete = constants.%BF]
// CHECK:STDOUT:   %BF.call: init %empty_tuple.type = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: %CF.type = name_ref CF, @C.%CF.decl [concrete = constants.%CF]
// CHECK:STDOUT:   %CF.call: init %empty_tuple.type = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: %DF.type = name_ref DF, @D.%DF.decl [concrete = constants.%DF]
// CHECK:STDOUT:   %DF.call: init %empty_tuple.type = call %DF.ref()
// CHECK:STDOUT:   %impl.elem0.loc36_7: %.8ce = impl_witness_access constants.%Destroy.impl_witness.102, element0 [concrete = constants.%Op.47c]
// CHECK:STDOUT:   %bound_method.loc36_7.1: <bound method> = bound_method %d.var, %impl.elem0.loc36_7
// CHECK:STDOUT:   %specific_fn.loc36_7: <specific function> = specific_function %impl.elem0.loc36_7, @Op.3(constants.%D) [concrete = constants.%Op.specific_fn.ceb]
// CHECK:STDOUT:   %bound_method.loc36_7.2: <bound method> = bound_method %d.var, %specific_fn.loc36_7
// CHECK:STDOUT:   %addr.loc36: %ptr.321 = addr_of %d.var
// CHECK:STDOUT:   %no_op.loc36: init %empty_tuple.type = call %bound_method.loc36_7.2(%addr.loc36)
// CHECK:STDOUT:   %impl.elem0.loc35_7: %.e53 = impl_witness_access constants.%Destroy.impl_witness.c38, element0 [concrete = constants.%Op.f77]
// CHECK:STDOUT:   %bound_method.loc35_7.1: <bound method> = bound_method %c.var, %impl.elem0.loc35_7
// CHECK:STDOUT:   %specific_fn.loc35_7: <specific function> = specific_function %impl.elem0.loc35_7, @Op.3(constants.%C) [concrete = constants.%Op.specific_fn.c65]
// CHECK:STDOUT:   %bound_method.loc35_7.2: <bound method> = bound_method %c.var, %specific_fn.loc35_7
// CHECK:STDOUT:   %addr.loc35: %ptr.388 = addr_of %c.var
// CHECK:STDOUT:   %no_op.loc35: init %empty_tuple.type = call %bound_method.loc35_7.2(%addr.loc35)
// CHECK:STDOUT:   %impl.elem0.loc34_7: %.24f = impl_witness_access constants.%Destroy.impl_witness.7c6, element0 [concrete = constants.%Op.4a4]
// CHECK:STDOUT:   %bound_method.loc34_7.1: <bound method> = bound_method %b.var, %impl.elem0.loc34_7
// CHECK:STDOUT:   %specific_fn.loc34_7: <specific function> = specific_function %impl.elem0.loc34_7, @Op.3(constants.%B) [concrete = constants.%Op.specific_fn.70c]
// CHECK:STDOUT:   %bound_method.loc34_7.2: <bound method> = bound_method %b.var, %specific_fn.loc34_7
// CHECK:STDOUT:   %addr.loc34: %ptr.01b = addr_of %b.var
// CHECK:STDOUT:   %no_op.loc34: init %empty_tuple.type = call %bound_method.loc34_7.2(%addr.loc34)
// CHECK:STDOUT:   %impl.elem0.loc33_7: %.af1 = impl_witness_access constants.%Destroy.impl_witness.b6e, element0 [concrete = constants.%Op.885]
// CHECK:STDOUT:   %bound_method.loc33_7.1: <bound method> = bound_method %a.var, %impl.elem0.loc33_7
// CHECK:STDOUT:   %specific_fn.loc33_7: <specific function> = specific_function %impl.elem0.loc33_7, @Op.3(constants.%A) [concrete = constants.%Op.specific_fn.716]
// CHECK:STDOUT:   %bound_method.loc33_7.2: <bound method> = bound_method %a.var, %specific_fn.loc33_7
// CHECK:STDOUT:   %addr.loc33: %ptr.6db = addr_of %a.var
// CHECK:STDOUT:   %no_op.loc33: init %empty_tuple.type = call %bound_method.loc33_7.2(%addr.loc33)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AF();
// CHECK:STDOUT:
