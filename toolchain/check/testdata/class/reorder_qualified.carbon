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
      var unused a: A = {.a = 1};
      var unused b: B = {.b = 2};
      var unused c: C = {.c = 3};
      var unused d: D = {.d = 4};

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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %B.BF.type: type = fn_type @B.BF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.BF: %B.BF.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.b.5bf: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.62d: <witness> = complete_type_witness %struct_type.b.5bf [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.F.type: type = fn_type @D.F [concrete]
// CHECK:STDOUT:   %D.F: %D.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.DF.type: type = fn_type @D.DF [concrete]
// CHECK:STDOUT:   %D.DF: %D.DF.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.d.06c: type = struct_type {.d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.5f1: <witness> = complete_type_witness %struct_type.d.06c [concrete]
// CHECK:STDOUT:   %C.CF.type: type = fn_type @C.CF [concrete]
// CHECK:STDOUT:   %C.CF: %C.CF.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.c.de0: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.f13: <witness> = complete_type_witness %struct_type.c.de0 [concrete]
// CHECK:STDOUT:   %A.AF.type: type = fn_type @A.AF [concrete]
// CHECK:STDOUT:   %A.AF: %A.AF.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.15e: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.59e: <witness> = complete_type_witness %struct_type.a.15e [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.48c: %struct_type.a.a6c = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.68a) [concrete]
// CHECK:STDOUT:   %pattern_type.e78: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.e78 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.e78 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.26f: %struct_type.b.a15 = struct_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%int_2.2e0) [concrete]
// CHECK:STDOUT:   %pattern_type.ca8: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.ca8 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.ca8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.5b8: type = struct_type {.c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.d98: %struct_type.c.5b8 = struct_value (%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_3.9f3) [concrete]
// CHECK:STDOUT:   %pattern_type.a0b: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.a0b = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt: %pattern_type.a0b = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.d.3ea: type = struct_type {.d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.5a9: %struct_type.d.3ea = struct_value (%int_4.0c1) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.6bf: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_4.6bf) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc36_7.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.845: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.c2b) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc36_7.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ab9: %SubobjectDestroy.type = facet_value %D, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.316: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ab9) [concrete]
// CHECK:STDOUT:   %Destroy.facet.07e: %Destroy.type = facet_value %D, (%DestroyImpl.impl_witness.316, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.332: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.07e) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc35_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.544: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dcd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.544) [concrete]
// CHECK:STDOUT:   %Destroy.facet.54f: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.dcd, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4cb: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.54f) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b88: %SubobjectDestroy.type = facet_value %B, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.569: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b88) [concrete]
// CHECK:STDOUT:   %Destroy.facet.8e4: %Destroy.type = facet_value %B, (%DestroyImpl.impl_witness.569, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.a05: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.8e4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc33_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35e: %Destroy.type = facet_value %A, (%DestroyImpl.impl_witness.6d1, %custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.270: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.35e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:   %A.AF.decl: %A.AF.type = fn_decl @A.AF [concrete = constants.%A.AF] {} {}
// CHECK:STDOUT:   %field_decl: %A.elem = field_decl a, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.15e [concrete = constants.%complete_type.59e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %A.AF.decl
// CHECK:STDOUT:   .a = %field_decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %B.BF.decl: %B.BF.type = fn_decl @B.BF [concrete = constants.%B.BF] {} {}
// CHECK:STDOUT:   %field_decl: %B.elem = field_decl b, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.b.5bf [concrete = constants.%complete_type.62d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %B.BF.decl
// CHECK:STDOUT:   .b = %field_decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %D.DF.decl: %D.DF.type = fn_decl @D.DF [concrete = constants.%D.DF] {} {}
// CHECK:STDOUT:   %C.CF.decl: %C.CF.type = fn_decl @C.CF [concrete = constants.%C.CF] {} {}
// CHECK:STDOUT:   %field_decl: %C.elem = field_decl c, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.c.de0 [concrete = constants.%complete_type.f13]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %C.CF.decl
// CHECK:STDOUT:   .c = %field_decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %D.F.decl: %D.F.type = fn_decl @D.F [concrete = constants.%D.F] {} {}
// CHECK:STDOUT:   %D.DF.decl: %D.DF.type = fn_decl @D.DF [concrete = constants.%D.DF] {} {}
// CHECK:STDOUT:   %field_decl: %D.elem = field_decl d, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.d.06c [concrete = constants.%complete_type.5f1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %D.F.decl
// CHECK:STDOUT:   .DF = %D.DF.decl
// CHECK:STDOUT:   .d = %field_decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT:   .CF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.BF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var_storage %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc33_32.1: %struct_type.a.a6c = struct_literal (%int_1) [concrete = constants.%struct.48c]
// CHECK:STDOUT:   %impl.elem0.loc33: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc33_32.1: <bound method> = bound_method %int_1, %impl.elem0.loc33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_32.2: <bound method> = bound_method %int_1, %specific_fn.loc33 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33: init %i32 = call %bound_method.loc33_32.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc33_32.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc33_32.3: ref %i32 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc33_32.4: init %i32 to %.loc33_32.3 = in_place_init %.loc33_32.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc33_32.5: init %A to %a.var = class_init (%.loc33_32.4) [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc33_7.1: init %A = converted %.loc33_32.1, %.loc33_32.5 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc33_7.1
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var_storage %b.var_patt
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc34_32.1: %struct_type.b.a15 = struct_literal (%int_2) [concrete = constants.%struct.26f]
// CHECK:STDOUT:   %impl.elem0.loc34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc34_32.1: <bound method> = bound_method %int_2, %impl.elem0.loc34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc34: <specific function> = specific_function %impl.elem0.loc34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_32.2: <bound method> = bound_method %int_2, %specific_fn.loc34 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc34: init %i32 = call %bound_method.loc34_32.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc34_32.2: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc34 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc34_32.3: ref %i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc34_32.4: init %i32 to %.loc34_32.3 = in_place_init %.loc34_32.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc34_32.5: init %B to %b.var = class_init (%.loc34_32.4) [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc34_7.1: init %B = converted %.loc34_32.1, %.loc34_32.5 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc34_7.1
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.e78 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.e78 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc35_32.1: %struct_type.c.5b8 = struct_literal (%int_3) [concrete = constants.%struct.d98]
// CHECK:STDOUT:   %impl.elem0.loc35: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc35_32.1: <bound method> = bound_method %int_3, %impl.elem0.loc35 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_32.2: <bound method> = bound_method %int_3, %specific_fn.loc35 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35: init %i32 = call %bound_method.loc35_32.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc35_32.2: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc35_32.3: ref %i32 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc35_32.4: init %i32 to %.loc35_32.3 = in_place_init %.loc35_32.2 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc35_32.5: init %C to %c.var = class_init (%.loc35_32.4) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc35_7.1: init %C = converted %.loc35_32.1, %.loc35_32.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc35_7.1
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.ca8 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.ca8 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %D = var_storage %d.var_patt
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc36_32.1: %struct_type.d.3ea = struct_literal (%int_4) [concrete = constants.%struct.5a9]
// CHECK:STDOUT:   %impl.elem0.loc36: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc36_32.1: <bound method> = bound_method %int_4, %impl.elem0.loc36 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:   %specific_fn.loc36: <specific function> = specific_function %impl.elem0.loc36, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_32.2: <bound method> = bound_method %int_4, %specific_fn.loc36 [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36: init %i32 = call %bound_method.loc36_32.2(%int_4) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc36_32.2: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc36_32.3: ref %i32 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc36_32.4: init %i32 to %.loc36_32.3 = in_place_init %.loc36_32.2 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc36_32.5: init %D to %d.var = class_init (%.loc36_32.4) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc36_7.1: init %D = converted %.loc36_32.1, %.loc36_32.5 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign %d.var, %.loc36_7.1
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: ref %D = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.a0b = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.a0b = var_pattern %d.patt [concrete = constants.%d.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AF.ref: %A.AF.type = name_ref AF, @A.%A.AF.decl [concrete = constants.%A.AF]
// CHECK:STDOUT:   %A.AF.call: init %empty_tuple.type = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: %B.BF.type = name_ref BF, @B.%B.BF.decl [concrete = constants.%B.BF]
// CHECK:STDOUT:   %B.BF.call: init %empty_tuple.type = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: %C.CF.type = name_ref CF, @C.%C.CF.decl [concrete = constants.%C.CF]
// CHECK:STDOUT:   %C.CF.call: init %empty_tuple.type = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: %D.DF.type = name_ref DF, @D.%D.DF.decl [concrete = constants.%D.DF]
// CHECK:STDOUT:   %D.DF.call: init %empty_tuple.type = call %DF.ref()
// CHECK:STDOUT:   %Destroy.facet.loc36_7.1: %Destroy.type = facet_value constants.%D, (constants.%DestroyImpl.impl_witness.316, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.07e]
// CHECK:STDOUT:   %.loc36_7.2: %Destroy.type = converted constants.%D, %Destroy.facet.loc36_7.1 [concrete = constants.%Destroy.facet.07e]
// CHECK:STDOUT:   %Destroy.facet.loc36_7.2: %Destroy.type = facet_value constants.%D, (constants.%DestroyImpl.impl_witness.316, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.07e]
// CHECK:STDOUT:   %.loc36_7.3: %Destroy.type = converted constants.%D, %Destroy.facet.loc36_7.2 [concrete = constants.%Destroy.facet.07e]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.07e) [concrete = constants.%SelfDestruct.specific_fn.332]
// CHECK:STDOUT:   %SelfDestruct.call.loc36: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%d.var)
// CHECK:STDOUT:   %Destroy.facet.loc35_7.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.dcd, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.54f]
// CHECK:STDOUT:   %.loc35_7.2: %Destroy.type = converted constants.%C, %Destroy.facet.loc35_7.1 [concrete = constants.%Destroy.facet.54f]
// CHECK:STDOUT:   %Destroy.facet.loc35_7.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.dcd, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.54f]
// CHECK:STDOUT:   %.loc35_7.3: %Destroy.type = converted constants.%C, %Destroy.facet.loc35_7.2 [concrete = constants.%Destroy.facet.54f]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.54f) [concrete = constants.%SelfDestruct.specific_fn.4cb]
// CHECK:STDOUT:   %SelfDestruct.call.loc35: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%c.var)
// CHECK:STDOUT:   %Destroy.facet.loc34_7.1: %Destroy.type = facet_value constants.%B, (constants.%DestroyImpl.impl_witness.569, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.8e4]
// CHECK:STDOUT:   %.loc34_7.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc34_7.1 [concrete = constants.%Destroy.facet.8e4]
// CHECK:STDOUT:   %Destroy.facet.loc34_7.2: %Destroy.type = facet_value constants.%B, (constants.%DestroyImpl.impl_witness.569, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.8e4]
// CHECK:STDOUT:   %.loc34_7.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc34_7.2 [concrete = constants.%Destroy.facet.8e4]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.8e4) [concrete = constants.%SelfDestruct.specific_fn.a05]
// CHECK:STDOUT:   %SelfDestruct.call.loc34: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%b.var)
// CHECK:STDOUT:   %Destroy.facet.loc33_7.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.9) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc33_7.2: %Destroy.type = converted constants.%A, %Destroy.facet.loc33_7.1 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc33_7.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.9) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc33_7.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc33_7.2 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.35e) [concrete = constants.%SelfDestruct.specific_fn.270]
// CHECK:STDOUT:   %SelfDestruct.call.loc33: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.CF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.AF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_7.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_7.2(%self.param: ref %struct_type.d.06c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc36_7.1: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc36_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc36_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc36_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc36_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc36_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc36_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc36_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_7.3(%self.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D.elem = name_ref d, @D.%field_decl [concrete = @D.%field_decl]
// CHECK:STDOUT:   %.loc36_7.1: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc36_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc36_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc36_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc36_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc36_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc36_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc36_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc35_7.1(%self.param: ref %struct_type.c.de0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc35_7.1: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc35_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc35_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc35_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc35_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc35_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc35_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc35_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc35_7.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.elem = name_ref c, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc35_7.1: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc35_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc35_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc35_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc35_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc35_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc35_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc35_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_7.1(%self.param: ref %struct_type.b.5bf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_7.1: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc34_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc34_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc34_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc34_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc34_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc34_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc34_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_7.2(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %B.elem = name_ref b, @B.%field_decl [concrete = @B.%field_decl]
// CHECK:STDOUT:   %.loc34_7.1: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc34_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc34_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc34_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc34_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc34_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc34_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc34_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc33_7.1(%self.param: ref %struct_type.a.15e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc33_7.1: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc33_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc33_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc33_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc33_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc33_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc33_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc33_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc33_7.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.elem = name_ref a, @A.%field_decl [concrete = @A.%field_decl]
// CHECK:STDOUT:   %.loc33_7.1: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc33_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc33_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc33_7.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc33_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc33_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc33_7.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc33_7.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
