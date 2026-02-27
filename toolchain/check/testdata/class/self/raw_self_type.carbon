// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/raw_self_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/raw_self_type.carbon

class Class {
  fn F() {
    var r#Self: Self*;
    var unused p: Self* = r#Self;
  }
}

class MemberNamedSelf {
  class r#Self {}

  fn F(x: Self, y: r#Self);
}

fn MemberNamedSelf.F(unused x: Self, unused y: r#Self) {}

// CHECK:STDOUT: --- raw_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.8e5: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.018: type = pattern_type %ptr.8e5 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.9d3: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.02e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.120: %ptr.as.Copy.impl.Op.type.02e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.8e5, (%Copy.impl_witness.9d3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.130: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.370: type = fn_type_with_self_type %Copy.WithSelf.Op.type.130, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.120, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %MemberNamedSelf: type = class_type @MemberNamedSelf [concrete]
// CHECK:STDOUT:   %Self.0d4: type = class_type @Self [concrete]
// CHECK:STDOUT:   %pattern_type.4b2: type = pattern_type %MemberNamedSelf [concrete]
// CHECK:STDOUT:   %pattern_type.f56: type = pattern_type %Self.0d4 [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.F.type: type = fn_type @MemberNamedSelf.F [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.F: %MemberNamedSelf.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .MemberNamedSelf = %MemberNamedSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %MemberNamedSelf.decl: type = class_decl @MemberNamedSelf [concrete = constants.%MemberNamedSelf] {} {}
// CHECK:STDOUT:   %MemberNamedSelf.F.decl: %MemberNamedSelf.F.type = fn_decl @MemberNamedSelf.F [concrete = constants.%MemberNamedSelf.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.4b2 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.4b2 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.f56 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.f56 = value_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc28: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_32: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc28: %MemberNamedSelf = value_binding x, %x.param.loc28
// CHECK:STDOUT:     %y.param.loc28: %Self.0d4 = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_48: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.0d4]
// CHECK:STDOUT:     %y.loc28: %Self.0d4 = value_binding y, %y.param.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MemberNamedSelf {
// CHECK:STDOUT:   %Self.decl: type = class_decl @Self [concrete = constants.%Self.0d4] {} {}
// CHECK:STDOUT:   %MemberNamedSelf.F.decl: %MemberNamedSelf.F.type = fn_decl @MemberNamedSelf.F [concrete = constants.%MemberNamedSelf.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.4b2 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.4b2 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.f56 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.f56 = value_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc25: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc25_11: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc25: %MemberNamedSelf = value_binding x, %x.param.loc25
// CHECK:STDOUT:     %y.param.loc25: %Self.0d4 = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc25_20: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.0d4]
// CHECK:STDOUT:     %y.loc25: %Self.0d4 = value_binding y, %y.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MemberNamedSelf
// CHECK:STDOUT:   .r#Self = %Self.decl
// CHECK:STDOUT:   .F = %MemberNamedSelf.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Self {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Self.0d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Self.patt: %pattern_type.018 = ref_binding_pattern r#Self [concrete]
// CHECK:STDOUT:     %Self.var_patt: %pattern_type.018 = var_pattern %Self.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.var: ref %ptr.8e5 = var %Self.var_patt
// CHECK:STDOUT:   %.loc17: type = splice_block %ptr.loc17 [concrete = constants.%ptr.8e5] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %Self.ref.loc17 [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self: ref %ptr.8e5 = ref_binding r#Self, %Self.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.018 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.018 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.8e5 = var %p.var_patt
// CHECK:STDOUT:   %Self.ref.loc18_27: ref %ptr.8e5 = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc18_27: %ptr.8e5 = acquire_value %Self.ref.loc18_27
// CHECK:STDOUT:   %impl.elem0: %.370 = impl_witness_access constants.%Copy.impl_witness.9d3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.120]
// CHECK:STDOUT:   %bound_method.loc18_27.1: <bound method> = bound_method %.loc18_27, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_27.2: <bound method> = bound_method %.loc18_27, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.8e5 = call %bound_method.loc18_27.2(%.loc18_27)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc18_23: type = splice_block %ptr.loc18 [concrete = constants.%ptr.8e5] {
// CHECK:STDOUT:     %Self.ref.loc18_19: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %Self.ref.loc18_19 [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.8e5 = ref_binding p, %p.var
// CHECK:STDOUT:   %DestroyOp.bound.loc18: <bound method> = bound_method %p.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc18: init %empty_tuple.type = call %DestroyOp.bound.loc18(%p.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc17: <bound method> = bound_method %Self.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc17: init %empty_tuple.type = call %DestroyOp.bound.loc17(%Self.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %ptr.8e5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberNamedSelf.F(%x.param.loc28: %MemberNamedSelf, %y.param.loc28: %Self.0d4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
