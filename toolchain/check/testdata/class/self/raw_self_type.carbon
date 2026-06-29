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
// CHECK:STDOUT:   %ptr.7ee: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.8c5: type = pattern_type %ptr.7ee [concrete]
// CHECK:STDOUT:   %Self.patt: %pattern_type.8c5 = ref_binding_pattern r#Self [concrete]
// CHECK:STDOUT:   %Self.var_patt: %pattern_type.8c5 = var_pattern %Self.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.a7b: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.374, @T.as.DefaultOrUnformed.impl(%ptr.7ee) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.4ec: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%ptr.7ee) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.aa5: %T.as.DefaultOrUnformed.impl.Op.type.4ec = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %ptr.7ee, (%DefaultOrUnformed.impl_witness.a7b) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.aa5, @T.as.DefaultOrUnformed.impl.Op(%ptr.7ee) [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.8c5 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.8c5 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fa4: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.6c2: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8fa: %ptr.as.Copy.impl.Op.type.6c2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.7ee, (%Copy.impl_witness.fa4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.8bf: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.4c3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.8bf, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.8fa, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MemberNamedSelf: type = class_type @MemberNamedSelf [concrete]
// CHECK:STDOUT:   %Self.7fc: type = class_type @Self [concrete]
// CHECK:STDOUT:   %pattern_type.bb1: type = pattern_type %MemberNamedSelf [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.bb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.bb1 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %pattern_type.917: type = pattern_type %Self.7fc [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.917 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.917 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.F.type: type = fn_type @MemberNamedSelf.F [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.F: %MemberNamedSelf.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.374 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.bb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.bb1 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.917 = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.917 = at_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc28: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_32: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc28: %MemberNamedSelf = wrapper_binding x, %x.param.loc28
// CHECK:STDOUT:     %y.param.loc28: %Self.7fc = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_48: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.7fc]
// CHECK:STDOUT:     %y.loc28: %Self.7fc = wrapper_binding y, %y.param.loc28
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
// CHECK:STDOUT:   %Self.decl: type = class_decl @Self [concrete = constants.%Self.7fc] {} {}
// CHECK:STDOUT:   %MemberNamedSelf.F.decl: %MemberNamedSelf.F.type = fn_decl @MemberNamedSelf.F [concrete = constants.%MemberNamedSelf.F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.bb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.bb1 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.917 = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.917 = at_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc25: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc25_11: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc25: %MemberNamedSelf = wrapper_binding x, %x.param.loc25
// CHECK:STDOUT:     %y.param.loc25: %Self.7fc = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc25_20: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.7fc]
// CHECK:STDOUT:     %y.loc25: %Self.7fc = wrapper_binding y, %y.param.loc25
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
// CHECK:STDOUT:   .Self = constants.%Self.7fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Self.patt: %pattern_type.8c5 = ref_binding_pattern r#Self [concrete = constants.%Self.patt]
// CHECK:STDOUT:     %Self.var_patt: %pattern_type.8c5 = var_pattern %Self.patt [concrete = constants.%Self.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.var: ref %ptr.7ee = var %Self.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%ptr.7ee, (constants.%DefaultOrUnformed.impl_witness.a7b) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc17_22.1: %DefaultOrUnformed.type = converted constants.%ptr.7ee, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc17_22.1 [concrete = constants.%ptr.7ee]
// CHECK:STDOUT:   %.loc17_22.2: type = converted %.loc17_22.1, %as_type [concrete = constants.%ptr.7ee]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.aa5, @T.as.DefaultOrUnformed.impl.Op(constants.%ptr.7ee) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %ptr.7ee = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %Self.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc17_21: type = splice_block %ptr.loc17 [concrete = constants.%ptr.7ee] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %Self.ref.loc17 [concrete = constants.%ptr.7ee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self: ref %ptr.7ee = wrapper_binding r#Self, %Self.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.8c5 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.8c5 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.7ee = var %p.var_patt
// CHECK:STDOUT:   %Self.ref.loc18_27: ref %ptr.7ee = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc18_27: %ptr.7ee = acquire_value %Self.ref.loc18_27
// CHECK:STDOUT:   %impl.elem0: %.4c3 = impl_witness_access constants.%Copy.impl_witness.fa4, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8fa]
// CHECK:STDOUT:   %bound_method.loc18_27.1: <bound method> = bound_method %.loc18_27, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_27.2: <bound method> = bound_method %.loc18_27, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.7ee = call %bound_method.loc18_27.2(%.loc18_27)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc18_23: type = splice_block %ptr.loc18 [concrete = constants.%ptr.7ee] {
// CHECK:STDOUT:     %Self.ref.loc18_19: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %Self.ref.loc18_19 [concrete = constants.%ptr.7ee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.7ee = wrapper_binding p, %p.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %p.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%p.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc17: <bound method> = bound_method %Self.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc17: init %empty_tuple.type = call %Destroy.Op.bound.loc17(%Self.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.7ee) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberNamedSelf.F(%x.param.loc28: %MemberNamedSelf, %y.param.loc28: %Self.7fc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
