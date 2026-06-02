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
// CHECK:STDOUT:   %ptr.e55: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.732: type = pattern_type %ptr.e55 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.bbe: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.8f6: %T.as.DefaultOrUnformed.impl.Op.type.bbe = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.976: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.597, @T.as.DefaultOrUnformed.impl(%ptr.e55) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.80f: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%ptr.e55) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.238: %T.as.DefaultOrUnformed.impl.Op.type.80f = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %ptr.e55, (%DefaultOrUnformed.impl_witness.976) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.238, @T.as.DefaultOrUnformed.impl.Op(%ptr.e55) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f8f: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.4ac: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.b07: %ptr.as.Copy.impl.Op.type.4ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e55, (%Copy.impl_witness.f8f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7db: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.044: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7db, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.b07, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MemberNamedSelf: type = class_type @MemberNamedSelf [concrete]
// CHECK:STDOUT:   %Self.65a: type = class_type @Self [concrete]
// CHECK:STDOUT:   %pattern_type.6db: type = pattern_type %MemberNamedSelf [concrete]
// CHECK:STDOUT:   %pattern_type.bd9: type = pattern_type %Self.65a [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.1d0: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.bbe) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.8f6)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.597 = impl_witness_table (%Core.import_ref.1d0), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6db = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.6db = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.bd9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.bd9 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc28: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_32: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc28: %MemberNamedSelf = value_binding x, %x.param.loc28
// CHECK:STDOUT:     %y.param.loc28: %Self.65a = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_48: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.65a]
// CHECK:STDOUT:     %y.loc28: %Self.65a = value_binding y, %y.param.loc28
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
// CHECK:STDOUT:   %Self.decl: type = class_decl @Self [concrete = constants.%Self.65a] {} {}
// CHECK:STDOUT:   %MemberNamedSelf.F.decl: %MemberNamedSelf.F.type = fn_decl @MemberNamedSelf.F [concrete = constants.%MemberNamedSelf.F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6db = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.6db = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.bd9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.bd9 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc25: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc25_11: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc25: %MemberNamedSelf = value_binding x, %x.param.loc25
// CHECK:STDOUT:     %y.param.loc25: %Self.65a = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc25_20: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.65a]
// CHECK:STDOUT:     %y.loc25: %Self.65a = value_binding y, %y.param.loc25
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
// CHECK:STDOUT:   .Self = constants.%Self.65a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Self.patt: %pattern_type.732 = ref_binding_pattern r#Self [concrete]
// CHECK:STDOUT:     %Self.var_patt: %pattern_type.732 = var_pattern %Self.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.var: ref %ptr.e55 = var %Self.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%ptr.e55, (constants.%DefaultOrUnformed.impl_witness.976) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc17_22.1: %DefaultOrUnformed.type = converted constants.%ptr.e55, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc17_22.1 [concrete = constants.%ptr.e55]
// CHECK:STDOUT:   %.loc17_22.2: type = converted %.loc17_22.1, %as_type [concrete = constants.%ptr.e55]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.238, @T.as.DefaultOrUnformed.impl.Op(constants.%ptr.e55) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %ptr.e55 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %Self.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc17_21: type = splice_block %ptr.loc17 [concrete = constants.%ptr.e55] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %Self.ref.loc17 [concrete = constants.%ptr.e55]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self: ref %ptr.e55 = ref_binding r#Self, %Self.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.732 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.732 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.e55 = var %p.var_patt
// CHECK:STDOUT:   %Self.ref.loc18_27: ref %ptr.e55 = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc18_27: %ptr.e55 = acquire_value %Self.ref.loc18_27
// CHECK:STDOUT:   %impl.elem0: %.044 = impl_witness_access constants.%Copy.impl_witness.f8f, element0 [concrete = constants.%ptr.as.Copy.impl.Op.b07]
// CHECK:STDOUT:   %bound_method.loc18_27.1: <bound method> = bound_method %.loc18_27, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_27.2: <bound method> = bound_method %.loc18_27, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e55 = call %bound_method.loc18_27.2(%.loc18_27)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc18_23: type = splice_block %ptr.loc18 [concrete = constants.%ptr.e55] {
// CHECK:STDOUT:     %Self.ref.loc18_19: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %Self.ref.loc18_19 [concrete = constants.%ptr.e55]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.e55 = ref_binding p, %p.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %p.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%p.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc17: <bound method> = bound_method %Self.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc17: init %empty_tuple.type = call %Destroy.Op.bound.loc17(%Self.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.e55) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberNamedSelf.F(%x.param.loc28: %MemberNamedSelf, %y.param.loc28: %Self.65a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
