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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/raw_self_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/raw_self_type.carbon

class Class {
  fn F() {
    var r#Self: Self*;
    var p: Self* = r#Self;
  }
}

class MemberNamedSelf {
  class r#Self {}

  fn F(x: Self, y: r#Self);
}

fn MemberNamedSelf.F(x: Self, y: r#Self) {}

// CHECK:STDOUT: --- raw_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.73b: <witness> = impl_witness @Class.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.17c: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%ptr.e71) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d98: %T.as.Destroy.impl.Op.type.17c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.0dd: type = ptr_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.d98, @T.as.Destroy.impl.Op(%ptr.e71) [concrete]
// CHECK:STDOUT:   %MemberNamedSelf: type = class_type @MemberNamedSelf [concrete]
// CHECK:STDOUT:   %Self.362: type = class_type @Self [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d63: <witness> = impl_witness @Self.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.fbf: type = ptr_type %Self.362 [concrete]
// CHECK:STDOUT:   %pattern_type.364: type = pattern_type %ptr.fbf [concrete]
// CHECK:STDOUT:   %Self.as.Destroy.impl.Op.type: type = fn_type @Self.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Self.as.Destroy.impl.Op: %Self.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.356: type = pattern_type %MemberNamedSelf [concrete]
// CHECK:STDOUT:   %pattern_type.c06: type = pattern_type %Self.362 [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.F.type: type = fn_type @MemberNamedSelf.F [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.F: %MemberNamedSelf.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.db0: <witness> = impl_witness @MemberNamedSelf.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.9b1: type = ptr_type %MemberNamedSelf [concrete]
// CHECK:STDOUT:   %pattern_type.64a: type = pattern_type %ptr.9b1 [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.as.Destroy.impl.Op.type: type = fn_type @MemberNamedSelf.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %MemberNamedSelf.as.Destroy.impl.Op: %MemberNamedSelf.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %x.patt: %pattern_type.356 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.356 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.c06 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.c06 = value_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc28: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_25: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc28: %MemberNamedSelf = bind_name x, %x.param.loc28
// CHECK:STDOUT:     %y.param.loc28: %Self.362 = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_34: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.362]
// CHECK:STDOUT:     %y.loc28: %Self.362 = bind_name y, %y.param.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Class.as.Destroy.impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.decl: %Class.as.Destroy.impl.Op.type = fn_decl @Class.as.Destroy.impl.Op [concrete = constants.%Class.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Self.as.Destroy.impl: @Self.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Self.as.Destroy.impl.Op.decl: %Self.as.Destroy.impl.Op.type = fn_decl @Self.as.Destroy.impl.Op [concrete = constants.%Self.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.364 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.364 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc23: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.fbf = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Self.362 [concrete = constants.%Self.362]
// CHECK:STDOUT:     %self: %ptr.fbf = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Self.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Self.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @MemberNamedSelf.as.Destroy.impl: @MemberNamedSelf.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %MemberNamedSelf.as.Destroy.impl.Op.decl: %MemberNamedSelf.as.Destroy.impl.Op.type = fn_decl @MemberNamedSelf.as.Destroy.impl.Op [concrete = constants.%MemberNamedSelf.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.64a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.64a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc22: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.9b1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %self: %ptr.9b1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %MemberNamedSelf.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @MemberNamedSelf.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.73b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MemberNamedSelf {
// CHECK:STDOUT:   %Self.decl: type = class_decl @Self [concrete = constants.%Self.362] {} {}
// CHECK:STDOUT:   %MemberNamedSelf.F.decl: %MemberNamedSelf.F.type = fn_decl @MemberNamedSelf.F [concrete = constants.%MemberNamedSelf.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.356 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.356 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.c06 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.c06 = value_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc25: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc25_11: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc25: %MemberNamedSelf = bind_name x, %x.param.loc25
// CHECK:STDOUT:     %y.param.loc25: %Self.362 = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc25_20: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self.362]
// CHECK:STDOUT:     %y.loc25: %Self.362 = bind_name y, %y.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:   impl_decl @MemberNamedSelf.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@MemberNamedSelf.as.Destroy.impl.%MemberNamedSelf.as.Destroy.impl.Op.decl), @MemberNamedSelf.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.db0]
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Self.362 [concrete = constants.%Self.362]
// CHECK:STDOUT:   impl_decl @Self.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Self.as.Destroy.impl.%Self.as.Destroy.impl.Op.decl), @Self.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.d63]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Self.362
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Self.patt: %pattern_type.796 = binding_pattern r#Self [concrete]
// CHECK:STDOUT:     %Self.var_patt: %pattern_type.796 = var_pattern %Self.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.var: ref %ptr.e71 = var %Self.var_patt
// CHECK:STDOUT:   %.loc17: type = splice_block %ptr.loc17 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %Self.ref.loc17 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self: ref %ptr.e71 = bind_name r#Self, %Self.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.796 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.796 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.e71 = var %p.var_patt
// CHECK:STDOUT:   %Self.ref.loc18_20: ref %ptr.e71 = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc18_20: %ptr.e71 = bind_value %Self.ref.loc18_20
// CHECK:STDOUT:   assign %p.var, %.loc18_20
// CHECK:STDOUT:   %.loc18_16: type = splice_block %ptr.loc18 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:     %Self.ref.loc18_12: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %Self.ref.loc18_12 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.e71 = bind_name p, %p.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18: <bound method> = bound_method %p.var, constants.%T.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.d98, @T.as.Destroy.impl.Op(constants.%ptr.e71) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %p.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc18: %ptr.0dd = addr_of %p.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18: init %empty_tuple.type = call %bound_method.loc18(%addr.loc18)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc17: <bound method> = bound_method %Self.var, constants.%T.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.d98, @T.as.Destroy.impl.Op(constants.%ptr.e71) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17: <bound method> = bound_method %Self.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc17: %ptr.0dd = addr_of %Self.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc17: init %empty_tuple.type = call %bound_method.loc17(%addr.loc17)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.as.Destroy.impl.Op(%self.param: %ptr.e71) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Self.as.Destroy.impl.Op(%self.param: %ptr.fbf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberNamedSelf.F(%x.param.loc28: %MemberNamedSelf, %y.param.loc28: %Self.362) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberNamedSelf.as.Destroy.impl.Op(%self.param: %ptr.9b1) = "no_op";
// CHECK:STDOUT:
