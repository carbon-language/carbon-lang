// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %F.type.f1b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.1f2: %F.type.f1b = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %MemberNamedSelf: type = class_type @MemberNamedSelf [concrete]
// CHECK:STDOUT:   %Self: type = class_type @Self [concrete]
// CHECK:STDOUT:   %F.type.648: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.e97: %F.type.648 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %F.decl: %F.type.648 = fn_decl @F.2 [concrete = constants.%F.e97] {
// CHECK:STDOUT:     %x.patt: %MemberNamedSelf = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %MemberNamedSelf = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %y.patt: %Self = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: %Self = value_param_pattern %y.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc24: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc24_25: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc24: %MemberNamedSelf = bind_name x, %x.param.loc24
// CHECK:STDOUT:     %y.param.loc24: %Self = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc24_34: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self]
// CHECK:STDOUT:     %y.loc24: %Self = bind_name y, %y.param.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.f1b = fn_decl @F.1 [concrete = constants.%F.1f2] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MemberNamedSelf {
// CHECK:STDOUT:   %Self.decl: type = class_decl @Self [concrete = constants.%Self] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.648 = fn_decl @F.2 [concrete = constants.%F.e97] {
// CHECK:STDOUT:     %x.patt: %MemberNamedSelf = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %MemberNamedSelf = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %y.patt: %Self = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: %Self = value_param_pattern %y.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc21: %MemberNamedSelf = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc21_11: type = name_ref Self, constants.%MemberNamedSelf [concrete = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc21: %MemberNamedSelf = bind_name x, %x.param.loc21
// CHECK:STDOUT:     %y.param.loc21: %Self = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc21_20: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [concrete = constants.%Self]
// CHECK:STDOUT:     %y.loc21: %Self = bind_name y, %y.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MemberNamedSelf
// CHECK:STDOUT:   .r#Self = %Self.decl
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Self {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Self.patt: %ptr.e71 = binding_pattern r#Self
// CHECK:STDOUT:     %.loc13_5: %ptr.e71 = var_pattern %Self.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.var: ref %ptr.e71 = var r#Self
// CHECK:STDOUT:   %.loc13_21: type = splice_block %ptr.loc13 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc13: type = ptr_type %Self.ref.loc13 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self: ref %ptr.e71 = bind_name r#Self, %Self.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %.loc14_5: %ptr.e71 = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.e71 = var p
// CHECK:STDOUT:   %Self.ref.loc14_20: ref %ptr.e71 = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc14_20: %ptr.e71 = bind_value %Self.ref.loc14_20
// CHECK:STDOUT:   assign %p.var, %.loc14_20
// CHECK:STDOUT:   %.loc14_16: type = splice_block %ptr.loc14 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:     %Self.ref.loc14_12: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc14: type = ptr_type %Self.ref.loc14_12 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.e71 = bind_name p, %p.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%x.param_patt: %MemberNamedSelf, %y.param_patt: %Self) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
