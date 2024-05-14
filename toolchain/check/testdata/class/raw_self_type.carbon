// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type Class [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %MemberNamedSelf: type = class_type @MemberNamedSelf [template]
// CHECK:STDOUT:   %Self: type = class_type @Self [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .MemberNamedSelf = %MemberNamedSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %MemberNamedSelf.decl: type = class_decl @MemberNamedSelf [template = constants.%MemberNamedSelf] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc20_25: type = name_ref Self, constants.%MemberNamedSelf [template = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc20_22.1: MemberNamedSelf = param x
// CHECK:STDOUT:     @F.2.%x: MemberNamedSelf = bind_name x, %x.loc20_22.1
// CHECK:STDOUT:     %Self.ref.loc20_34: type = name_ref r#Self, @MemberNamedSelf.%Self.decl [template = constants.%Self]
// CHECK:STDOUT:     %y.loc20_31.1: r#Self = param y
// CHECK:STDOUT:     @F.2.%y: r#Self = bind_name y, %y.loc20_31.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MemberNamedSelf {
// CHECK:STDOUT:   %Self.decl: type = class_decl @Self [template = constants.%Self] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc17_11: type = name_ref Self, constants.%MemberNamedSelf [template = constants.%MemberNamedSelf]
// CHECK:STDOUT:     %x.loc17_8.1: MemberNamedSelf = param x
// CHECK:STDOUT:     %x.loc17_8.2: MemberNamedSelf = bind_name x, %x.loc17_8.1
// CHECK:STDOUT:     %Self.ref.loc17_20: type = name_ref r#Self, %Self.decl [template = constants.%Self]
// CHECK:STDOUT:     %y.loc17_17.1: r#Self = param y
// CHECK:STDOUT:     %y.loc17_17.2: r#Self = bind_name y, %y.loc17_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MemberNamedSelf
// CHECK:STDOUT:   .r#Self = %Self.decl
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Self {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc9: type = ptr_type Class [template = constants.%.3]
// CHECK:STDOUT:   %Self.var: ref Class* = var r#Self
// CHECK:STDOUT:   %Self: ref Class* = bind_name r#Self, %Self.var
// CHECK:STDOUT:   %Self.ref.loc10_12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc10_16: type = ptr_type Class [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref Class* = var p
// CHECK:STDOUT:   %p: ref Class* = bind_name p, %p.var
// CHECK:STDOUT:   %Self.ref.loc10_20: ref Class* = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc10_20: Class* = bind_value %Self.ref.loc10_20
// CHECK:STDOUT:   assign %p.var, %.loc10_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%x: MemberNamedSelf, %y: r#Self) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
