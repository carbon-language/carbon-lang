// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_base.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn F[self: Self]();
  fn Unused[self: Self]();

  var x: i32;
  var unused: i32;
}

class Child {
  extend base: Base;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var a: Child = {.base = {.x = 0, .unused = 1}};
  a.x = 2;
  a.F();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Unused.type: type = fn_type @Unused [template]
// CHECK:STDOUT:   %Unused: %Unused.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.unused: type = struct_type {.x: %i32, .unused: %i32} [template]
// CHECK:STDOUT:   %complete_type.39a9: <witness> = complete_type_witness %struct_type.x.unused [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %Child.elem: type = unbound_element_type %Child, %Base [template]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.15ce: <witness> = complete_type_witness %struct_type.base [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Child = %Child.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Child.decl: type = class_decl @Child [template = constants.%Child] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unused.decl: %Unused.type = fn_decl @Unused [template = constants.%Unused] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: %Base.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc9: %Base.elem = field_decl unused, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.unused [template = constants.%complete_type.39a9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Unused = %Unused.decl
// CHECK:STDOUT:   .x = %.loc8
// CHECK:STDOUT:   .unused = %.loc9
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: %Child.elem = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [template = constants.%complete_type.15ce]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Child
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Unused[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.x.unused.53c9: type = struct_type {.x: %i32, .unused: %i32} [template]
// CHECK:STDOUT:   %complete_type.8472: <witness> = complete_type_witness %struct_type.x.unused.53c9 [template]
// CHECK:STDOUT:   %struct_type.base.b1e8: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.15ce: <witness> = complete_type_witness %struct_type.base.b1e8 [template]
// CHECK:STDOUT:   %int_0.5c61: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.x.unused.c456: type = struct_type {.x: Core.IntLiteral, .unused: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct_type.base.6c74: type = struct_type {.base: %struct_type.x.unused.c456} [template]
// CHECK:STDOUT:   %Convert.type.e4db: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b412: <bound method> = bound_method %int_0.5c61, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c350: <specific function> = specific_function %Convert.bound.b412, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.3007: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.0593: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_0.3007, %int_1.0593) [template]
// CHECK:STDOUT:   %Child.val: %Child = struct_value (%Base.val) [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.ed97: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.fb5c = import_ref Main//a, Base, unloaded
// CHECK:STDOUT:   %import_ref.d08d: type = import_ref Main//a, Child, loaded [template = constants.%Child]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.38fa: <witness> = import_ref Main//a, loc10_1, loaded [template = constants.%complete_type.8472]
// CHECK:STDOUT:   %import_ref.1f39 = import_ref Main//a, inst16 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.1e14: %F.type = import_ref Main//a, loc5_21, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.a0a3 = import_ref Main//a, loc6_26, unloaded
// CHECK:STDOUT:   %import_ref.56c1: %Base.elem = import_ref Main//a, loc8_8, loaded [template = %.09f4]
// CHECK:STDOUT:   %import_ref.7c55 = import_ref Main//a, loc9_13, unloaded
// CHECK:STDOUT:   %import_ref.c5f9: <witness> = import_ref Main//a, loc14_1, loaded [template = constants.%complete_type.15ce]
// CHECK:STDOUT:   %import_ref.9a97 = import_ref Main//a, inst70 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.0009 = import_ref Main//a, loc13_20, unloaded
// CHECK:STDOUT:   %import_ref.909ec2.2: type = import_ref Main//a, loc13_16, loaded [template = constants.%Base]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = imports.%import_ref.fb5c
// CHECK:STDOUT:     .Child = imports.%import_ref.d08d
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9a97
// CHECK:STDOUT:   .base = imports.%import_ref.0009
// CHECK:STDOUT:   extend imports.%import_ref.909ec2.2
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.c5f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.1f39
// CHECK:STDOUT:   .F = imports.%import_ref.1e14
// CHECK:STDOUT:   .Unused = imports.%import_ref.a0a3
// CHECK:STDOUT:   .x = imports.%import_ref.56c1
// CHECK:STDOUT:   .unused = imports.%import_ref.7c55
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.38fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Child = var a
// CHECK:STDOUT:   %a: ref %Child = bind_name a, %a.var
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc7_47.1: %struct_type.x.unused.c456 = struct_literal (%int_0, %int_1)
// CHECK:STDOUT:   %.loc7_48.1: %struct_type.base.6c74 = struct_literal (%.loc7_47.1)
// CHECK:STDOUT:   %impl.elem0.loc7_47.1: %Convert.type.e4db = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc7_47.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_47.1 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_47.1: <specific function> = specific_function %Convert.bound.loc7_47.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc7_47.1: init %i32 = call %Convert.specific_fn.loc7_47.1(%int_0) [template = constants.%int_0.3007]
// CHECK:STDOUT:   %.loc7_47.2: init %i32 = converted %int_0, %int.convert_checked.loc7_47.1 [template = constants.%int_0.3007]
// CHECK:STDOUT:   %.loc7_48.2: ref %Base = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_47.3: ref %i32 = class_element_access %.loc7_48.2, element0
// CHECK:STDOUT:   %.loc7_47.4: init %i32 = initialize_from %.loc7_47.2 to %.loc7_47.3 [template = constants.%int_0.3007]
// CHECK:STDOUT:   %impl.elem0.loc7_47.2: %Convert.type.e4db = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc7_47.2: <bound method> = bound_method %int_1, %impl.elem0.loc7_47.2 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_47.2: <specific function> = specific_function %Convert.bound.loc7_47.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc7_47.2: init %i32 = call %Convert.specific_fn.loc7_47.2(%int_1) [template = constants.%int_1.0593]
// CHECK:STDOUT:   %.loc7_47.5: init %i32 = converted %int_1, %int.convert_checked.loc7_47.2 [template = constants.%int_1.0593]
// CHECK:STDOUT:   %.loc7_47.6: ref %i32 = class_element_access %.loc7_48.2, element1
// CHECK:STDOUT:   %.loc7_47.7: init %i32 = initialize_from %.loc7_47.5 to %.loc7_47.6 [template = constants.%int_1.0593]
// CHECK:STDOUT:   %.loc7_47.8: init %Base = class_init (%.loc7_47.4, %.loc7_47.7), %.loc7_48.2 [template = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_48.3: init %Base = converted %.loc7_47.1, %.loc7_47.8 [template = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_48.4: init %Child = class_init (%.loc7_48.3), %a.var [template = constants.%Child.val]
// CHECK:STDOUT:   %.loc7_49: init %Child = converted %.loc7_48.1, %.loc7_48.4 [template = constants.%Child.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_49
// CHECK:STDOUT:   %a.ref.loc8: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%import_ref.56c1 [template = imports.%.09f4]
// CHECK:STDOUT:   %.loc8_4.1: ref %Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref %Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref %i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.e4db = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_2, %impl.elem0.loc8 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.specific_fn.loc8(%int_2) [template = constants.%int_2.ed97]
// CHECK:STDOUT:   %.loc8_7: init %i32 = converted %int_2, %int.convert_checked.loc8 [template = constants.%int_2.ed97]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_7
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.1e14 [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %a.ref.loc9, %F.ref
// CHECK:STDOUT:   %.loc9_3.1: ref %Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_3.2: ref %Base = converted %a.ref.loc9, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.3: %Base = bind_value %.loc9_3.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Base]() [from "a.carbon"];
// CHECK:STDOUT:
