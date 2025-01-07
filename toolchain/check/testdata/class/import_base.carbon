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
// CHECK:STDOUT:   %complete_type.39a: <witness> = complete_type_witness %struct_type.x.unused [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %Child.elem: type = unbound_element_type %Child, %Base [template]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.unused [template = constants.%complete_type.39a]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [template = constants.%complete_type.15c]
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
// CHECK:STDOUT:   %struct_type.x.unused.53c: type = struct_type {.x: %i32, .unused: %i32} [template]
// CHECK:STDOUT:   %complete_type.847: <witness> = complete_type_witness %struct_type.x.unused.53c [template]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.x.unused.c45: type = struct_type {.x: Core.IntLiteral, .unused: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct_type.base.6c7: type = struct_type {.base: %struct_type.x.unused.c45} [template]
// CHECK:STDOUT:   %Convert.type.e4d: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b41: <bound method> = bound_method %int_0.5c6, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c35: <specific function> = specific_function %Convert.bound.b41, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.300: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.059: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_0.300, %int_1.059) [template]
// CHECK:STDOUT:   %Child.val: %Child = struct_value (%Base.val) [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.ed9: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.fb5 = import_ref Main//a, Base, unloaded
// CHECK:STDOUT:   %import_ref.d08d: type = import_ref Main//a, Child, loaded [template = constants.%Child]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.38f: <witness> = import_ref Main//a, loc10_1, loaded [template = constants.%complete_type.847]
// CHECK:STDOUT:   %import_ref.1f3 = import_ref Main//a, inst16 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.1e1: %F.type = import_ref Main//a, loc5_21, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.a0a = import_ref Main//a, loc6_26, unloaded
// CHECK:STDOUT:   %import_ref.56c: %Base.elem = import_ref Main//a, loc8_8, loaded [template = %.09f]
// CHECK:STDOUT:   %import_ref.7c5 = import_ref Main//a, loc9_13, unloaded
// CHECK:STDOUT:   %import_ref.c5f: <witness> = import_ref Main//a, loc14_1, loaded [template = constants.%complete_type.15c]
// CHECK:STDOUT:   %import_ref.9a9 = import_ref Main//a, inst70 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.000 = import_ref Main//a, loc13_20, unloaded
// CHECK:STDOUT:   %import_ref.909ec2.2: type = import_ref Main//a, loc13_16, loaded [template = constants.%Base]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = imports.%import_ref.fb5
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
// CHECK:STDOUT:   .Self = imports.%import_ref.9a9
// CHECK:STDOUT:   .base = imports.%import_ref.000
// CHECK:STDOUT:   extend imports.%import_ref.909ec2.2
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.c5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.1f3
// CHECK:STDOUT:   .F = imports.%import_ref.1e1
// CHECK:STDOUT:   .Unused = imports.%import_ref.a0a
// CHECK:STDOUT:   .x = imports.%import_ref.56c
// CHECK:STDOUT:   .unused = imports.%import_ref.7c5
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.38f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Child = var a
// CHECK:STDOUT:   %a: ref %Child = bind_name a, %a.var
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_47.1: %struct_type.x.unused.c45 = struct_literal (%int_0, %int_1)
// CHECK:STDOUT:   %.loc7_48.1: %struct_type.base.6c7 = struct_literal (%.loc7_47.1)
// CHECK:STDOUT:   %impl.elem0.loc7_47.1: %Convert.type.e4d = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc7_47.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_47.1 [template = constants.%Convert.bound.b41]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_47.1: <specific function> = specific_function %Convert.bound.loc7_47.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c35]
// CHECK:STDOUT:   %int.convert_checked.loc7_47.1: init %i32 = call %Convert.specific_fn.loc7_47.1(%int_0) [template = constants.%int_0.300]
// CHECK:STDOUT:   %.loc7_47.2: init %i32 = converted %int_0, %int.convert_checked.loc7_47.1 [template = constants.%int_0.300]
// CHECK:STDOUT:   %.loc7_48.2: ref %Base = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_47.3: ref %i32 = class_element_access %.loc7_48.2, element0
// CHECK:STDOUT:   %.loc7_47.4: init %i32 = initialize_from %.loc7_47.2 to %.loc7_47.3 [template = constants.%int_0.300]
// CHECK:STDOUT:   %impl.elem0.loc7_47.2: %Convert.type.e4d = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc7_47.2: <bound method> = bound_method %int_1, %impl.elem0.loc7_47.2 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_47.2: <specific function> = specific_function %Convert.bound.loc7_47.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc7_47.2: init %i32 = call %Convert.specific_fn.loc7_47.2(%int_1) [template = constants.%int_1.059]
// CHECK:STDOUT:   %.loc7_47.5: init %i32 = converted %int_1, %int.convert_checked.loc7_47.2 [template = constants.%int_1.059]
// CHECK:STDOUT:   %.loc7_47.6: ref %i32 = class_element_access %.loc7_48.2, element1
// CHECK:STDOUT:   %.loc7_47.7: init %i32 = initialize_from %.loc7_47.5 to %.loc7_47.6 [template = constants.%int_1.059]
// CHECK:STDOUT:   %.loc7_47.8: init %Base = class_init (%.loc7_47.4, %.loc7_47.7), %.loc7_48.2 [template = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_48.3: init %Base = converted %.loc7_47.1, %.loc7_47.8 [template = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_48.4: init %Child = class_init (%.loc7_48.3), %a.var [template = constants.%Child.val]
// CHECK:STDOUT:   %.loc7_49: init %Child = converted %.loc7_48.1, %.loc7_48.4 [template = constants.%Child.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_49
// CHECK:STDOUT:   %a.ref.loc8: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%import_ref.56c [template = imports.%.09f]
// CHECK:STDOUT:   %.loc8_4.1: ref %Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref %Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref %i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.e4d = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_2, %impl.elem0.loc8 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.specific_fn.loc8(%int_2) [template = constants.%int_2.ed9]
// CHECK:STDOUT:   %.loc8_7: init %i32 = converted %int_2, %int.convert_checked.loc8 [template = constants.%int_2.ed9]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_7
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.1e1 [template = constants.%F]
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
