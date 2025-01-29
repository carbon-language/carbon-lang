// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/adapt.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class SomeClass {
  var a: i32;
  var b: i32;
}

class SomeClassAdapter {
  adapt SomeClass;
}

class StructAdapter {
  adapt {.a: i32, .b: i32};
}

// --- fail_not_extend.carbon

library "[[@TEST_NAME]]";

class Adapted {
  fn F();
}

class AdaptNotExtend {
  adapt Adapted;
}

fn F(a: AdaptNotExtend) {
  // `Adapted` is not extended, so lookup for `F` finds nothing.
  // CHECK:STDERR: fail_not_extend.carbon:[[@LINE+4]]:3: error: member name `F` not found in `AdaptNotExtend` [MemberNameNotFoundInScope]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  a.F();
}

// --- fail_misplaced.carbon

fn F() {
  // CHECK:STDERR: fail_misplaced.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt i32;
}

interface I {
  // CHECK:STDERR: fail_misplaced.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt i32;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %SomeClass.elem: type = unbound_element_type %SomeClass, %i32 [template]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [template = constants.%StructAdapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc5_8: %SomeClass.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %SomeClass.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %SomeClass.elem = var <none>
// CHECK:STDOUT:   %.loc6_8: %SomeClass.elem = field_decl b, element1 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %SomeClass.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %SomeClass.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [template = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [template = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %int_32.loc14_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_23: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b]
// CHECK:STDOUT:   adapt_decl %struct_type.a.b [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [template = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapted: type = class_type @Adapted [template]
// CHECK:STDOUT:   %F.type.967: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.9eb: %F.type.967 = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %AdaptNotExtend: type = class_type @AdaptNotExtend [template]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Adapted = %Adapted.decl
// CHECK:STDOUT:     .AdaptNotExtend = %AdaptNotExtend.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapted.decl: type = class_decl @Adapted [template = constants.%Adapted] {} {}
// CHECK:STDOUT:   %AdaptNotExtend.decl: type = class_decl @AdaptNotExtend [template = constants.%AdaptNotExtend] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [template = constants.%F.c41] {
// CHECK:STDOUT:     %a.patt: %AdaptNotExtend = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %AdaptNotExtend = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %AdaptNotExtend = value_param runtime_param0
// CHECK:STDOUT:     %AdaptNotExtend.ref: type = name_ref AdaptNotExtend, file.%AdaptNotExtend.decl [template = constants.%AdaptNotExtend]
// CHECK:STDOUT:     %a: %AdaptNotExtend = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapted {
// CHECK:STDOUT:   %F.decl: %F.type.967 = fn_decl @F.1 [template = constants.%F.9eb] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapted
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNotExtend {
// CHECK:STDOUT:   %Adapted.ref: type = name_ref Adapted, file.%Adapted.decl [template = constants.%Adapted]
// CHECK:STDOUT:   adapt_decl %Adapted.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNotExtend
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param_patt: %AdaptNotExtend) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %AdaptNotExtend = name_ref a, %a
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_misplaced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
