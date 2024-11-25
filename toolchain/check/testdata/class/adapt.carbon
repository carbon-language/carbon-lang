// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapt.carbon

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
  // CHECK:STDERR: fail_not_extend.carbon:[[@LINE+3]]:3: error: name `F` not found [NameNotFound]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^~~
  a.F();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %SomeClass, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %SomeClass [template]
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
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
// CHECK:STDOUT:   %.loc5_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.3: type = converted %int.make_type_signed.loc5, %.loc5_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.3: type = converted %int.make_type_signed.loc6, %.loc6_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %SomeClass [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %.loc14_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_14: init type = call constants.%Int(%.loc14_14.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_14.2: type = value_of_initializer %int.make_type_signed.loc14_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_14.3: type = converted %int.make_type_signed.loc14_14, %.loc14_14.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_23: init type = call constants.%Int(%.loc14_23.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_23.2: type = value_of_initializer %int.make_type_signed.loc14_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_23.3: type = converted %int.make_type_signed.loc14_23, %.loc14_23.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_26: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.3]
// CHECK:STDOUT:   adapt_decl %.3
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapted: type = class_type @Adapted [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %AdaptNotExtend: type = class_type @AdaptNotExtend [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %Adapted [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
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
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %a.patt: %AdaptNotExtend = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %AdaptNotExtend = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNotExtend.ref: type = name_ref AdaptNotExtend, file.%AdaptNotExtend.decl [template = constants.%AdaptNotExtend]
// CHECK:STDOUT:     %a.param: %AdaptNotExtend = value_param runtime_param0
// CHECK:STDOUT:     %a: %AdaptNotExtend = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapted {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapted
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNotExtend {
// CHECK:STDOUT:   %Adapted.ref: type = name_ref Adapted, file.%Adapted.decl [template = constants.%Adapted]
// CHECK:STDOUT:   adapt_decl %Adapted
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %Adapted [template = constants.%.4]
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
