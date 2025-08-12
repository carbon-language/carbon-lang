// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
  // CHECK:STDERR: fail_not_extend.carbon:[[@LINE+4]]:3: error: member name `F` not found in `AdaptNotExtend` [MemberNameNotFoundInInstScope]
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
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %SomeClass.elem: type = unbound_element_type %SomeClass, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ab1: <witness> = impl_witness @SomeClass.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.c8a: type = ptr_type %SomeClass [concrete]
// CHECK:STDOUT:   %pattern_type.a47: type = pattern_type %ptr.c8a [concrete]
// CHECK:STDOUT:   %SomeClass.as.Destroy.impl.Op.type: type = fn_type @SomeClass.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %SomeClass.as.Destroy.impl.Op: %SomeClass.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.84c: <witness> = impl_witness @SomeClassAdapter.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.b51: type = ptr_type %SomeClassAdapter [concrete]
// CHECK:STDOUT:   %pattern_type.76d: type = pattern_type %ptr.b51 [concrete]
// CHECK:STDOUT:   %SomeClassAdapter.as.Destroy.impl.Op.type: type = fn_type @SomeClassAdapter.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %SomeClassAdapter.as.Destroy.impl.Op: %SomeClassAdapter.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.6e0: <witness> = impl_witness @StructAdapter.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.670: type = ptr_type %StructAdapter [concrete]
// CHECK:STDOUT:   %pattern_type.671: type = pattern_type %ptr.670 [concrete]
// CHECK:STDOUT:   %StructAdapter.as.Destroy.impl.Op.type: type = fn_type @StructAdapter.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %StructAdapter.as.Destroy.impl.Op: %StructAdapter.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [concrete = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [concrete = constants.%StructAdapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SomeClass.as.Destroy.impl: @SomeClass.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %SomeClass.as.Destroy.impl.Op.decl: %SomeClass.as.Destroy.impl.Op.type = fn_decl @SomeClass.as.Destroy.impl.Op [concrete = constants.%SomeClass.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a47 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a47 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.c8a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SomeClass [concrete = constants.%SomeClass]
// CHECK:STDOUT:     %self: %ptr.c8a = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %SomeClass.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @SomeClass.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SomeClassAdapter.as.Destroy.impl: @SomeClassAdapter.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %SomeClassAdapter.as.Destroy.impl.Op.decl: %SomeClassAdapter.as.Destroy.impl.Op.type = fn_decl @SomeClassAdapter.as.Destroy.impl.Op [concrete = constants.%SomeClassAdapter.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.76d = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.76d = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.b51 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SomeClassAdapter [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %self: %ptr.b51 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %SomeClassAdapter.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @SomeClassAdapter.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @StructAdapter.as.Destroy.impl: @StructAdapter.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %StructAdapter.as.Destroy.impl.Op.decl: %StructAdapter.as.Destroy.impl.Op.type = fn_decl @StructAdapter.as.Destroy.impl.Op [concrete = constants.%StructAdapter.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.671 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.671 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc13: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.670 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%StructAdapter [concrete = constants.%StructAdapter]
// CHECK:STDOUT:     %self: %ptr.670 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %StructAdapter.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @StructAdapter.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %SomeClass.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %SomeClass.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%SomeClass [concrete = constants.%SomeClass]
// CHECK:STDOUT:   impl_decl @SomeClass.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@SomeClass.as.Destroy.impl.%SomeClass.as.Destroy.impl.Op.decl), @SomeClass.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ab1]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [concrete = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%SomeClassAdapter [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:   impl_decl @SomeClassAdapter.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@SomeClassAdapter.as.Destroy.impl.%SomeClassAdapter.as.Destroy.impl.Op.decl), @SomeClassAdapter.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.84c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   .SomeClass = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %int_32.loc14_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   adapt_decl %struct_type.a.b [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%StructAdapter [concrete = constants.%StructAdapter]
// CHECK:STDOUT:   impl_decl @StructAdapter.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@StructAdapter.as.Destroy.impl.%StructAdapter.as.Destroy.impl.Op.decl), @StructAdapter.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.6e0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeClass.as.Destroy.impl.Op(%self.param: %ptr.c8a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeClassAdapter.as.Destroy.impl.Op(%self.param: %ptr.b51) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @StructAdapter.as.Destroy.impl.Op(%self.param: %ptr.670) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapted: type = class_type @Adapted [concrete]
// CHECK:STDOUT:   %Adapted.F.type: type = fn_type @Adapted.F [concrete]
// CHECK:STDOUT:   %Adapted.F: %Adapted.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d23: <witness> = impl_witness @Adapted.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.08d: type = ptr_type %Adapted [concrete]
// CHECK:STDOUT:   %pattern_type.c04: type = pattern_type %ptr.08d [concrete]
// CHECK:STDOUT:   %Adapted.as.Destroy.impl.Op.type: type = fn_type @Adapted.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Adapted.as.Destroy.impl.Op: %Adapted.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNotExtend: type = class_type @AdaptNotExtend [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.f9b: <witness> = impl_witness @AdaptNotExtend.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.ae9: type = ptr_type %AdaptNotExtend [concrete]
// CHECK:STDOUT:   %pattern_type.73e: type = pattern_type %ptr.ae9 [concrete]
// CHECK:STDOUT:   %AdaptNotExtend.as.Destroy.impl.Op.type: type = fn_type @AdaptNotExtend.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptNotExtend.as.Destroy.impl.Op: %AdaptNotExtend.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.893: type = pattern_type %AdaptNotExtend [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Adapted = %Adapted.decl
// CHECK:STDOUT:     .AdaptNotExtend = %AdaptNotExtend.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapted.decl: type = class_decl @Adapted [concrete = constants.%Adapted] {} {}
// CHECK:STDOUT:   %AdaptNotExtend.decl: type = class_decl @AdaptNotExtend [concrete = constants.%AdaptNotExtend] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.893 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.893 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %AdaptNotExtend = value_param call_param0
// CHECK:STDOUT:     %AdaptNotExtend.ref: type = name_ref AdaptNotExtend, file.%AdaptNotExtend.decl [concrete = constants.%AdaptNotExtend]
// CHECK:STDOUT:     %a: %AdaptNotExtend = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Adapted.as.Destroy.impl: @Adapted.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Adapted.as.Destroy.impl.Op.decl: %Adapted.as.Destroy.impl.Op.type = fn_decl @Adapted.as.Destroy.impl.Op [concrete = constants.%Adapted.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c04 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c04 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.08d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Adapted [concrete = constants.%Adapted]
// CHECK:STDOUT:     %self: %ptr.08d = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Adapted.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Adapted.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AdaptNotExtend.as.Destroy.impl: @AdaptNotExtend.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptNotExtend.as.Destroy.impl.Op.decl: %AdaptNotExtend.as.Destroy.impl.Op.type = fn_decl @AdaptNotExtend.as.Destroy.impl.Op [concrete = constants.%AdaptNotExtend.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.73e = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.73e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.ae9 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptNotExtend [concrete = constants.%AdaptNotExtend]
// CHECK:STDOUT:     %self: %ptr.ae9 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptNotExtend.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptNotExtend.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapted {
// CHECK:STDOUT:   %Adapted.F.decl: %Adapted.F.type = fn_decl @Adapted.F [concrete = constants.%Adapted.F] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Adapted [concrete = constants.%Adapted]
// CHECK:STDOUT:   impl_decl @Adapted.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Adapted.as.Destroy.impl.%Adapted.as.Destroy.impl.Op.decl), @Adapted.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.d23]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapted
// CHECK:STDOUT:   .F = %Adapted.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNotExtend {
// CHECK:STDOUT:   %Adapted.ref: type = name_ref Adapted, file.%Adapted.decl [concrete = constants.%Adapted]
// CHECK:STDOUT:   adapt_decl %Adapted.ref [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptNotExtend [concrete = constants.%AdaptNotExtend]
// CHECK:STDOUT:   impl_decl @AdaptNotExtend.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptNotExtend.as.Destroy.impl.%AdaptNotExtend.as.Destroy.impl.Op.decl), @AdaptNotExtend.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.f9b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNotExtend
// CHECK:STDOUT:   .Adapted = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Adapted.F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Adapted.as.Destroy.impl.Op(%self.param: %ptr.08d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptNotExtend.as.Destroy.impl.Op(%self.param: %ptr.ae9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %AdaptNotExtend) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %AdaptNotExtend = name_ref a, %a
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_misplaced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
