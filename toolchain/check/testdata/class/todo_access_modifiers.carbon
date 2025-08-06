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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/todo_access_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/todo_access_modifiers.carbon

// TODO: Test calls to these (member access control is not yet implemented).
class Access {
  private fn F();

  protected fn G();

  private var k: i32;

  protected var l: i32;
}

// CHECK:STDOUT: --- todo_access_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Access: type = class_type @Access [concrete]
// CHECK:STDOUT:   %Access.F.type: type = fn_type @Access.F [concrete]
// CHECK:STDOUT:   %Access.F: %Access.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Access.G.type: type = fn_type @Access.G [concrete]
// CHECK:STDOUT:   %Access.G: %Access.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Access.elem: type = unbound_element_type %Access, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Access.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.2af: type = ptr_type %Access [concrete]
// CHECK:STDOUT:   %pattern_type.010: type = pattern_type %ptr.2af [concrete]
// CHECK:STDOUT:   %Access.as.Destroy.impl.Op.type: type = fn_type @Access.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Access.as.Destroy.impl.Op: %Access.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.k.l: type = struct_type {.k: %i32, .l: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.48a: <witness> = complete_type_witness %struct_type.k.l [concrete]
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
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Access.decl: type = class_decl @Access [concrete = constants.%Access] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Access.as.Destroy.impl: constants.%Access as constants.%Destroy.type {
// CHECK:STDOUT:   %Access.as.Destroy.impl.Op.decl: %Access.as.Destroy.impl.Op.type = fn_decl @Access.as.Destroy.impl.Op [concrete = constants.%Access.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.010 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.010 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.2af = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Access [concrete = constants.%Access]
// CHECK:STDOUT:     %self: %ptr.2af = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Access.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Access.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Access {
// CHECK:STDOUT:   %Access.F.decl: %Access.F.type = fn_decl @Access.F [concrete = constants.%Access.F] {} {}
// CHECK:STDOUT:   %Access.G.decl: %Access.G.type = fn_decl @Access.G [concrete = constants.%Access.G] {} {}
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21: %Access.elem = field_decl k, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc23: %Access.elem = field_decl l, element1 [concrete]
// CHECK:STDOUT:   impl_decl @Access.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Access.as.Destroy.impl.%Access.as.Destroy.impl.Op.decl), @Access.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.k.l [concrete = constants.%complete_type.48a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Access
// CHECK:STDOUT:   .F [private] = %Access.F.decl
// CHECK:STDOUT:   .G [protected] = %Access.G.decl
// CHECK:STDOUT:   .k [private] = %.loc21
// CHECK:STDOUT:   .l [protected] = %.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access.F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access.G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access.as.Destroy.impl.Op(%self.param: %ptr.2af) = "no_op";
// CHECK:STDOUT:
