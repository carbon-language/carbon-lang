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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_function_unqualified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_function_unqualified.carbon

base class Base {
  fn F();
}

class Derived {
  extend base: Base;

  fn G() { F(); }
  fn H();
}

fn Derived.H() {
  F();
}

// CHECK:STDOUT: --- base_function_unqualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.090: <witness> = impl_witness @Base.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.G.type: type = fn_type @Derived.G [concrete]
// CHECK:STDOUT:   %Derived.G: %Derived.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.H.type: type = fn_type @Derived.H [concrete]
// CHECK:STDOUT:   %Derived.H: %Derived.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e58: <witness> = impl_witness @Derived.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: %Derived.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Derived.H.decl: %Derived.H.type = fn_decl @Derived.H [concrete = constants.%Derived.H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Base.as.Destroy.impl: @Base.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.decl: %Base.as.Destroy.impl.Op.type = fn_decl @Base.as.Destroy.impl.Op [concrete = constants.%Base.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Derived.as.Destroy.impl: @Derived.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.decl: %Derived.as.Destroy.impl.Op.type = fn_decl @Derived.as.Destroy.impl.Op [concrete = constants.%Derived.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.605 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.605 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc19: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %ptr.404 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Derived.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Derived.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:   impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.090]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc20: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %Derived.G.decl: %Derived.G.type = fn_decl @Derived.G [concrete = constants.%Derived.G] {} {}
// CHECK:STDOUT:   %Derived.H.decl: %Derived.H.type = fn_decl @Derived.H [concrete = constants.%Derived.H] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   impl_decl @Derived.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.decl), @Derived.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.e58]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .G = %Derived.G.decl
// CHECK:STDOUT:   .H = %Derived.H.decl
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.as.Destroy.impl.Op(%self.param: %ptr.11f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, @Base.%Base.F.decl [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, @Base.%Base.F.decl [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.as.Destroy.impl.Op(%self.param: %ptr.404) = "no_op";
// CHECK:STDOUT:
