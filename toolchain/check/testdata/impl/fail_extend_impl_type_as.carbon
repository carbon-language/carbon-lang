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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_extend_impl_type_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_extend_impl_type_as.carbon

interface I {}

class C {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+4]]:3: error: cannot `extend` an `impl` with an explicit self type [ExtendImplSelfAs]
  // CHECK:STDERR:   extend impl i32 as I {}
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend impl i32 as I {}
}

class D {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+7]]:3: error: cannot `extend` an `impl` with an explicit self type [ExtendImplSelfAs]
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+4]]:15: note: remove the explicit `Self` type here [ExtendImplSelfAsDefault]
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  extend impl D as I;
}

class E {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+11]]:3: error: cannot `extend` an `impl` with an explicit self type [ExtendImplSelfAs]
  // CHECK:STDERR:   extend impl Self as I {}
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+8]]:15: note: remove the explicit `Self` type here [ExtendImplSelfAsDefault]
  // CHECK:STDERR:   extend impl Self as I {}
  // CHECK:STDERR:               ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE-11]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend impl Self as I {}
}

// CHECK:STDOUT: --- fail_extend_impl_type_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness.bf7: <witness> = impl_witness @C.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %I.impl_witness.fa5: <witness> = impl_witness @D.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.73d: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %I.impl_witness.7f3: <witness> = impl_witness @E.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c1f: <witness> = impl_witness @E.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.742: type = ptr_type %E [concrete]
// CHECK:STDOUT:   %pattern_type.88c: type = pattern_type %ptr.742 [concrete]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op.type: type = fn_type @E.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op: %E.as.Destroy.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: constants.%C as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.I.impl: %D.ref as %I.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: constants.%D as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc25: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @E.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.Destroy.impl: constants.%E as constants.%Destroy.type {
// CHECK:STDOUT:   %E.as.Destroy.impl.Op.decl: %E.as.Destroy.impl.Op.type = fn_decl @E.as.Destroy.impl.Op [concrete = constants.%E.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.88c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.88c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc36: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.742 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %ptr.742 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %E.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @E.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.bf7]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @D.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @D.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.fa5]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.73d]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   extend @D.as.I.impl.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @E.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.7f3]
// CHECK:STDOUT:   impl_decl @E.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@E.as.Destroy.impl.%E.as.Destroy.impl.Op.decl), @E.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.c1f]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   extend @E.as.I.impl.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.Destroy.impl.Op(%self.param: %ptr.742) = "no_op";
// CHECK:STDOUT:
