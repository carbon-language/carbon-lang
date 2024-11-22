// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/instance_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/instance_method.carbon

class C;

interface I {
  fn F[self: Self]() -> i32;
}

class C {
  extend impl as I {
    fn F[self: Self]() -> i32;
  }
}

fn F(c: C) -> i32 {
  return c.F();
}

// CHECK:STDOUT: --- instance_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: %I.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.7: type = struct_type {} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
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
// CHECK:STDOUT:     .C = %C.decl.loc11
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl.loc11: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl.loc17: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc11 [template = constants.%C]
// CHECK:STDOUT:     %.loc23_15.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc23_15.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_15.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_15.3: type = converted %int.make_type_signed, %.loc23_15.2 [template = constants.%i32]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%.loc14_14.1 (%.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%.loc14_14.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc14_14.2: type = facet_access_type %Self.ref [symbolic = %.loc14_14.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc14_14.3: type = converted %Self.ref, %.loc14_14.2 [symbolic = %.loc14_14.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc14_25.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc14_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_25.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_25.3: type = converted %int.make_type_signed, %.loc14_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: @F.1.%.loc14_14.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.1.%.loc14_14.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc14
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %.loc19_27.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc19_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_27.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_27.3: type = converted %int.make_type_signed, %.loc19_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: <witness> = complete_type_witness %.7 [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend @impl.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.loc14_14.1: type = facet_access_type %Self [symbolic = %.loc14_14.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%.loc14_14.1 (%.1)]() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C]() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%c.param_patt: %C) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %.3 = name_ref F, @I.%.loc14 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24_11.1: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %.loc24_11.2: <bound method> = bound_method %c.ref, %.loc24_11.1
// CHECK:STDOUT:   %F.call: init %i32 = call %.loc24_11.2(%c.ref)
// CHECK:STDOUT:   %.loc24_15.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc24_15.2: %i32 = converted %F.call, %.loc24_15.1
// CHECK:STDOUT:   return %.loc24_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %.loc14_14.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.5) {
// CHECK:STDOUT:   %Self => constants.%.5
// CHECK:STDOUT:   %.loc14_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
