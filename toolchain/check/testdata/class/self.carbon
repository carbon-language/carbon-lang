// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self.carbon

// --- self.carbon

library "[[@TEST_NAME]]";

class Class {
  fn F[self: Self]() -> i32;
  fn G[addr self: Self*]() -> i32;

  var n: i32;
}

fn Class.F[self: Self]() -> i32 {
  return self.n;
}

fn Class.G[addr self: Self*]() -> i32 {
  return (*self).n;
}

// --- fail_return_self_value.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+6]]:25: error: cannot implicitly convert from `Class` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self]() -> self;
  // CHECK:STDERR:                         ^~~~
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+3]]:25: note: type `Class` does not implement interface `ImplicitAs` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self]() -> self;
  // CHECK:STDERR:                         ^~~~
  fn F[self: Self]() -> self;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc11: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_29.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_29.2: type = converted %int.make_type_32.loc11, %.loc11_29.1 [template = i32]
// CHECK:STDOUT:     %self.param.loc11: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc11: %Class = bind_name self, %self.param.loc11
// CHECK:STDOUT:     %return.param.loc11: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc15_12: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc15: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc15_27: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_35.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_35.2: type = converted %int.make_type_32.loc15, %.loc15_35.1 [template = i32]
// CHECK:STDOUT:     %self.param.loc15: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc15: %.1 = bind_name self, %self.param.loc15
// CHECK:STDOUT:     %return.param.loc15: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_25.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:     %.loc5_25.2: type = converted %int.make_type_32.loc5, %.loc5_25.1 [template = i32]
// CHECK:STDOUT:     %self.param.loc5: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc5: %Class = bind_name self, %self.param.loc5
// CHECK:STDOUT:     %return.param.loc5: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %.loc5_22: ref i32 = return_slot %return.param.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc15_12: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc6_23: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_31.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:     %.loc6_31.2: type = converted %int.make_type_32.loc6, %.loc6_31.1 [template = i32]
// CHECK:STDOUT:     %self.param.loc6: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc6: %.1 = bind_name self, %self.param.loc6
// CHECK:STDOUT:     %return.param.loc6: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %.loc6_28: ref i32 = return_slot %return.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_32, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:   %.loc8_8: %.2 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc8_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc11
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, @Class.%.loc8_8 [template = @Class.%.loc8_8]
// CHECK:STDOUT:   %.loc12_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc12_14.2: i32 = bind_value %.loc12_14.1
// CHECK:STDOUT:   return %.loc12_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self.param_patt: %.1]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.1 = name_ref self, %self.loc15
// CHECK:STDOUT:   %.loc16_11: ref %Class = deref %self.ref
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, @Class.%.loc8_8 [template = @Class.%.loc8_8]
// CHECK:STDOUT:   %.loc16_17.1: ref i32 = class_element_access %.loc16_11, element0
// CHECK:STDOUT:   %.loc16_17.2: i32 = bind_value %.loc16_17.1
// CHECK:STDOUT:   return %.loc16_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_self_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(type)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.5: %.1 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+48, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+54, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.1 (%.1) = import_ref Core//prelude/operators/as, inst+76, loaded [symbolic = @ImplicitAs.%.2 (constants.%.5)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.1) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:     %.loc11: type = converted %self.ref, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Class]() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:   %Dest.patt => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT:   %.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
