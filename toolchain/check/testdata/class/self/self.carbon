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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/self.carbon

// --- self.carbon
library "[[@TEST_NAME]]";

class Class {
  // The type of `self` may be written explicitly...
  fn F(self: Self) -> i32;
  // ...or omitted, in which case it defaults to `Self`.
  fn G(ref self) -> i32;

  var n: i32;
}

// A redeclaration must spell `self` the same way as the original.
fn Class.F(self: Self) -> i32 {
  return self.n;
}

fn Class.G(ref self) -> i32 {
  return self.n;
}

// --- fail_self_syntax_mismatch.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F(self: Self);
  fn G(self);
}

// A redeclaration must spell `self` the same way as the original.
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE+7]]:19: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Class.F(unused self) {}
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE-8]]:14: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn F(self: Self);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
fn Class.F(unused self) {}

// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE+7]]:25: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Class.G(unused self: Self) {}
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE-16]]:8: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn G(self);
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
fn Class.G(unused self: Self) {}

// --- fail_self_type_mismatch.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F(self: Self);
}

// Two explicit `self` types that differ syntactically (here `Class` instead of
// `Self`, even though they refer to the same type).
// CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+7]]:25: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Class.F(unused self: Class) {}
// CHECK:STDERR:                         ^~~~~
// CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE-8]]:14: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn F(self: Self);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
fn Class.F(unused self: Class) {}

// --- fail_return_self_value.carbon
library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+7]]:17: error: cannot implicitly convert non-type value of type `Class` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   fn F(self) -> self;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+4]]:17: note: type `Class` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   fn F(self) -> self;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  fn F(self) -> self;
}


// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.f15: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt.c01: %pattern_type.f15 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f26: %pattern_type.f15 = at_binding_pattern self, %self.param_patt.c01 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b2b: %pattern_type.f15 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.123: %pattern_type.f15 = at_binding_pattern self, %self.param_patt.b2b [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = value_param_pattern [concrete = constants.%self.param_patt.c01]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f26]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc13 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13: Core.Form = init_form %i32.loc13 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param.loc13: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc13: %Class = wrapper_binding self, %self.param.loc13
// CHECK:STDOUT:     %return.param.loc13: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc13: ref %i32 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = ref_param_pattern [concrete = constants.%self.param_patt.b2b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.123]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc17 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc17: Core.Form = init_form %i32.loc17 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param.loc17: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc17: ref %Class = wrapper_binding self, %self.param.loc17
// CHECK:STDOUT:     %return.param.loc17: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc17: ref %i32 = return_slot %return.param.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = value_param_pattern [concrete = constants.%self.param_patt.c01]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f26]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc13 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5: Core.Form = init_form %i32.loc5 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param.loc5: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc5: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc5: %Class = wrapper_binding self, %self.param.loc5
// CHECK:STDOUT:     %return.param.loc5: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc5: ref %i32 = return_slot %return.param.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = ref_param_pattern [concrete = constants.%self.param_patt.b2b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.123]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc17 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7: Core.Form = init_form %i32.loc7 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param.loc7: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc7: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc7: ref %Class = wrapper_binding self, %self.param.loc7
// CHECK:STDOUT:     %return.param.loc7: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc7: ref %i32 = return_slot %return.param.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: %Class.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.cdf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT:   .n = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param.loc13: %Class) -> out %return.param.loc13: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc13
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc9 [concrete = @Class.%.loc9]
// CHECK:STDOUT:   %.loc14_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc14_14.2: %i32 = acquire_value %.loc14_14.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc14_14.1: <bound method> = bound_method %.loc14_14.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_14.2: <bound method> = bound_method %.loc14_14.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_14.2(%.loc14_14.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param.loc17: ref %Class) -> out %return.param.loc17: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: ref %Class = name_ref self, %self.loc17
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc9 [concrete = @Class.%.loc9]
// CHECK:STDOUT:   %.loc18_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc18_14.2: %i32 = acquire_value %.loc18_14.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc18_14.1: <bound method> = bound_method %.loc18_14.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_14.2: <bound method> = bound_method %.loc18_14.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc18_14.2(%.loc18_14.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_syntax_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %Class.F.type.48cdab.1: type = fn_type @Class.F.loc4 [concrete]
// CHECK:STDOUT:   %Class.F.bd5be1.1: %Class.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.G.type.321a2c.1: type = fn_type @Class.G.loc5 [concrete]
// CHECK:STDOUT:   %Class.G.d407b2.1: %Class.G.type.321a2c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Class.F.type.48cdab.2: type = fn_type @Class.F.loc16 [concrete]
// CHECK:STDOUT:   %Class.F.bd5be1.2: %Class.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.G.type.321a2c.2: type = fn_type @Class.G.loc25 [concrete]
// CHECK:STDOUT:   %Class.G.d407b2.2: %Class.G.type.321a2c.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.48cdab.2 = fn_decl @Class.F.loc16 [concrete = constants.%Class.F.bd5be1.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.321a2c.2 = fn_decl @Class.G.loc25 [concrete = constants.%Class.G.d407b2.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.48cdab.1 = fn_decl @Class.F.loc4 [concrete = constants.%Class.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.321a2c.1 = fn_decl @Class.G.loc5 [concrete = constants.%Class.G.d407b2.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F.loc4(%self.param: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G.loc5(%self.param: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F.loc16(%self.param: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G.loc25(%self.param: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %Class.F.type.48cdab.1: type = fn_type @Class.F.loc4 [concrete]
// CHECK:STDOUT:   %Class.F.bd5be1.1: %Class.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Class.F.type.48cdab.2: type = fn_type @Class.F.loc16 [concrete]
// CHECK:STDOUT:   %Class.F.bd5be1.2: %Class.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.48cdab.2 = fn_decl @Class.F.loc16 [concrete = constants.%Class.F.bd5be1.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.48cdab.1 = fn_decl @Class.F.loc4 [concrete = constants.%Class.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F.loc4(%self.param: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F.loc16(%self.param: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_self_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.f15: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt.c01: %pattern_type.f15 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f26: %pattern_type.f15 = at_binding_pattern self, %self.param_patt.c01 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = value_param_pattern [concrete = constants.%self.param_patt.c01]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f26]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern <error>, <error> [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:     %.loc11: type = converted %self.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return: <error> = return_slot <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param: %Class) -> <error>;
// CHECK:STDOUT:
