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

// A redeclaration must spell `self` the same way as the original; mismatches
// are diagnosed but recovered from as if the forms matched.
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE+11]]:12: error: redeclaration differs in whether the `self` type is written explicitly [RedeclParamSelfSyntaxDiffers]
// CHECK:STDERR: fn Class.F(self) {}
// CHECK:STDERR:            ^~~~
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE-9]]:14: note: comparing with previous declaration here [RedeclParamSelfSyntaxPrevious]
// CHECK:STDERR:   fn F(self: Self);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE+4]]:12: warning: binding `self` unused [UnusedBinding]
// CHECK:STDERR: fn Class.F(self) {}
// CHECK:STDERR:            ^~~~
// CHECK:STDERR:
fn Class.F(self) {}

// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE+11]]:18: error: redeclaration differs in whether the `self` type is written explicitly [RedeclParamSelfSyntaxDiffers]
// CHECK:STDERR: fn Class.G(self: Self) {}
// CHECK:STDERR:                  ^~~~
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE-21]]:8: note: comparing with previous declaration here [RedeclParamSelfSyntaxPrevious]
// CHECK:STDERR:   fn G(self);
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_self_syntax_mismatch.carbon:[[@LINE+4]]:12: warning: binding `self` unused [UnusedBinding]
// CHECK:STDERR: fn Class.G(self: Self) {}
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
fn Class.G(self: Self) {}

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
// CHECK:STDOUT:   %pattern_type.7c2: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.059: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc14 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %i32.loc14 [concrete = constants.%.70c]
// CHECK:STDOUT:     %self.param.loc14: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc14: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc14: %Class = value_binding self, %self.param.loc14
// CHECK:STDOUT:     %return.param.loc14: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc14: ref %i32 = return_slot %return.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c2 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc18 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32.loc18 [concrete = constants.%.70c]
// CHECK:STDOUT:     %self.param.loc18: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc18: ref %Class = ref_binding self, %self.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc18: ref %i32 = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc14 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32.loc6 [concrete = constants.%.70c]
// CHECK:STDOUT:     %self.param.loc6: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc6: %Class = value_binding self, %self.param.loc6
// CHECK:STDOUT:     %return.param.loc6: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc6: ref %i32 = return_slot %return.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c2 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc18 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32.loc8 [concrete = constants.%.70c]
// CHECK:STDOUT:     %self.param.loc8: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc8: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc8: ref %Class = ref_binding self, %self.param.loc8
// CHECK:STDOUT:     %return.param.loc8: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc8: ref %i32 = return_slot %return.param.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: %Class.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.059]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT:   .n = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param.loc14: %Class) -> out %return.param.loc14: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc14
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc10 [concrete = @Class.%.loc10]
// CHECK:STDOUT:   %.loc15_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc15_14.2: %i32 = acquire_value %.loc15_14.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc15_14.1: <bound method> = bound_method %.loc15_14.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_14.2: <bound method> = bound_method %.loc15_14.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_14.2(%.loc15_14.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param.loc18: ref %Class) -> out %return.param.loc18: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: ref %Class = name_ref self, %self.loc18
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc10 [concrete = @Class.%.loc10]
// CHECK:STDOUT:   %.loc19_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc19_14.2: %i32 = acquire_value %.loc19_14.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc19_14.1: <bound method> = bound_method %.loc19_14.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_14.2: <bound method> = bound_method %.loc19_14.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc19_14.2(%.loc19_14.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_syntax_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc22: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc22: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc22: %Class = value_binding self, %self.param.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc35: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc35: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc35: %Class = value_binding self, %self.param.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc5: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc5: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc5: %Class = value_binding self, %self.param.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc6: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc6: %Class = value_binding self, %self.param.loc6
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
// CHECK:STDOUT: fn @Class.F(%self.param.loc22: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param.loc35: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_self_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.345: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.345 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.345 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern <error>, <error> [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:     %.loc12: type = converted %self.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = value_binding self, %self.param
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
