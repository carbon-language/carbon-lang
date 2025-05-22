// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/destroy_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/destroy_decl.carbon

// --- self.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C {
  fn destroy[self: Self]();
}
//@dump-sem-ir-end

// --- addr_self.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C {
  fn destroy[addr self: Self*]();
}
//@dump-sem-ir-end

// --- explicit_return.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C {
  fn destroy[self: Self]() -> ();
}
//@dump-sem-ir-end

// --- fail_class_function.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_class_function.carbon:[[@LINE+4]]:3: error: missing implicit `self` parameter [DestroyFunctionMissingSelf]
  // CHECK:STDERR:   fn destroy();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  fn destroy();
}

// --- fail_extra_implicit_params_second.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_extra_implicit_params_second.carbon:[[@LINE+4]]:26: error: unexpected implicit parameter [DestroyFunctionUnexpectedImplicitParam]
  // CHECK:STDERR:   fn destroy[self: Self, T:! type]();
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  fn destroy[self: Self, T:! type]();
}

// --- fail_extra_implicit_params_first.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_extra_implicit_params_first.carbon:[[@LINE+4]]:14: error: unexpected implicit parameter [DestroyFunctionUnexpectedImplicitParam]
  // CHECK:STDERR:   fn destroy[T:! type, self: Self]();
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  fn destroy[T:! type, self: Self]();
}

// --- fail_positional_params.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_positional_params.carbon:[[@LINE+4]]:3: error: missing empty explicit parameter list [DestroyFunctionPositionalParams]
  // CHECK:STDERR:   fn destroy[self: Self];
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn destroy[self: Self];
}

// --- fail_explicit_params.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_explicit_params.carbon:[[@LINE+4]]:26: error: unexpected parameter [DestroyFunctionNonEmptyExplicitParams]
  // CHECK:STDERR:   fn destroy[self: Self](x: ());
  // CHECK:STDERR:                          ^~~~~
  // CHECK:STDERR:
  fn destroy[self: Self](x: ());
}

// --- fail_return_type.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_return_type.carbon:[[@LINE+4]]:28: error: incorrect return type; must be unspecified or `()` [DestroyFunctionIncorrectReturnType]
  // CHECK:STDERR:   fn destroy[self: Self]() -> {};
  // CHECK:STDERR:                            ^~~~~
  // CHECK:STDERR:
  fn destroy[self: Self]() -> {};
}

// --- fail_out_of_line_missing_params.carbon

library "[[@TEST_NAME]]";


class C {
  fn destroy[self: Self]();
}

// CHECK:STDERR: fail_out_of_line_missing_params.carbon:[[@LINE+7]]:1: error: redeclaration differs because of missing implicit parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fn C.destroy {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_out_of_line_missing_params.carbon:[[@LINE-6]]:3: note: previously declared with implicit parameter list [RedeclParamListPrevious]
// CHECK:STDERR:   fn destroy[self: Self]();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C.destroy {}

// --- fail_destroy_in_file_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_destroy_in_file_scope.carbon:[[@LINE+4]]:1: error: declaring `fn destroy` in non-class scope [DestroyFunctionOutsideClass]
// CHECK:STDERR: fn destroy[self: ()]();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn destroy[self: ()]();

// --- fail_destroy_in_namespace_scope.carbon

library "[[@TEST_NAME]]";

namespace NS;

// CHECK:STDERR: fail_destroy_in_namespace_scope.carbon:[[@LINE+4]]:1: error: declaring `fn destroy` in non-class scope [DestroyFunctionOutsideClass]
// CHECK:STDERR: fn NS.destroy();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NS.destroy();

// --- fail_invalid_qualifier_with_params.carbon

library "[[@TEST_NAME]]";

class C {
  fn destroy[self: Self]();
}

// CHECK:STDERR: fail_invalid_qualifier_with_params.carbon:[[@LINE+7]]:6: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: fn C.destroy[self: Self]().Foo() {}
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: fail_invalid_qualifier_with_params.carbon:[[@LINE-6]]:3: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR:   fn destroy[self: Self]();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C.destroy[self: Self]().Foo() {}

// --- fail_invalid_qualifier_no_params.carbon

library "[[@TEST_NAME]]";

class C {
  fn destroy[self: Self]();
}

// CHECK:STDERR: fail_invalid_qualifier_no_params.carbon:[[@LINE+7]]:6: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: fn C.destroy.Foo() {}
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: fail_invalid_qualifier_no_params.carbon:[[@LINE-6]]:3: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR:   fn destroy[self: Self]();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C.destroy.Foo() {}

// --- fail_return_type_after_no_params.carbon

library "[[@TEST_NAME]]";

// Return type handling special-cases the "no params" case.
class C {
  // CHECK:STDERR: fail_return_type_after_no_params.carbon:[[@LINE+4]]:3: error: missing implicit `self` parameter [DestroyFunctionMissingSelf]
  // CHECK:STDERR:   fn destroy -> ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn destroy -> ();
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %destroy.type: type = fn_type @destroy [concrete]
// CHECK:STDOUT:   %destroy: %destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %destroy.decl: %destroy.type = fn_decl @destroy [concrete = constants.%destroy] {
// CHECK:STDOUT:     %self.patt: %pattern_type = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- addr_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %destroy.type: type = fn_type @destroy [concrete]
// CHECK:STDOUT:   %destroy: %destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %destroy.decl: %destroy.type = fn_decl @destroy [concrete = constants.%destroy] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy(%self.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %destroy.type: type = fn_type @destroy [concrete]
// CHECK:STDOUT:   %destroy: %destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %destroy.decl: %destroy.type = fn_decl @destroy [concrete = constants.%destroy] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_32.2: type = converted %.loc6_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy(%self.param: %C) -> %empty_tuple.type;
// CHECK:STDOUT:
