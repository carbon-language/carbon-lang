// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/destroy_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/destroy_decl.carbon

// --- self.carbon

library "[[@TEST_NAME]]";

class C {
  fn destroy[self: Self]();
}

// --- addr_self.carbon

library "[[@TEST_NAME]]";

class C {
  fn destroy[addr self: Self*]();
}

// --- explicit_return.carbon

library "[[@TEST_NAME]]";

class C {
  fn destroy[self: Self]() -> ();
}

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
