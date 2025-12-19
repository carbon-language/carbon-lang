// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/misplaced_decl_in_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/misplaced_decl_in_constraint.carbon

// --- fail_adapt_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_adapt_in_constraint.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt ();
}

// --- fail_base_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_base_in_constraint.carbon:[[@LINE+4]]:3: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   extend base: ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: ();
}

// --- fail_class_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_class_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``class` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   class C;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  class C;
}

// --- fail_constraint_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_constraint_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``constraint` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   constraint N {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  constraint N {}
}

// --- fail_export_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_export_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``export` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   export E;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  export E;
}

// --- fail_fn_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint E {
  // CHECK:STDERR: fail_fn_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``fn` in named constraint` [SemanticsTodo]
  // CHECK:STDERR:   fn DeclaredFunction();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn DeclaredFunction();
  // CHECK:STDERR: fail_fn_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``fn` in named constraint` [SemanticsTodo]
  // CHECK:STDERR:   fn DeclaredMethod[self: Self](b: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn DeclaredMethod[self: Self](b: Self);
}

constraint F {
  // CHECK:STDERR: fail_fn_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``fn` in named constraint` [SemanticsTodo]
  // CHECK:STDERR:   fn DefinedFunction() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn DefinedFunction() {}
  // CHECK:STDERR: fail_fn_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``fn` in named constraint` [SemanticsTodo]
  // CHECK:STDERR:   fn DefinedMethod[self: Self](b: Self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn DefinedMethod[self: Self](b: Self) {}
}

// --- fail_impl_in_constraint.carbon
library "[[@TEST_NAME]]";

interface Y;

constraint N {
  // CHECK:STDERR: fail_impl_in_constraint.carbon:[[@LINE+4]]:8: error: semantics TODO: ``impl` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   impl () as Y;
  // CHECK:STDERR:        ^~~~~
  // CHECK:STDERR:
  impl () as Y;
}

// --- fail_interface_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_interface_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: ``interface` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   interface Y {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  interface Y {}
}

// --- fail_let_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_let_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated constant in constraint` [SemanticsTodo]
  // CHECK:STDERR:   let X:! type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let X:! type;
}

// --- fail_namespace_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_namespace_in_constraint.carbon:[[@LINE+4]]:3: error: `namespace` declaration not at top level [NamespaceDeclNotAtTopLevel]
  // CHECK:STDERR:   namespace N;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  namespace N;
}

// --- fail_var_in_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {
  // CHECK:STDERR: fail_var_in_constraint.carbon:[[@LINE+8]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   var v: ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var v: ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  var v: ();
}
