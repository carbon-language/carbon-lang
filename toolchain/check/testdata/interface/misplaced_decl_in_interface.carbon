// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/misplaced_decl_in_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/misplaced_decl_in_interface.carbon

// --- fail_adapt_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_adapt_in_interface.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt ();
}

// --- fail_base_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_base_in_interface.carbon:[[@LINE+4]]:3: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   extend base: ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: ();
}

// --- fail_class_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_class_in_interface.carbon:[[@LINE+4]]:3: error: semantics TODO: ``class` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   class C;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  class C;
}

// --- fail_constraint_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_constraint_in_interface.carbon:[[@LINE+4]]:3: error: semantics TODO: ``constraint` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   constraint N {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  constraint N {}
}

// --- fail_export_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_export_in_interface.carbon:[[@LINE+4]]:3: error: semantics TODO: ``export` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   export E;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  export E;
}

// --- fail_impl_in_interface.carbon
library "[[@TEST_NAME]]";

interface Y;

interface Z {
  // CHECK:STDERR: fail_impl_in_interface.carbon:[[@LINE+4]]:8: error: semantics TODO: ``impl` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   impl () as Y;
  // CHECK:STDERR:        ^~~~~
  // CHECK:STDERR:
  impl () as Y;
}

// --- fail_interface_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_interface_in_interface.carbon:[[@LINE+4]]:3: error: semantics TODO: ``interface` decl in interface or constraint` [SemanticsTodo]
  // CHECK:STDERR:   interface Y {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  interface Y {}
}

// --- fail_namespace_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_namespace_in_interface.carbon:[[@LINE+4]]:3: error: `namespace` declaration not at top level [NamespaceDeclNotAtTopLevel]
  // CHECK:STDERR:   namespace N;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  namespace N;
}

// --- fail_var_in_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  // CHECK:STDERR: fail_var_in_interface.carbon:[[@LINE+8]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   var v: ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var_in_interface.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var v: ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  var v: ();
}
