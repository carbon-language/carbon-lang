// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/require_invalid_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/require_invalid_modifiers.carbon

// --- fail_abstract_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_abstract_require.carbon:[[@LINE+4]]:3: error: `abstract` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   abstract require impls Y;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  abstract require impls Y;
}

// --- fail_base_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_base_require.carbon:[[@LINE+4]]:3: error: `base` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   base require impls Y;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  base require impls Y;
}

// --- fail_default_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_default_require.carbon:[[@LINE+4]]:3: error: `default` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   default require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default require impls Y;
}

// --- fail_extern_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_extern_require.carbon:[[@LINE+4]]:3: error: `extern` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   extern require impls Y;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extern require impls Y;
}

// --- fail_final_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_final_require.carbon:[[@LINE+4]]:3: error: `final` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   final require impls Y;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final require impls Y;
}

// --- fail_impl_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_impl_require.carbon:[[@LINE+4]]:3: error: `impl` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   impl require impls Y;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  impl require impls Y;
}

// --- fail_override_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_override_require.carbon:[[@LINE+4]]:3: error: `override` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   override require impls Y;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  override require impls Y;
}

// --- fail_returned_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_returned_require.carbon:[[@LINE+8]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   returned require impls Y;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_returned_require.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   returned require impls Y;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  returned require impls Y;
}

// --- fail_private_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_private_require.carbon:[[@LINE+4]]:3: error: `private` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   private require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private require impls Y;
}

// --- fail_protected_require.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_protected_require.carbon:[[@LINE+4]]:3: error: `protected` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   protected require impls Y;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  protected require impls Y;
}

// --- fail_override_virtual.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_override_virtual.carbon:[[@LINE+4]]:3: error: `virtual` not allowed on `require` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   virtual require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual require impls Y;
}
