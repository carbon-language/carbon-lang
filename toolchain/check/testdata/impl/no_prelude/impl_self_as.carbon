// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_self_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_self_as.carbon

// --- match.carbon
library "[[@TEST_NAME]]";

interface I1 {}
interface I2 {}
interface J1(T1:! type) {}
interface J2(T2:! type) {}

// `impl Self as` should match `impl as`, so these should not trigger impl
// declaration without definition diagnostics.

class C1 {
  impl Self as I1;
  impl as I1 {}

  impl as I2;
  impl Self as I2 {}

  impl forall [U:! type] Self as J1(U);
  impl forall [U:! type] as J1(U) {}

  impl forall [V:! type] as J2(V);
  impl forall [V:! type] Self as J2(V) {}
}

class C2(W:! type) {
  impl Self as I1;
  impl as I1 {}

  impl as I2;
  impl Self as I2 {}

  impl forall [X:! type] Self as J1(X);
  impl forall [X:! type] as J1(X) {}

  impl forall [Y:! type] as J2(Y);
  impl forall [Y:! type] Self as J2(Y) {}
}


// --- fail_no_match.carbon
library "[[@TEST_NAME]]";

interface I3 {}
interface I4 {}
interface I5 {}
interface I6 {}
interface J3(T3:! type) {}
interface J4(T4:! type) {}
interface J5(T5:! type) {}
interface J6(T6:! type) {}

// `impl C as` should not match `impl Self as` or `impl as`.

class C3 {
  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl C3 as I3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C3 as I3;
  impl as I3 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl C3 as I4;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C3 as I4;
  impl Self as I4 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl as I5;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I5;
  impl C3 as I5 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl Self as I6;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl Self as I6;
  impl C3 as I6 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [Z3:! type] C3 as J3(Z3);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z3:! type] C3 as J3(Z3);
  impl forall [Z3:! type] as J3(Z3) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [Z4:! type] C3 as J4(Z4);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z4:! type] C3 as J4(Z4);
  impl forall [Z4:! type] Self as J4(Z4) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [Z5:! type] as J5(Z5);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z5:! type] as J5(Z5);
  impl forall [Z5:! type] C3 as J5(Z5) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [Z6:! type] Self as J6(Z6);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z6:! type] Self as J6(Z6);
  impl forall [Z6:! type] C3 as J6(Z6) {}
}

class C4(A:! type) {
  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl C4(A) as I3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C4(A) as I3;
  impl as I3 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl C4(A) as I4;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C4(A) as I4;
  impl Self as I4 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl as I5;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I5;
  impl C4(A) as I5 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl Self as I6;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl Self as I6;
  impl C4(A) as I6 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [B3:! type] C4(A) as J3(B3);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B3:! type] C4(A) as J3(B3);
  impl forall [B3:! type] as J3(B3) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [B4:! type] C4(A) as J4(B4);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B4:! type] C4(A) as J4(B4);
  impl forall [B4:! type] Self as J4(B4) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [B5:! type] as J5(B5);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B5:! type] as J5(B5);
  impl forall [B5:! type] C4(A) as J5(B5) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   impl forall [B6:! type] Self as J6(B6);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B6:! type] Self as J6(B6);
  impl forall [B6:! type] C4(A) as J6(B6) {}
}
