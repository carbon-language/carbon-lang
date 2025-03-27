// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

interface I;

// `N.F` uses `N.I` and not `package.I`.
namespace N;
interface N.I {}
fn N.F(x:! I) {}

fn TestCall(x:! N.I) {
  // `N.F` accepts an `N.I` not a `package.I`.
  N.F(x);
}

// --- poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F(x:! I);

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:12: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x:! I);
// CHECK:STDERR:            ^
fn N.F(x:! I);

// Failure: N.I declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface N.I {}

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F1(x:! I);

// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:13: error: member name `I` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn N.F2(x:! N.I);
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
fn N.F2(x:! N.I);

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:13: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x:! I);
// CHECK:STDERR:             ^
fn N.F1(x:! I);

// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I;
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface N.I;

// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:13: error: member name `I` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn N.F2(x:! N.I);
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
fn N.F2(x:! N.I);

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

interface I;
namespace N;

// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:13: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.I = I;
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N.I = I;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias N.I = I;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

interface I1;

interface I2 {
  interface I3 {
    interface I4 {
      // Use `package.I1` and poison:
      // * `I2.I1`
      // * `I2.I3.I1`
      // * `I2.I3.I4.I1`
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:16: error: name `I1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x:! I1);
      // CHECK:STDERR:                ^~
      fn F(x:! I1);

      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:17: note: declared here [NameUseBeforeDeclNote]
      // CHECK:STDERR:       interface I1;
      // CHECK:STDERR:                 ^~
      // CHECK:STDERR:
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:16: error: name `I1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x:! I1);
      // CHECK:STDERR:                ^~
      interface I1;
    }
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:15: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     interface I1;
    // CHECK:STDERR:               ^~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-15]]:16: error: name `I1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:       fn F(x:! I1);
    // CHECK:STDERR:                ^~
    interface I1;
  }
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   interface I1;
  // CHECK:STDERR:             ^~
  // CHECK:STDERR:
  interface I1;
}

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
interface N.I;

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N.I` was poisoned in the api.
interface N.I;

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N;
// `package.I` and `N.I` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:12: error: name `I` not found [NameNotFound]
// CHECK:STDERR: fn N.F(x:! I);
// CHECK:STDERR:            ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:12: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x:! I);
// CHECK:STDERR:            ^
fn N.F(x:! I);

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface I;
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:12: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x:! I);
// CHECK:STDERR:            ^
interface I;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I;
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface N.I;

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F() {
  interface I1 {}

  class C {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+3]]:12: error: name `I1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     var v: I1;
    // CHECK:STDERR:            ^~
    var v: I1;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+4]]:15: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     interface I1;
    // CHECK:STDERR:               ^~
    // CHECK:STDERR:
    interface I1;
  }
}
