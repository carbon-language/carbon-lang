// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

class C1 {}
class C2 {}

fn F1(x: C1);

// `N.F2` uses `N.F1` and not `package.F1`.
namespace N;
fn N.F1(x: C2);
alias N.F2 = F1;

fn TestCall(x: C2) {
  // `N.F2` accepts a `C2` not a `C1`.
  N.F2(x);
}

// --- poison.carbon

library "[[@TEST_NAME]]";

fn F1();

// Use `package.F1` and poison `N.F1`.
namespace N;
alias N.F2 = F1;

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

fn F1();

namespace N;
// Use `package.F1` and poison `N.F1`.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+1]]:14-15: error: name `F1` used before it was declared [NameUseBeforeDecl]
alias N.F2 = F1;

// Failure: `N.F1` declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+2]]:6-7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn N.F1();

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

fn F1();

// Use `package.F1` and poison `N.F1`.
namespace N;
alias N.F2 = F1;

// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+2]]:14-17: error: member name `F1` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR:
alias N.F3 = N.F1;

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

fn F1();

namespace N;
// Use `package.F1` and poison `N.F1`.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+1]]:14-15: error: name `F1` used before it was declared [NameUseBeforeDecl]
alias N.F2 = F1;

// Failure: `N.F1` declared after it was poisoned.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+2]]:6-7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn N.F1();

// Don't produce an additional diagnostic here.
alias N.F3 = N.F1;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

fn F1();

namespace N1;
namespace N1.N2;
namespace N1.N2.N3;
// Use `package.F1` and poison:
// * `N1.F1`
// * `N1.N2.F1`
// * `N1.N2.N3.F1`
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+1]]:21-22: error: name `F1` used before it was declared [NameUseBeforeDecl]
alias N1.N2.N3.F2 = F1;

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:7-8: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-4]]:21-22: error: name `F1` used before it was declared [NameUseBeforeDecl]
fn N1.F1();
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:10-11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-8]]:21-22: error: name `F1` used before it was declared [NameUseBeforeDecl]
fn N1.N2.F1();
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+2]]:13-14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn N1.N2.N3.F1();

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

fn F();

namespace N;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+3]]:13: error: name `F` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fail_alias.carbon:[[@LINE+2]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
alias N.F = F;

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
fn N.F1();

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N.F1` was poisoned in the api.
fn N.F1() {}

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N;
// `N.F1` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:14-15: error: name `F1` not found [NameNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+1]]:14-15: error: name `F1` used before it was declared [NameUseBeforeDecl]
alias N.F2 = F1;

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:4-5: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-5]]:14-15: error: name `F1` used before it was declared [NameUseBeforeDecl]
fn F1();
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+2]]:6-7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR:
fn N.F1();

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F1() {
  fn F2();

  class C {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+1]]:16-17: error: name `F2` used before it was declared [NameUseBeforeDecl]
    alias F3 = F2;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+2]]:8-9: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:
    fn F2();
  }
}
