// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/name_poisoning.carbon

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
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:14: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.F2 = F1;
// CHECK:STDERR:              ^~
alias N.F2 = F1;

// Failure: `N.F1` declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:6: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N.F1();
// CHECK:STDERR:      ^~
// CHECK:STDERR:
fn N.F1();

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

fn F1();

// Use `package.F1` and poison `N.F1`.
namespace N;
alias N.F2 = F1;

// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:14: error: member name `F1` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: alias N.F3 = N.F1;
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
alias N.F3 = N.F1;

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

fn F1();

namespace N;
// Use `package.F1` and poison `N.F1`.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:14: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.F2 = F1;
// CHECK:STDERR:              ^~
alias N.F2 = F1;

// Failure: `N.F1` declared after it was poisoned.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:6: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N.F1();
// CHECK:STDERR:      ^~
// CHECK:STDERR:
fn N.F1();

// Failure: `N.F1` used after declaration failed.
// TODO: #4622 - Allow defining a poisoned name so it would be found if used after it's declared.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:14: error: member name `F1` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: alias N.F3 = N.F1;
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
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
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:21: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2.N3.F2 = F1;
// CHECK:STDERR:                     ^~
alias N1.N2.N3.F2 = F1;

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N1.F1();
// CHECK:STDERR:       ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:21: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2.N3.F2 = F1;
// CHECK:STDERR:                     ^~
fn N1.F1();
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:10: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N1.N2.F1();
// CHECK:STDERR:          ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-14]]:21: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2.N3.F2 = F1;
// CHECK:STDERR:                     ^~
fn N1.N2.F1();
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N1.N2.N3.F1();
// CHECK:STDERR:             ^~
// CHECK:STDERR:
fn N1.N2.N3.F1();

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

fn F();

namespace N;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:13: error: name `F` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.F = F;
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N.F = F;
// CHECK:STDERR:         ^
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
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:14: error: name `F1` not found [NameNotFound]
// CHECK:STDERR: alias N.F2 = F1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:14: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.F2 = F1;
// CHECK:STDERR:              ^~
alias N.F2 = F1;

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:4: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn F1();
// CHECK:STDERR:    ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:14: error: name `F1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.F2 = F1;
// CHECK:STDERR:              ^~
fn F1();
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:6: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N.F1();
// CHECK:STDERR:      ^~
// CHECK:STDERR:
fn N.F1();

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F1() {
  fn F2();

  class C {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+3]]:16: error: name `F2` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     alias F3 = F2;
    // CHECK:STDERR:                ^~
    alias F3 = F2;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+4]]:8: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     fn F2();
    // CHECK:STDERR:        ^~
    // CHECK:STDERR:
    fn F2();
  }
}
