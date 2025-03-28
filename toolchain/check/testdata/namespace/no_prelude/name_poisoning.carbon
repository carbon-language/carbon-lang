// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// `N2.N3` uses `N2.N1` and not `package.N1`.
namespace N2;
namespace N2.N1;
alias N2.N3 = N1;

class N2.N1.C {}
class N1.C {}
fn TestNamespaces() {
  var x: N2.N1.C;
  var y: N2.N3.C* = &x;
}

// --- poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
alias N2.N3 = N1;

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:15: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3 = N1;
// CHECK:STDERR:               ^~
alias N2.N3 = N1;

// Failure: N2.N1 declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N2.N1;

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
alias N2.N3 = N1;

// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:15: error: member name `N1` not found in `N2` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: alias N2.N4 = N2.N1;
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR:
alias N2.N4 = N2.N1;

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:15: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3 = N1;
// CHECK:STDERR:               ^~
alias N2.N3 = N1;

// Failure: `N2.N1` declared after it was poisoned.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N2.N1;

// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:15: error: member name `N1` not found in `N2` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: alias N2.N4 = N2.N1;
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR:
alias N2.N4 = N2.N1;

// --- fail_alias.carbon

namespace N1;

namespace N2;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:15: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N1 = N1;
// CHECK:STDERR:               ^~
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:10: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N2.N1 = N1;
// CHECK:STDERR:          ^~
// CHECK:STDERR:
alias N2.N1 = N1;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

namespace N1;
namespace N2;
namespace N2.N3;
namespace N2.N3.N4;
// Use `package.N1` and poison:
// * `N2.N1`
// * `N2.N3.N1`
// * `N2.N3.N4.N1`
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:21: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3.N4.N5 = N1;
// CHECK:STDERR:                     ^~
alias N2.N3.N4.N5 = N1;

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:20: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N3.N4.N1;
// CHECK:STDERR:                    ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:21: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3.N4.N5 = N1;
// CHECK:STDERR:                     ^~
namespace N2.N3.N4.N1;
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:17: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N3.N1;
// CHECK:STDERR:                 ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-14]]:21: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3.N4.N5 = N1;
// CHECK:STDERR:                     ^~
namespace N2.N3.N1;
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N2.N1;

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
namespace N2.N1;

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N2.N1` was poisoned in the api.
namespace N2.N1;

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N1;
// `N1.N3` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:15: error: name `N3` not found [NameNotFound]
// CHECK:STDERR: alias N1.N2 = N3;
// CHECK:STDERR:               ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:15: error: name `N3` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2 = N3;
// CHECK:STDERR:               ^~
alias N1.N2 = N3;

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N3;
// CHECK:STDERR:           ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:15: error: name `N3` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2 = N3;
// CHECK:STDERR:               ^~
namespace N3;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N1.N3;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N1.N3;
