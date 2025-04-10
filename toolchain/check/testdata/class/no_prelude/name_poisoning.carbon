// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

class C;

// `N.F` uses `N.C` and not `package.C`.
namespace N;
class N.C {}
fn N.F(x: C);

fn TestCall(x: N.C) {
  // `N.F` accepts an `N.C` not a `package.C`.
  N.F(x);
}

// --- poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
fn N.F(x: C);

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:11: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
fn N.F(x: C);

// Failure: `N.C` declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
class N.C;

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
fn N.F1() -> C;

// Use `N.C` which was poisoned and not declared.
// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:14: error: member name `C` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn N.F2() -> N.C;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
fn N.F2() -> N.C;

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:12: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:            ^
fn N.F1(x: C);

// Failure: N.C declared after it was poisoned.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
class N.C;

// Failure: `N.C` used after declaration failed.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:12: error: member name `C` not found in `N` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn N.F2(x: N.C);
// CHECK:STDERR:            ^~~
// CHECK:STDERR:
fn N.F2(x: N.C);

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:13: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.C = C;
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N.C = C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias N.C = C;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

class C1;

class C2 {
  class C3 {
    class C4 {
      // Use `package.C1` and poison:
      // * `C2.C1`
      // * `C2.C3.C1`
      // * `C2.C3.C4.C1`
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:15: error: name `C1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: C1);
      // CHECK:STDERR:               ^~
      fn F(x: C1);

      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:13: note: declared here [NameUseBeforeDeclNote]
      // CHECK:STDERR:       class C1;
      // CHECK:STDERR:             ^~
      // CHECK:STDERR:
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:15: error: name `C1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: C1);
      // CHECK:STDERR:               ^~
      class C1;
    }
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:11: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class C1;
    // CHECK:STDERR:           ^~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-15]]:15: error: name `C1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:       fn F(x: C1);
    // CHECK:STDERR:               ^~
    class C1;
  }
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   class C1;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  class C1;
}

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
class N.C;

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N.C` was poisoned in the api.
class N.C {}

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N;
// `package.C` and `N.C` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: error: name `C` not found [NameNotFound]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:11: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
fn N.F(x: C);

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class C;
// CHECK:STDERR:       ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:11: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
class C;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
class N.C;

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F() {
  class C1 {}

  class C2 {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+3]]:12: error: name `C1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     var v: C1;
    // CHECK:STDERR:            ^~
    var v: C1;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+4]]:11: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class C1;
    // CHECK:STDERR:           ^~
    // CHECK:STDERR:
    class C1;
  }
}

// --- fail_declare_data_member_after_poison.carbon

library "[[@TEST_NAME]]";

class C1;

class C2 {
  // Use `package.C1` and poison `C2.C1`.
  // CHECK:STDERR: fail_declare_data_member_after_poison.carbon:[[@LINE+3]]:11: error: name `C1` used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:   fn F(x: C1);
  // CHECK:STDERR:           ^~
  fn F(x: C1);

  class C2 {}
  // Failure: `C2.C1` declared after it was poisoned.
  // CHECK:STDERR: fail_declare_data_member_after_poison.carbon:[[@LINE+4]]:7: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   var C1: C2;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var C1: C2;
}

// --- fail_extend_poison_class_members.carbon

library "[[@TEST_NAME]]";

base class B {}

class C {
  // CHECK:STDERR: fail_extend_poison_class_members.carbon:[[@LINE+3]]:16: error: name `B` used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:   extend base: B;
  // CHECK:STDERR:                ^
  extend base: B;

  // CHECK:STDERR: fail_extend_poison_class_members.carbon:[[@LINE+4]]:6: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   fn B();
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  fn B();
}

// --- todo_fail_poisoned_name_in_extend.carbon

library "[[@TEST_NAME]]";

base class B {
  fn F1();
}

class F1 {}

class C {
  // Use `package.F1` and poison `C.F1`.
  fn F2(x: F1);

  // TODO: #4622 This should diagnose since `C.F1` is poisoned.
  extend base: B;
}
