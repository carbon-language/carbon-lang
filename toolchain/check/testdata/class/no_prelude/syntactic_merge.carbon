// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/syntactic_merge.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
class Foo(a:! C) {}

class Bar(a:! D);
class Bar(a:! D) {}

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo [ ] ( a :! C );
class Foo[](a:! C) {}

// --- fail_parens.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);
// CHECK:STDERR: fail_parens.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! (C)) {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_parens.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo(a:! (C)) {}

// --- todo_fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);
class Foo(a:! r#C) {}

// --- two_file.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
class Bar(a:! D);

// --- two_file.impl.carbon

impl library "[[@TEST_NAME]]";

class Foo(a:! C) {}
class Bar(a:! D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+7]]:11: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class Foo(b:! D) {}
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:11: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
class Foo(b:! D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! D) {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo(a:! D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo[a:! C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo[a:! D]() {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo[a:! C]();
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo[a:! D]() {}

// --- alias_two_file.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);

// --- todo_fail_alias_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

alias D = C;

class Foo(a:! D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+11]]:15: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: class Foo(a:! const (const C)) {}
// CHECK:STDERR:               ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+7]]:21: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! const (const C)) {}
// CHECK:STDERR:                     ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:21: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! const C);
// CHECK:STDERR:                     ^
// CHECK:STDERR:
class Foo(a:! const (const C)) {}

// --- fail_self_type.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: ();

  fn F[addr self: Self*]();
}

// CHECK:STDERR: fail_self_type.carbon:[[@LINE+7]]:22: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Base.F[addr self: Base*]() {
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR: fail_self_type.carbon:[[@LINE-6]]:19: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn F[addr self: Self*]();
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR:
fn Base.F[addr self: Base*]() {
  (*self).a = ();
}
