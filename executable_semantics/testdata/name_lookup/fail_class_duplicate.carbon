// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} --prelude=%{prelude} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --prelude=%{prelude} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/name_lookup/fail_class_duplicate.carbon:16: Duplicate name `x` also found at {{.*}}/executable_semantics/testdata/name_lookup/fail_class_duplicate.carbon:15

package ExecutableSemanticsTest api;

class Foo {
  var x: i32;
  var x: i32;
}

fn Main() -> i32 {
  return 0;
}
