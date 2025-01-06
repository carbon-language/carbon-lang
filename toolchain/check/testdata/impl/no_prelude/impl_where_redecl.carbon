// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_where_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_where_redecl.carbon

interface J {}

// `impl` matching ignores the `where` clause. It should not get confused by
// nested `where`, so the following should not trigger impl declaration without
// definition diagnostics.

impl () as J;
impl () as J where .Self impls type and .Self impls (type where .Self impls type) {}

impl {} as J where .Self impls type and .Self impls (type where .Self impls type);
impl {} as J {}
