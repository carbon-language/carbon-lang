// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: config --target=x86_64-unknown-linux-gnu
//
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/config.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/config.carbon

// We don't include the digest in `file_test` to avoid unnecessary dependencies.
// CHECK:STDERR: error: unable to read the installation's digest file: {{.+}}
// CHECK:STDERR:

// The rest of the config should still be displayed.
//
// Note that the results here are not "portable", and depend on the `--target`
// argument above. We pass explicit targets to get predictable results in Bazel
// tests.
//
// CHECK:STDOUT: CLANG_INCLUDE_DIRS:
// CHECK:STDOUT:     {{.*}}/toolchain/install/llvm/lib/clang/{{[^/]+}}/include
// CHECK:STDOUT: CLANG_RESOURCE_DIR: {{.*}}/toolchain/install/llvm/lib/clang/{{[^/]+}}
// CHECK:STDOUT: CLANG_SYSROOT: /
// CHECK:STDOUT: INSTALL_ROOT: {{.*}}/toolchain/install/
// CHECK:STDOUT: LLVM_BINDIR: {{.*}}/toolchain/install/llvm/bin
// CHECK:STDOUT: VERSION: {{[0-9]+}}.{{[0-9]+}}.{{[0-9]+}}-{{.+}}
