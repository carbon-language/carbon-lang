// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile foo.carbon --asm-output --output=- --target=x86_64-linux-gnu -- -fno-use-init-array
//
// NOAUTOUPDATE
// SET-CHECK-SUBSET
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/target_machine.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/target_machine.carbon

// --- foo.carbon

import Cpp inline '''
int f();
inline int n = f();
''';

// We should not use init_array, following the command-line flag.
// CHECK:STDOUT: {{\s*}}.section{{\s+}}.ctors,{{.*}}
// CHECK:STDOUT: {{\s*}}.quad{{\s+}}__cxx_global_var_init
