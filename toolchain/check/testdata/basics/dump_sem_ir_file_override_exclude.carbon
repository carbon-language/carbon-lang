// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --exclude-dump-file-prefix=
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/dump_sem_ir_file_override_exclude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/dump_sem_ir_file_override_exclude.carbon

// --- file.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-file

// --- file_with_range.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-file

//@dump-sem-ir-begin
//@dump-sem-ir-end

// --- excluded_file.carbon

library "[[@TEST_NAME]]";

// --- excluded_file_with_range.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
//@dump-sem-ir-end

// CHECK:STDOUT: --- file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- file_with_range.carbon
// CHECK:STDOUT:
