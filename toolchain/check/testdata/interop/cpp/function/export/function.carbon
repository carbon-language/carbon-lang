// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/function.carbon

// --- other_function.carbon

package OtherFunction;
fn F1() {}

// --- function.carbon

library "[[@TEST_NAME]]";

import OtherFunction;
import Cpp inline '''
void CallF1() {
  Carbon::OtherFunction::F1();
}
''';

// --- other_return_int.carbon

package OtherReturnInt;
fn F2() -> i32 { return 0; }

// --- return_int.carbon

library "[[@TEST_NAME]]";

import OtherReturnInt;
import Cpp inline '''
int CallF2() {
  return Carbon::OtherReturnInt::F2();
}
''';

// --- other_args.carbon

package OtherArgs;
fn F3(_: i32) {}

// --- args.carbon

library "[[@TEST_NAME]]";

import OtherArgs;
import Cpp inline '''
void CallF3() {
  Carbon::OtherArgs::F3(123);
}
''';

// --- ref_arg.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn RefF(ref n: i32);

inline Cpp '''
void CallRefF() {
  int n = 0;
  Carbon::RefF(n);
}
''';

// --- using.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn UsingF();

inline Cpp '''
using Carbon::UsingF;
void CallUsingF() { UsingF(); }
''';

// --- other_generic.carbon

package OtherGeneric;
fn HasGenericArg(generic T: type, a: T) { a; }

// --- fail_todo_generic.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: imported from: fail_todo_generic.carbon:[[@LINE+4]]:1 [InImport]
// CHECK:STDERR: other_generic.carbon:3:1-41: error: 1 argument passed to function expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: other_generic.carbon:3:1-41: note: calling function declared here [InCallToEntity]
// CHECK:STDERR:
import OtherGeneric;
import Cpp inline '''
void CallHasGenericArg() {
  Carbon::OtherGeneric::HasGenericArg<int>(123);
}
''';

// --- other_deduced.carbon

package OtherDeduced;
fn HasDeducedArg[T: type](a: T) { a; }

// --- deduced.carbon

library "[[@TEST_NAME]]";

import OtherDeduced;
import Cpp inline '''
void CallHasDeducedArg() {
  Carbon::OtherDeduced::HasDeducedArg(123);
}
''';
