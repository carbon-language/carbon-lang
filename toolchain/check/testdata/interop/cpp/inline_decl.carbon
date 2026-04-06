// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/inline_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/inline_decl.carbon

// --- inline_decl.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
void func1() {}
''';

fn Run1() {
  Cpp.func1();
}

inline Cpp '''
void func2() {
  Carbon::Run1();
}
''';

fn Run2() {
  Cpp.func2();
}

// --- fail_redefinition.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
void f() {}
''';

inline Cpp '''
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:6: error: redefinition of 'f' [CppInteropParseError]
// CHECK:STDERR:    18 | void f() {}
// CHECK:STDERR:       |      ^
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-7]]:6: note: previous definition is here [CppInteropParseNote]
// CHECK:STDERR:     7 | void f() {}
// CHECK:STDERR:       |      ^
// CHECK:STDERR:
void f() {}
''';

// --- fail_diag_location.carbon

library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_diag_location.carbon:[[@LINE+4]]:12: error: use of undeclared identifier 'undeclared' [CppInteropParseError]
// CHECK:STDERR:    10 | void f() { undeclared = 0; }
// CHECK:STDERR:       |            ^~~~~~~~~~
// CHECK:STDERR:
inline Cpp "void f() { undeclared = 0; }";

inline Cpp '''
void g() {
  // CHECK:STDERR: fail_diag_location.carbon:[[@LINE+4]]:3: error: use of undeclared identifier 'undeclared' [CppInteropParseError]
  // CHECK:STDERR:    18 |   undeclared = 0;
  // CHECK:STDERR:       |   ^~~~~~~~~~
  // CHECK:STDERR:
  undeclared = 0;
}
''';

// --- fail_inline_no_import.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_inline_no_import.carbon:[[@LINE+4]]:8: error: name `Cpp` not found [NameNotFound]
// CHECK:STDERR: inline Cpp "";
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
inline Cpp "";

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:3: error: `inline Cpp` declaration not at file scope [InlineDeclNotAtFileScope]
  // CHECK:STDERR:   inline Cpp "";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  inline Cpp "";
}

interface I {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:3: error: `inline Cpp` declaration not at file scope [InlineDeclNotAtFileScope]
  // CHECK:STDERR:   inline Cpp "";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  inline Cpp "";
}

constraint N {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:3: error: `inline Cpp` declaration not at file scope [InlineDeclNotAtFileScope]
  // CHECK:STDERR:   inline Cpp "";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  inline Cpp "";
}

impl C as I {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:3: error: `inline Cpp` declaration not at file scope [InlineDeclNotAtFileScope]
  // CHECK:STDERR:   inline Cpp "";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  inline Cpp "";
}

// --- fail_nested_fn.carbon

fn F() {
  // TODO: Should this be a check error rather than a parse error?
  // CHECK:STDERR: fail_nested_fn.carbon:[[@LINE+8]]:3: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   inline Cpp "";
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_nested_fn.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   inline Cpp "";
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  inline Cpp "";
}
