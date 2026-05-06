// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_cpp_import_name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_cpp_import_name_poisoning.carbon

// --- fail_fn_after_cpp_import_poison.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace Carbon {
  class Name {};
}
''';

// CHECK:STDERR: fail_fn_after_cpp_import_poison.carbon:[[@LINE+7]]:4: error: name `Name` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn Name() {}
// CHECK:STDERR:    ^~~~
// CHECK:STDERR: fail_fn_after_cpp_import_poison.carbon:[[@LINE+4]]:4: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn Name() {}
// CHECK:STDERR:    ^~~~
// CHECK:STDERR:
fn Name() {}

// --- fail_class_after_cpp_import_poison.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace Carbon {
  class Name {};
}
''';

// CHECK:STDERR: fail_class_after_cpp_import_poison.carbon:[[@LINE+7]]:7: error: name `Name` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: class Name {}
// CHECK:STDERR:       ^~~~
// CHECK:STDERR: fail_class_after_cpp_import_poison.carbon:[[@LINE+4]]:7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class Name {}
// CHECK:STDERR:       ^~~~
// CHECK:STDERR:
class Name {}

// --- fail_interface_after_cpp_import_poison.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace Carbon {
  class Name {};
}
''';

// CHECK:STDERR: fail_interface_after_cpp_import_poison.carbon:[[@LINE+7]]:11: error: name `Name` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: interface Name {}
// CHECK:STDERR:           ^~~~
// CHECK:STDERR: fail_interface_after_cpp_import_poison.carbon:[[@LINE+4]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface Name {}
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
interface Name {}

// --- fail_namespace_after_cpp_import_poison.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace Carbon {
  class Name {};
}
''';

// CHECK:STDERR: fail_namespace_after_cpp_import_poison.carbon:[[@LINE+7]]:11: error: name `Name` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: namespace Name;
// CHECK:STDERR:           ^~~~
// CHECK:STDERR: fail_namespace_after_cpp_import_poison.carbon:[[@LINE+4]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace Name;
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
namespace Name;

// --- ok_no_conflict.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace Carbon {
  class Name {};
}
''';

fn Different() {}
