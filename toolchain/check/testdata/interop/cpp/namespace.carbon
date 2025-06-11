// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-prelude-import
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/namespace.carbon

// ============================================================================
// Single
// ============================================================================

// --- single.h

namespace my_namespace { void foo(); }

// --- import_single.carbon

library "[[@TEST_NAME]]";

import Cpp library "single.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.my_namespace.foo();
  //@dump-sem-ir-end
}

// --- fail_import_namespace_wrong_member_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "single.h";

fn MyF() {
  // CHECK:STDERR: fail_import_namespace_wrong_member_name.carbon:[[@LINE+4]]:3: error: member name `not_foo` not found in `Cpp.my_namespace` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.my_namespace.not_foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.my_namespace.not_foo();
}

// --- fail_import_namespace_wrong_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "single.h";

fn MyF() {
  // CHECK:STDERR: fail_import_namespace_wrong_name.carbon:[[@LINE+4]]:3: error: member name `not_my_namespace` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.not_my_namespace.foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.not_my_namespace.foo();
}

// ============================================================================
// Multiple
// ============================================================================

// --- multiple.h

namespace my_namespace1 {
  void foo1();
  namespace my_namespace2 {
    void foo2();
    namespace my_namespace3 {
      void foo3();
    }
  }
}

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.my_namespace1.foo1();
  Cpp.my_namespace1.my_namespace2.foo2();
  Cpp.my_namespace1.my_namespace2.my_namespace3.foo3();
  //@dump-sem-ir-end
}

// ============================================================================
// Carbon keyword name
// ============================================================================

// --- special_name.h

namespace base { void foo(); }

// --- fail_import_special_name_call_unescpaed.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  // CHECK:STDERR: fail_import_special_name_call_unescpaed.carbon:[[@LINE+4]]:3: error: member name `base` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.base.foo();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  Cpp.base.foo();
}

// --- import_special_name_call_escpaed.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.r#base.foo();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_single.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .my_namespace = %my_namespace
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %my_namespace: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace.ref: <namespace> = name_ref my_namespace, imports.%my_namespace [concrete = imports.%my_namespace]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo1.type: type = fn_type @foo1 [concrete]
// CHECK:STDOUT:   %foo1: %foo1.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo2.type: type = fn_type @foo2 [concrete]
// CHECK:STDOUT:   %foo2: %foo2.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo3.type: type = fn_type @foo3 [concrete]
// CHECK:STDOUT:   %foo3: %foo3.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .my_namespace1 = %my_namespace1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %my_namespace1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo1 = %foo1.decl
// CHECK:STDOUT:     .my_namespace2 = %my_namespace2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo1.decl: %foo1.type = fn_decl @foo1 [concrete = constants.%foo1] {} {}
// CHECK:STDOUT:   %my_namespace2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo2 = %foo2.decl
// CHECK:STDOUT:     .my_namespace3 = %my_namespace3
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo2.decl: %foo2.type = fn_decl @foo2 [concrete = constants.%foo2] {} {}
// CHECK:STDOUT:   %my_namespace3: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo3 = %foo3.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo3.decl: %foo3.type = fn_decl @foo3 [concrete = constants.%foo3] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace1.ref.loc8: <namespace> = name_ref my_namespace1, imports.%my_namespace1 [concrete = imports.%my_namespace1]
// CHECK:STDOUT:   %foo1.ref: %foo1.type = name_ref foo1, imports.%foo1.decl [concrete = constants.%foo1]
// CHECK:STDOUT:   %foo1.call: init %empty_tuple.type = call %foo1.ref()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace1.ref.loc9: <namespace> = name_ref my_namespace1, imports.%my_namespace1 [concrete = imports.%my_namespace1]
// CHECK:STDOUT:   %my_namespace2.ref.loc9: <namespace> = name_ref my_namespace2, imports.%my_namespace2 [concrete = imports.%my_namespace2]
// CHECK:STDOUT:   %foo2.ref: %foo2.type = name_ref foo2, imports.%foo2.decl [concrete = constants.%foo2]
// CHECK:STDOUT:   %foo2.call: init %empty_tuple.type = call %foo2.ref()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace1.ref.loc10: <namespace> = name_ref my_namespace1, imports.%my_namespace1 [concrete = imports.%my_namespace1]
// CHECK:STDOUT:   %my_namespace2.ref.loc10: <namespace> = name_ref my_namespace2, imports.%my_namespace2 [concrete = imports.%my_namespace2]
// CHECK:STDOUT:   %my_namespace3.ref: <namespace> = name_ref my_namespace3, imports.%my_namespace3 [concrete = imports.%my_namespace3]
// CHECK:STDOUT:   %foo3.ref: %foo3.type = name_ref foo3, imports.%foo3.decl [concrete = constants.%foo3]
// CHECK:STDOUT:   %foo3.call: init %empty_tuple.type = call %foo3.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_special_name_call_escpaed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .r#base = %base
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %base: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %base.ref: <namespace> = name_ref r#base, imports.%base [concrete = imports.%base]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
