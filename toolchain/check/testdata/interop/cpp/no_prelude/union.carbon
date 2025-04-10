// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/no_prelude/union.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/no_prelude/union.carbon

// --- union_declaration.h

union Bar;

// --- fail_import_union_declaration.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_declaration.h";

// CHECK:STDERR: fail_import_union_declaration.carbon:[[@LINE+7]]:13: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_import_union_declaration.carbon:[[@LINE+4]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);

// --- union_definition.h

union Bar {};

// --- import_union_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_definition.h";

fn MyF(bar: Cpp.Bar*);

// --- union_declaration_and_definition.h

union Bar;
union Bar {};

// --- import_union_declaration_and_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_declaration_and_definition.h";

fn MyF(bar: Cpp.Bar*);

// --- union_public_static_member_function.h

union Bar {
 public:
  static auto foo() -> void;
};

// --- import_union_public_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_public_static_member_function.h";

fn MyF() {
  Cpp.Bar.foo();
}

// --- union_private_static_member_function.h

union Bar {
 private:
  static auto foo() -> void;
};

// --- todo_fail_import_union_private_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_private_static_member_function.h";

fn MyF() {
  Cpp.Bar.foo();
}

// --- union_public_member_function.h

union Bar {
 public:
  auto foo() -> void;
};

// --- fail_import_union_public_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_public_member_function.h";

fn MyF(bar : Cpp.Bar*) {
  // CHECK:STDERR: fail_import_union_public_member_function.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Non-global function` [SemanticsTodo]
  // CHECK:STDERR:   bar->foo();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_import_union_public_member_function.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   bar->foo();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  bar->foo();
}

// --- union_public_static_data_member.h

union Bar {
 public:
  static Bar* foo;
};

// --- fail_import_union_public_static_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_public_static_data_member.h";

fn MyF() {
  // CHECK:STDERR: fail_import_union_public_static_data_member.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_union_public_static_data_member.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_union_public_static_data_member.carbon:[[@LINE+4]]:23: error: member name `foo` not found in `Cpp.Bar` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR:
  let bar: Cpp.Bar* = Cpp.Bar.foo();
}

// --- union_public_data_member.h

union Bar {
 public:
  Bar* foo;
};

// --- fail_import_union_public_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_public_static_data_member.h";

fn MyF(bar : Cpp.Bar*) {
  // CHECK:STDERR: fail_import_union_public_data_member.carbon:[[@LINE+11]]:27: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR: fail_import_union_public_data_member.carbon:[[@LINE+8]]:27: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_union_public_data_member.carbon:[[@LINE+4]]:27: error: member name `foo` not found in `Cpp.Bar` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  let foo_bar: Cpp.Bar* = bar->foo;
}

// --- union_to_inherit_public.h

union Bar {
 public:
  static auto foo() -> void;
};

// --- todo_fail_import_union_to_inherit_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_to_inherit_public.h";

class Derived {
  extend base: Cpp.Bar;
}

fn MyF() {
  Derived.foo();
}

// --- union_template.h

template<typename T>
union Bar {};

// --- fail_import_union_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "union_template.h";

// CHECK:STDERR: fail_import_union_template.carbon:[[@LINE+11]]:13: error: semantics TODO: `Unsupported: Declaration type ClassTemplate` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_import_union_template.carbon:[[@LINE+8]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_union_template.carbon:[[@LINE+4]]:13: error: member name `Bar` not found in `Cpp` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);

// CHECK:STDOUT: --- fail_import_union_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_declaration.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: <error> = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: <error> = value_param_pattern %bar.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.ref: <error> = name_ref Bar, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: <error> = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_union_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_definition.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %ptr = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: %ptr = value_param_pattern %bar.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:       %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_union_declaration_and_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_declaration_and_definition.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %ptr = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: %ptr = value_param_pattern %bar.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:       %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_union_public_static_member_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_public_static_member_function.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   .foo = @MyF.%foo.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, %foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo[]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_union_private_static_member_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_private_static_member_function.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   .foo = @MyF.%foo.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, %foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo[]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_union_public_member_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_public_member_function.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %ptr.f68 = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: %ptr.f68 = value_param_pattern %bar.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.f68 = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:       %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr.f68]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.f68 = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   .foo = <error>
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param_patt: %ptr.f68) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %bar.ref: %ptr.f68 = name_ref bar, %bar
// CHECK:STDOUT:   %.loc14: ref %Bar = deref %bar.ref
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_union_public_static_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_public_static_data_member.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   .foo = <poisoned>
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bar.patt: %ptr.f68 = binding_pattern bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.ref.loc18_26: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc18: type = splice_block %ptr [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:     %Cpp.ref.loc18_12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     %Bar.ref.loc18_15: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:     %ptr: type = ptr_type %Bar.ref.loc18_15 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar: %ptr.f68 = bind_name bar, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_union_public_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @MyF.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_public_static_data_member.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %ptr.f68 = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: %ptr.f68 = value_param_pattern %bar.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.f68 = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr.loc6 [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:       %Cpp.ref.loc6: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:       %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:       %Bar.ref.loc6: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr.loc6: type = ptr_type %Bar.ref.loc6 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.f68 = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   .foo = <poisoned>
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param_patt: %ptr.f68) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %foo_bar.patt: %ptr.f68 = binding_pattern foo_bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar.ref: %ptr.f68 = name_ref bar, %bar
// CHECK:STDOUT:   %.loc18_30: ref %Bar = deref %bar.ref
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc18_23: type = splice_block %ptr.loc18 [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Bar.ref.loc18: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %Bar.ref.loc18 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_bar: %ptr.f68 = bind_name foo_bar, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_union_to_inherit_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Bar [concrete]
// CHECK:STDOUT:   %struct_type.base.36d: type = struct_type {.base: %Bar} [concrete]
// CHECK:STDOUT:   %complete_type.fff: <witness> = complete_type_witness %struct_type.base.36d [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = @Derived.%Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_to_inherit_public.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, %Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %.loc7: %Derived.elem = base_decl %Bar.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type.loc8: <witness> = complete_type_witness %struct_type.base.36d [concrete = constants.%complete_type.fff]
// CHECK:STDOUT:   complete_type_witness = %complete_type.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .foo = <poisoned>
// CHECK:STDOUT:   extend %Bar.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar {
// CHECK:STDOUT:   complete_type_witness = @Derived.%complete_type.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar
// CHECK:STDOUT:   .foo = @MyF.%foo.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, %foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo[]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_union_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = <poisoned>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union_template.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: <error> = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: <error> = value_param_pattern %bar.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.ref: <error> = name_ref Bar, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: <error> = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param_patt: <error>);
// CHECK:STDOUT:
