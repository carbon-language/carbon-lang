// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/no_prelude/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/no_prelude/class.carbon

// --- class_declaration.h

class Bar;

// --- fail_import_class_declaration.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_declaration.h";

// CHECK:STDERR: fail_import_class_declaration.carbon:[[@LINE+7]]:13: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_import_class_declaration.carbon:[[@LINE+4]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);

// --- class_definition.h

class Bar {};

// --- import_class_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_definition.h";

fn MyF(bar: Cpp.Bar*);

// --- class_declaration_and_definition.h

class Bar;
class Bar {};

// --- import_class_declaration_and_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_declaration_and_definition.h";

fn MyF(bar: Cpp.Bar*);

// --- class_public_static_member_function.h

class Bar {
 public:
  static auto foo() -> void;
};

// --- import_class_public_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_public_static_member_function.h";

fn MyF() {
  Cpp.Bar.foo();
}

// --- class_private_static_member_function.h

class Bar {
 private:
  static auto foo() -> void;
};

// --- todo_fail_import_class_private_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_private_static_member_function.h";

fn MyF() {
  Cpp.Bar.foo();
}

// --- class_public_member_function.h

class Bar {
 public:
  auto foo() -> void;
};

// --- fail_import_class_public_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_public_member_function.h";

fn MyF(bar : Cpp.Bar*) {
  // CHECK:STDERR: fail_import_class_public_member_function.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Non-global function` [SemanticsTodo]
  // CHECK:STDERR:   bar->foo();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_import_class_public_member_function.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   bar->foo();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  bar->foo();
}

// --- class_public_static_data_member.h

class Bar {
 public:
  static Bar* foo;
};

// --- fail_import_class_public_static_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_public_static_data_member.h";

fn MyF() {
  // CHECK:STDERR: fail_import_class_public_static_data_member.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_class_public_static_data_member.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_class_public_static_data_member.carbon:[[@LINE+4]]:23: error: member name `foo` not found in `Cpp.Bar` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR:
  let bar: Cpp.Bar* = Cpp.Bar.foo();
}

// --- class_public_data_member.h

class Bar {
 public:
  Bar* foo;
};

// --- fail_import_class_public_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_public_static_data_member.h";

fn MyF(bar : Cpp.Bar*) {
  // CHECK:STDERR: fail_import_class_public_data_member.carbon:[[@LINE+11]]:27: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR: fail_import_class_public_data_member.carbon:[[@LINE+8]]:27: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_class_public_data_member.carbon:[[@LINE+4]]:27: error: member name `foo` not found in `Cpp.Bar` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  let foo_bar: Cpp.Bar* = bar->foo;
}

// --- class_inheritance_static.h

class Bar1 {
 public:
  static auto foo1() -> void;
};

class Bar2 : public Bar1 {
 public:
  static auto foo2() -> void;
};

// --- import_class_inheritance_static.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_inheritance_static.h";

fn MyF() {
  Cpp.Bar1.foo1();
  Cpp.Bar2.foo1();
  Cpp.Bar2.foo2();
}

// --- class_inheritance_pointers.h

class Bar1 {};
class Bar2 : public Bar1 {};

// --- fail_import_class_inheritance_pointers.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_inheritance_pointers.h";

fn MyF1(bar : Cpp.Bar1*);
// TODO: Support C++ inheritance.
// CHECK:STDERR: fail_import_class_inheritance_pointers.carbon:[[@LINE+7]]:33: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn MyF2(bar : Cpp.Bar2*) { MyF1(bar); }
// CHECK:STDERR:                                 ^~~
// CHECK:STDERR: fail_import_class_inheritance_pointers.carbon:[[@LINE-5]]:9: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn MyF1(bar : Cpp.Bar1*);
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MyF2(bar : Cpp.Bar2*) { MyF1(bar); }

// --- class_dynamic.h

class Bar {
 public:
  virtual ~Bar();
};

// --- fail_import_class_dynamic.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_dynamic.h";

// CHECK:STDERR: fail_import_class_dynamic.carbon:[[@LINE+7]]:14: error: semantics TODO: `Unsupported: Dynamic Class` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar : Cpp.Bar*);
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR: fail_import_class_dynamic.carbon:[[@LINE+4]]:14: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar : Cpp.Bar*);
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
fn MyF(bar : Cpp.Bar*);

// --- class_to_inherit_public.h

class Bar {
 public:
  static auto foo() -> void;
};

// --- import_class_to_inherit_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_to_inherit_public.h";

class Derived {
  extend base: Cpp.Bar;
}

fn MyF() {
  Derived.foo();
}

// --- class_to_inherit_private.h

class Bar {
 private:
  static auto foo() -> void;
};

// --- todo_fail_class_to_inherit_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_to_inherit_private.h";

class Derived {
  extend base: Cpp.Bar;
}

fn MyF() {
  Derived.foo();
}

// --- class_template.h

template<typename T>
class Bar {};

// --- fail_import_class_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_template.h";

// CHECK:STDERR: fail_import_class_template.carbon:[[@LINE+11]]:13: error: semantics TODO: `Unsupported: Declaration type ClassTemplate` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_import_class_template.carbon:[[@LINE+8]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_class_template.carbon:[[@LINE+4]]:13: error: member name `Bar` not found in `Cpp` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);

// CHECK:STDOUT: --- fail_import_class_declaration.carbon
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
// CHECK:STDOUT:     import Cpp "class_declaration.h"
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
// CHECK:STDOUT: --- import_class_definition.carbon
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
// CHECK:STDOUT:     import Cpp "class_definition.h"
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
// CHECK:STDOUT: --- import_class_declaration_and_definition.carbon
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
// CHECK:STDOUT:     import Cpp "class_declaration_and_definition.h"
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
// CHECK:STDOUT: --- import_class_public_static_member_function.carbon
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
// CHECK:STDOUT:     import Cpp "class_public_static_member_function.h"
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
// CHECK:STDOUT: --- todo_fail_import_class_private_static_member_function.carbon
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
// CHECK:STDOUT:     import Cpp "class_private_static_member_function.h"
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
// CHECK:STDOUT: --- fail_import_class_public_member_function.carbon
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
// CHECK:STDOUT:     import Cpp "class_public_member_function.h"
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
// CHECK:STDOUT: --- fail_import_class_public_static_data_member.carbon
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
// CHECK:STDOUT:     import Cpp "class_public_static_data_member.h"
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
// CHECK:STDOUT: --- fail_import_class_public_data_member.carbon
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
// CHECK:STDOUT:     import Cpp "class_public_static_data_member.h"
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
// CHECK:STDOUT: --- import_class_inheritance_static.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar1: type = class_type @Bar1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %foo1.type.148: type = fn_type @foo1.1 [concrete]
// CHECK:STDOUT:   %foo1.8cd: %foo1.type.148 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar2: type = class_type @Bar2 [concrete]
// CHECK:STDOUT:   %foo1.type.0b8: type = fn_type @foo1.2 [concrete]
// CHECK:STDOUT:   %foo1.ba2: %foo1.type.0b8 = struct_value () [concrete]
// CHECK:STDOUT:   %foo2.type: type = fn_type @foo2 [concrete]
// CHECK:STDOUT:   %foo2: %foo2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar1 = @MyF.%Bar1.decl
// CHECK:STDOUT:     .Bar2 = @MyF.%Bar2.decl
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
// CHECK:STDOUT:     import Cpp "class_inheritance_static.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar1 {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar1
// CHECK:STDOUT:   .foo1 = @MyF.%foo1.decl.1
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar2 {
// CHECK:STDOUT:   complete_type_witness = @MyF.%complete_type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar2
// CHECK:STDOUT:   .foo1 = @MyF.%foo1.decl.2
// CHECK:STDOUT:   .foo2 = @MyF.%foo2.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar1.decl: type = class_decl @Bar1 [concrete = constants.%Bar1] {} {}
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Bar1.ref: type = name_ref Bar1, %Bar1.decl [concrete = constants.%Bar1]
// CHECK:STDOUT:   %foo1.decl.1: %foo1.type.148 = fn_decl @foo1.1 [concrete = constants.%foo1.8cd] {} {}
// CHECK:STDOUT:   %foo1.ref.loc7: %foo1.type.148 = name_ref foo1, %foo1.decl.1 [concrete = constants.%foo1.8cd]
// CHECK:STDOUT:   %foo1.call.loc7: init %empty_tuple.type = call %foo1.ref.loc7()
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar2.decl: type = class_decl @Bar2 [concrete = constants.%Bar2] {} {}
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Bar2.ref.loc8: type = name_ref Bar2, %Bar2.decl [concrete = constants.%Bar2]
// CHECK:STDOUT:   %foo1.decl.2: %foo1.type.0b8 = fn_decl @foo1.2 [concrete = constants.%foo1.ba2] {} {}
// CHECK:STDOUT:   %foo1.ref.loc8: %foo1.type.0b8 = name_ref foo1, %foo1.decl.2 [concrete = constants.%foo1.ba2]
// CHECK:STDOUT:   %foo1.call.loc8: init %empty_tuple.type = call %foo1.ref.loc8()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar2.ref.loc9: type = name_ref Bar2, %Bar2.decl [concrete = constants.%Bar2]
// CHECK:STDOUT:   %foo2.decl: %foo2.type = fn_decl @foo2 [concrete = constants.%foo2] {} {}
// CHECK:STDOUT:   %foo2.ref: %foo2.type = name_ref foo2, %foo2.decl [concrete = constants.%foo2]
// CHECK:STDOUT:   %foo2.call: init %empty_tuple.type = call %foo2.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo1.1[]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo1.2[]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo2[]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_class_inheritance_pointers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar1: type = class_type @Bar1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar1 [concrete]
// CHECK:STDOUT:   %MyF1.type: type = fn_type @MyF1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyF1: %MyF1.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar2: type = class_type @Bar2 [concrete]
// CHECK:STDOUT:   %ptr.eca: type = ptr_type %Bar2 [concrete]
// CHECK:STDOUT:   %MyF2.type: type = fn_type @MyF2 [concrete]
// CHECK:STDOUT:   %MyF2: %MyF2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar1 = @MyF1.%Bar1.decl
// CHECK:STDOUT:     .Bar2 = @MyF2.%Bar2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF1 = %MyF1.decl
// CHECK:STDOUT:     .MyF2 = %MyF2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "class_inheritance_pointers.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF1.decl: %MyF1.type = fn_decl @MyF1 [concrete = constants.%MyF1] {
// CHECK:STDOUT:     %bar.patt: %ptr.f68 = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: %ptr.f68 = value_param_pattern %bar.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.f68 = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar1.decl: type = class_decl @Bar1 [concrete = constants.%Bar1] {} {}
// CHECK:STDOUT:       %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:       %Bar1.ref: type = name_ref Bar1, %Bar1.decl [concrete = constants.%Bar1]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar1.ref [concrete = constants.%ptr.f68]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.f68 = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF2.decl: %MyF2.type = fn_decl @MyF2 [concrete = constants.%MyF2] {
// CHECK:STDOUT:     %bar.patt: %ptr.eca = binding_pattern bar
// CHECK:STDOUT:     %bar.param_patt: %ptr.eca = value_param_pattern %bar.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.eca = value_param call_param0
// CHECK:STDOUT:     %.loc15_23: type = splice_block %ptr [concrete = constants.%ptr.eca] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar2.decl: type = class_decl @Bar2 [concrete = constants.%Bar2] {} {}
// CHECK:STDOUT:       %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:       %Bar2.ref: type = name_ref Bar2, %Bar2.decl [concrete = constants.%Bar2]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar2.ref [concrete = constants.%ptr.eca]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.eca = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar1 {
// CHECK:STDOUT:   complete_type_witness = @MyF1.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar1
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bar2 {
// CHECK:STDOUT:   complete_type_witness = @MyF2.%complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bar2
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF1(%bar.param_patt: %ptr.f68);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF2(%bar.param_patt: %ptr.eca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MyF1.ref: %MyF1.type = name_ref MyF1, file.%MyF1.decl [concrete = constants.%MyF1]
// CHECK:STDOUT:   %bar.ref: %ptr.eca = name_ref bar, %bar
// CHECK:STDOUT:   %.loc15_33: %ptr.f68 = converted %bar.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %MyF1.call: init %empty_tuple.type = call %MyF1.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_class_dynamic.carbon
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
// CHECK:STDOUT:     import Cpp "class_dynamic.h"
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
// CHECK:STDOUT: --- import_class_to_inherit_public.carbon
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
// CHECK:STDOUT:     import Cpp "class_to_inherit_public.h"
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
// CHECK:STDOUT: --- todo_fail_class_to_inherit_private.carbon
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
// CHECK:STDOUT:     import Cpp "class_to_inherit_private.h"
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
// CHECK:STDOUT: --- fail_import_class_template.carbon
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
// CHECK:STDOUT:     import Cpp "class_template.h"
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
