// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=ignore
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/base.carbon

// --- derived_to_base_conversion.h

class Base {};
class Derived : public Base {};

// --- use_derived_to_base_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "derived_to_base_conversion.h";

//@dump-sem-ir-begin
fn ConvertPtr(d: Cpp.Derived*) -> Cpp.Base* {
  return d;
}
//@dump-sem-ir-end

fn AcceptVal(b: Cpp.Base);

//@dump-sem-ir-begin
fn ConvertVal(d: Cpp.Derived) {
  AcceptVal(d);
}
//@dump-sem-ir-end

// --- static_member.h

class Base {
 public:
  static auto base_fn() -> void;
};

class Derived : public Base {
 public:
  static auto derived_fn() -> void;
};

// --- use_static_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_member.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.Base.base_fn();
  Cpp.Derived.base_fn();
  Cpp.Derived.derived_fn();
  //@dump-sem-ir-end
}

// --- base_field.h

struct Base {
  int a;
  int b;
};

struct Derived : Base {
  int b;
};

// --- use_base_field.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_field.h";

fn AccessDirect(d: Cpp.Derived) -> i32 {
  //@dump-sem-ir-begin
  return d.a;
  //@dump-sem-ir-end
}

fn AccessQualified(d: Cpp.Derived) -> i32 {
  //@dump-sem-ir-begin
  return d.(Cpp.Base.b);
  //@dump-sem-ir-end
}

// --- base_method.h

struct Base {
  void f() const;
  void g() const;
};

struct Derived : Base {
  void g() const;
};

// --- use_base_method.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_method.h";

fn CallDirect(d: Cpp.Derived) {
  //@dump-sem-ir-begin
  d.f();
  //@dump-sem-ir-end
}

fn CallQualified(d: Cpp.Derived) {
  //@dump-sem-ir-begin
  d.(Cpp.Base.g)();
  //@dump-sem-ir-end
}

// --- multiple_inheritance.h

struct A { int a; };
struct B { int b; };
struct C : A, B {};

// --- fail_todo_use_multiple_inheritance.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_inheritance.h";

fn ConvertA(p: Cpp.C*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.C*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.C*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertB(p: Cpp.C*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.C*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.C*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- virtual_inheritance.h

struct A { int a; };
struct B : virtual A {};

// --- fail_todo_use_virtual_inheritance.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+2]]:1: in import [InImport]
// CHECK:STDERR: ./virtual_inheritance.h:3: error: semantics TODO: `class with virtual bases` [SemanticsTodo]
import Cpp library "virtual_inheritance.h";

// CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+4]]:15: note: in `Cpp` name lookup for `B` [InCppNameLookup]
// CHECK:STDERR: fn Convert(p: Cpp.B*) -> Cpp.A* {
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR:
fn Convert(p: Cpp.B*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.B*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.B*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- use_derived_to_base_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %.52d: type = custom_layout_type {size=1, align=1, .base@0: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.fd1: <witness> = complete_type_witness %.52d [concrete]
// CHECK:STDOUT:   %ptr.ddb: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.5c8: type = pattern_type %ptr.ddb [concrete]
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.72a: type = pattern_type %ptr.fb2 [concrete]
// CHECK:STDOUT:   %ConvertPtr.type: type = fn_type @ConvertPtr [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ConvertPtr: %ConvertPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a3a: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %AcceptVal.type: type = fn_type @AcceptVal [concrete]
// CHECK:STDOUT:   %AcceptVal: %AcceptVal.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.5de: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %ConvertVal.type: type = fn_type @ConvertVal [concrete]
// CHECK:STDOUT:   %ConvertVal: %ConvertVal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .ConvertPtr = %ConvertPtr.decl
// CHECK:STDOUT:     .AcceptVal = %AcceptVal.decl
// CHECK:STDOUT:     .ConvertVal = %ConvertVal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "derived_to_base_conversion.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertPtr.decl: %ConvertPtr.type = fn_decl @ConvertPtr [concrete = constants.%ConvertPtr] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5c8 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5c8 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.72a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.72a = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref.loc7_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %ptr.loc7_43: type = ptr_type %Base.ref [concrete = constants.%ptr.fb2]
// CHECK:STDOUT:     %d.param: %ptr.ddb = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr.loc7_29 [concrete = constants.%ptr.ddb] {
// CHECK:STDOUT:       %Cpp.ref.loc7_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc7_29: type = ptr_type %Derived.ref [concrete = constants.%ptr.ddb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %ptr.ddb = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %ptr.fb2 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.fb2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AcceptVal.decl: %AcceptVal.type = fn_decl @AcceptVal [concrete = constants.%AcceptVal] {
// CHECK:STDOUT:     %b.patt: %pattern_type.a3a = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.a3a = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %Base = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %Base = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertVal.decl: %ConvertVal.type = fn_decl @ConvertVal [concrete = constants.%ConvertVal] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5de = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5de = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %.1: %Derived.elem = base_decl imports.%Base.decl, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=1, align=1, .base@0: %Base} [concrete = constants.%.52d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.fd1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertPtr(%d.param: %ptr.ddb) -> %ptr.fb2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %ptr.ddb = name_ref d, %d
// CHECK:STDOUT:   %.loc8_11.1: ref %Derived = deref %d.ref
// CHECK:STDOUT:   %.loc8_11.2: ref %Base = class_element_access %.loc8_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.fb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %.loc8_11.3: %ptr.fb2 = converted %d.ref, %addr
// CHECK:STDOUT:   return %.loc8_11.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AcceptVal(%b.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertVal(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AcceptVal.ref: %AcceptVal.type = name_ref AcceptVal, file.%AcceptVal.decl [concrete = constants.%AcceptVal]
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc16_13.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc16_13.2: ref %Base = converted %d.ref, %.loc16_13.1
// CHECK:STDOUT:   %.loc16_13.3: %Base = bind_value %.loc16_13.2
// CHECK:STDOUT:   %AcceptVal.call: init %empty_tuple.type = call %AcceptVal.ref(%.loc16_13.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_static_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Base.base_fn.type: type = fn_type @Base.base_fn [concrete]
// CHECK:STDOUT:   %Base.base_fn: %Base.base_fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %.52d: type = custom_layout_type {size=1, align=1, .base@0: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.fd1: <witness> = complete_type_witness %.52d [concrete]
// CHECK:STDOUT:   %Derived.derived_fn.type: type = fn_type @Derived.derived_fn [concrete]
// CHECK:STDOUT:   %Derived.derived_fn: %Derived.derived_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Base.base_fn.decl: %Base.base_fn.type = fn_decl @Base.base_fn [concrete = constants.%Base.base_fn] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Derived.derived_fn.decl: %Derived.derived_fn.type = fn_decl @Derived.derived_fn [concrete = constants.%Derived.derived_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "static_member.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .base_fn = imports.%Base.base_fn.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %.1: %Derived.elem = base_decl imports.%Base.decl, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=1, align=1, .base@0: %Base} [concrete = constants.%.52d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.fd1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base_fn = imports.%Base.base_fn.decl
// CHECK:STDOUT:   .derived_fn = imports.%Derived.derived_fn.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %base_fn.ref.loc8: %Base.base_fn.type = name_ref base_fn, imports.%Base.base_fn.decl [concrete = constants.%Base.base_fn]
// CHECK:STDOUT:   %Base.base_fn.call.loc8: init %empty_tuple.type = call %base_fn.ref.loc8()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Derived.ref.loc9: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %base_fn.ref.loc9: %Base.base_fn.type = name_ref base_fn, imports.%Base.base_fn.decl [concrete = constants.%Base.base_fn]
// CHECK:STDOUT:   %Base.base_fn.call.loc9: init %empty_tuple.type = call %base_fn.ref.loc9()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Derived.ref.loc10: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %derived_fn.ref: %Derived.derived_fn.type = name_ref derived_fn, imports.%Derived.derived_fn.decl [concrete = constants.%Derived.derived_fn]
// CHECK:STDOUT:   %Derived.derived_fn.call: init %empty_tuple.type = call %derived_fn.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.base_fn();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.derived_fn();
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %.807: type = custom_layout_type {size=8, align=4, .a@0: %i32, .b@4: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.e3e: <witness> = complete_type_witness %.807 [concrete]
// CHECK:STDOUT:   %Derived.elem.b4e: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.16d: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %.f26: type = custom_layout_type {size=12, align=4, .base@0: %Base, .b@8: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0bc: <witness> = complete_type_witness %.f26 [concrete]
// CHECK:STDOUT:   %pattern_type.5de: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %AccessDirect.type: type = fn_type @AccessDirect [concrete]
// CHECK:STDOUT:   %AccessDirect: %AccessDirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessQualified.type: type = fn_type @AccessQualified [concrete]
// CHECK:STDOUT:   %AccessQualified: %AccessQualified.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .AccessDirect = %AccessDirect.decl
// CHECK:STDOUT:     .AccessQualified = %AccessQualified.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "base_field.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDirect.decl: %AccessDirect.type = fn_decl @AccessDirect [concrete = constants.%AccessDirect] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5de = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5de = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessQualified.decl: %AccessQualified.type = fn_decl @AccessQualified [concrete = constants.%AccessQualified] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5de = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5de = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %.1: %Derived.elem.b4e = base_decl imports.%Base.decl, element0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %Derived.elem.16d = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %.3: type = custom_layout_type {size=12, align=4, .base@0: %Base, .b@8: %i32} [concrete = constants.%.f26]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.3 [concrete = constants.%complete_type.0bc]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .a = @Base.%.1
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %Base.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %.3: type = custom_layout_type {size=8, align=4, .a@0: %i32, .b@4: %i32} [concrete = constants.%.807]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.3 [concrete = constants.%complete_type.e3e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.2
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDirect(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.1 [concrete = @Base.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: ref %Base = converted %d.ref, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %i32 = class_element_access %.loc8_11.2, element0
// CHECK:STDOUT:   %.loc8_11.4: %i32 = bind_value %.loc8_11.3
// CHECK:STDOUT:   return %.loc8_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessQualified(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.2 [concrete = @Base.%.2]
// CHECK:STDOUT:   %.loc14_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: ref %Base = converted %d.ref, %.loc14_11.1
// CHECK:STDOUT:   %.loc14_11.3: ref %i32 = class_element_access %.loc14_11.2, element1
// CHECK:STDOUT:   %.loc14_11.4: %i32 = bind_value %.loc14_11.3
// CHECK:STDOUT:   return %.loc14_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %.52d: type = custom_layout_type {size=1, align=1, .base@0: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.fd1: <witness> = complete_type_witness %.52d [concrete]
// CHECK:STDOUT:   %pattern_type.5de: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %CallDirect.type: type = fn_type @CallDirect [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CallDirect: %CallDirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a3a: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.f.type: type = fn_type @Base.f [concrete]
// CHECK:STDOUT:   %Base.f: %Base.f.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallQualified.type: type = fn_type @CallQualified [concrete]
// CHECK:STDOUT:   %CallQualified: %CallQualified.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.g.type: type = fn_type @Base.g [concrete]
// CHECK:STDOUT:   %Base.g: %Base.g.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Base.f.decl: %Base.f.type = fn_decl @Base.f [concrete = constants.%Base.f] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.g.decl: %Base.g.type = fn_decl @Base.g [concrete = constants.%Base.g] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .CallDirect = %CallDirect.decl
// CHECK:STDOUT:     .CallQualified = %CallQualified.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "base_method.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallDirect.decl: %CallDirect.type = fn_decl @CallDirect [concrete = constants.%CallDirect] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5de = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5de = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallQualified.decl: %CallQualified.type = fn_decl @CallQualified [concrete = constants.%CallQualified] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5de = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5de = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %.1: %Derived.elem = base_decl imports.%Base.decl, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=1, align=1, .base@0: %Base} [concrete = constants.%.52d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.fd1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .f = imports.%Base.f.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .g = imports.%Base.g.decl
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDirect(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %f.ref: %Base.f.type = name_ref f, imports.%Base.f.decl [concrete = constants.%Base.f]
// CHECK:STDOUT:   %Base.f.bound: <bound method> = bound_method %d.ref, %f.ref
// CHECK:STDOUT:   %.loc8_3.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc8_3.2: ref %Base = converted %d.ref, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_3.3: %Base = bind_value %.loc8_3.2
// CHECK:STDOUT:   %Base.f.call: init %empty_tuple.type = call %Base.f.bound(%.loc8_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.f(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallQualified(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %g.ref: %Base.g.type = name_ref g, imports.%Base.g.decl [concrete = constants.%Base.g]
// CHECK:STDOUT:   %Base.g.bound: <bound method> = bound_method %d.ref, %g.ref
// CHECK:STDOUT:   %.loc14_3.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc14_3.2: ref %Base = converted %d.ref, %.loc14_3.1
// CHECK:STDOUT:   %.loc14_3.3: %Base = bind_value %.loc14_3.2
// CHECK:STDOUT:   %Base.g.call: init %empty_tuple.type = call %Base.g.bound(%.loc14_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.g(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_multiple_inheritance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %.79a: type = custom_layout_type {size=4, align=4, .a@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.33a: <witness> = complete_type_witness %.79a [concrete]
// CHECK:STDOUT:   %C.elem.1e5: type = unbound_element_type %C, %A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %i32 [concrete]
// CHECK:STDOUT:   %.903: type = custom_layout_type {size=4, align=4, .b@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fb9: <witness> = complete_type_witness %.903 [concrete]
// CHECK:STDOUT:   %C.elem.874: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %.dba: type = custom_layout_type {size=8, align=4, .base@0: %A, .base@4: %B} [concrete]
// CHECK:STDOUT:   %complete_type.1ad: <witness> = complete_type_witness %.dba [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.a31: type = pattern_type %ptr.d9e [concrete]
// CHECK:STDOUT:   %ptr.270: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.bcf: type = pattern_type %ptr.270 [concrete]
// CHECK:STDOUT:   %ConvertA.type: type = fn_type @ConvertA [concrete]
// CHECK:STDOUT:   %ConvertA: %ConvertA.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a04: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.837: type = pattern_type %ptr.a04 [concrete]
// CHECK:STDOUT:   %ConvertB.type: type = fn_type @ConvertB [concrete]
// CHECK:STDOUT:   %ConvertB: %ConvertB.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .ConvertA = %ConvertA.decl
// CHECK:STDOUT:     .ConvertB = %ConvertB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_inheritance.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertA.decl: %ConvertA.type = fn_decl @ConvertA [concrete = constants.%ConvertA] {
// CHECK:STDOUT:     %p.patt: %pattern_type.a31 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.a31 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.bcf = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.bcf = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref.loc6_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc6_32: type = ptr_type %A.ref [concrete = constants.%ptr.270]
// CHECK:STDOUT:     %p.param: %ptr.d9e = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr.loc6_21 [concrete = constants.%ptr.d9e] {
// CHECK:STDOUT:       %Cpp.ref.loc6_16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc6_21: type = ptr_type %C.ref [concrete = constants.%ptr.d9e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.d9e = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.270 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.270 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertB.decl: %ConvertB.type = fn_decl @ConvertB [concrete = constants.%ConvertB] {
// CHECK:STDOUT:     %p.patt: %pattern_type.a31 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.a31 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.837 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.837 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref.loc17_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc17_32: type = ptr_type %B.ref [concrete = constants.%ptr.a04]
// CHECK:STDOUT:     %p.param: %ptr.d9e = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %ptr.loc17_21 [concrete = constants.%ptr.d9e] {
// CHECK:STDOUT:       %Cpp.ref.loc17_16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc17_21: type = ptr_type %C.ref [concrete = constants.%ptr.d9e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.d9e = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.a04 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.a04 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.1: %C.elem.1e5 = base_decl imports.%A.decl, element0 [concrete]
// CHECK:STDOUT:   %.2: %C.elem.874 = base_decl imports.%B.decl, element1 [concrete]
// CHECK:STDOUT:   %.3: type = custom_layout_type {size=8, align=4, .base@0: %A, .base@4: %B} [concrete = constants.%.dba]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.3 [concrete = constants.%complete_type.1ad]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %A.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=4, align=4, .a@0: %i32} [concrete = constants.%.79a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.33a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %B.elem = field_decl b, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=4, align=4, .b@0: %i32} [concrete = constants.%.903]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.fb9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertA(%p.param: %ptr.d9e) -> %ptr.270 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.d9e = name_ref p, %p
// CHECK:STDOUT:   %.loc14: %ptr.270 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertB(%p.param: %ptr.d9e) -> %ptr.a04 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.d9e = name_ref p, %p
// CHECK:STDOUT:   %.loc25: %ptr.a04 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_virtual_inheritance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.a04: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.837: type = pattern_type %ptr.a04 [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %.79a: type = custom_layout_type {size=4, align=4, .a@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.33a: <witness> = complete_type_witness %.79a [concrete]
// CHECK:STDOUT:   %ptr.270: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.bcf: type = pattern_type %ptr.270 [concrete]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert [concrete]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .Convert = %Convert.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "virtual_inheritance.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Convert.decl: %Convert.type = fn_decl @Convert [concrete = constants.%Convert] {
// CHECK:STDOUT:     %p.patt: %pattern_type.837 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.837 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.bcf = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.bcf = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref.loc12_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc12_31: type = ptr_type %A.ref [concrete = constants.%ptr.270]
// CHECK:STDOUT:     %p.param: %ptr.a04 = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %ptr.loc12_20 [concrete = constants.%ptr.a04] {
// CHECK:STDOUT:       %Cpp.ref.loc12_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %B.ref: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc12_20: type = ptr_type %B.ref [concrete = constants.%ptr.a04]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.a04 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.270 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.270 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %A.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=4, align=4, .a@0: %i32} [concrete = constants.%.79a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.33a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%p.param: %ptr.a04) -> %ptr.270 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.a04 = name_ref p, %p
// CHECK:STDOUT:   %.loc20: %ptr.270 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
