// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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

struct Empty1 {};
struct Empty2 {};
struct OneNonEmptyBase : Empty1, A, Empty2 {};
struct TwoNonEmptyBases : Empty1, A, B, Empty2 {};

struct Polymorphic1 {
  virtual void f();
};
struct Polymorphic2 {
  virtual void g();
};
struct OnePolymorphicBase : A, B, Polymorphic1, Empty1 {};
struct TwoPolymorphicBases : Polymorphic1, Polymorphic2 {};

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

fn ConvertOneNonEmptyBaseToEmpty1(p: Cpp.OneNonEmptyBase*) -> Cpp.Empty1* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OneNonEmptyBase*` to `Cpp.Empty1*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OneNonEmptyBase*` does not implement interface `Core.ImplicitAs(Cpp.Empty1*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOneNonEmptyBaseToEmpty2(p: Cpp.OneNonEmptyBase*) -> Cpp.Empty2* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OneNonEmptyBase*` to `Cpp.Empty2*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OneNonEmptyBase*` does not implement interface `Core.ImplicitAs(Cpp.Empty2*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoNonEmptyBasesToA(p: Cpp.TwoNonEmptyBases*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoNonEmptyBases*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoNonEmptyBases*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoNonEmptyBasesToB(p: Cpp.TwoNonEmptyBases*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoNonEmptyBases*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoNonEmptyBases*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOnePolymorphicBaseToA(p: Cpp.OnePolymorphicBase*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OnePolymorphicBase*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OnePolymorphicBase*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOnePolymorphicBaseToB(p: Cpp.OnePolymorphicBase*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OnePolymorphicBase*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OnePolymorphicBase*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoPolymorphicBasesToPolymorphic1(p: Cpp.TwoPolymorphicBases*) -> Cpp.Polymorphic1* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoPolymorphicBases*` to `Cpp.Polymorphic1*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoPolymorphicBases*` does not implement interface `Core.ImplicitAs(Cpp.Polymorphic1*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoPolymorphicBasesToPolymorphic2(p: Cpp.TwoPolymorphicBases*) -> Cpp.Polymorphic2* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoPolymorphicBases*` to `Cpp.Polymorphic2*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoPolymorphicBases*` does not implement interface `Core.ImplicitAs(Cpp.Polymorphic2*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- multiple_inheritance_with_preferred_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_inheritance.h";

// If there's only one non-empty base, that's our preferred base class. We can
// convert to that.
fn ConvertOneNonEmptyBaseToA(p: Cpp.OneNonEmptyBase*) -> Cpp.A* {
  return p;
}

// If there's only one polymorphic base, that's our preferred base class. We can
// convert to that.
fn ConvertOnePolymorphicBaseToPolymorphic1(p: Cpp.OnePolymorphicBase*) -> Cpp.Polymorphic1* {
  return p;
}

// --- virtual_inheritance.h

struct A { int a; };
struct B : virtual A {};

void UseB(B * _Nonnull p);

// --- use_virtual_inheritance_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "virtual_inheritance.h";

fn Convert(p: Cpp.B*) {
  // OK, doesn't require `Cpp.B` to be a complete type.
  Cpp.UseB(p);
}

// --- fail_todo_use_virtual_inheritance.carbon

library "[[@TEST_NAME]]";

import Cpp library "virtual_inheritance.h";

fn Convert(p: Cpp.B*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+21]]:3: error: semantics TODO: `class with virtual bases` [SemanticsTodo]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+18]]:3: note: while completing C++ type `Cpp.B` [InCppTypeCompletion]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+14]]:3: error: semantics TODO: `class with virtual bases` [SemanticsTodo]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+11]]:3: note: while completing C++ type `Cpp.B` [InCppTypeCompletion]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.B*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.B*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- final.h

struct A final {};

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

import Cpp library "final.h";

class B {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:16: error: deriving from final type `A`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Cpp.A;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: Cpp.A;
}

// --- union.h

union U {};

// --- fail_derive_from_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

class V {
  // CHECK:STDERR: fail_derive_from_union.carbon:[[@LINE+4]]:16: error: deriving from final type `U`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Cpp.U;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: Cpp.U;
}

// CHECK:STDOUT: --- use_derived_to_base_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ptr.ddb: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.5c8: type = pattern_type %ptr.ddb [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.72a: type = pattern_type %ptr.fb2 [concrete]
// CHECK:STDOUT:   %ConvertPtr.type: type = fn_type @ConvertPtr [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ConvertPtr: %ConvertPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fe1: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.07b: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.fb2, (%Copy.impl_witness.fe1) [concrete]
// CHECK:STDOUT:   %.e43: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.07b, @ptr.as.Copy.impl.Op(%Base) [concrete]
// CHECK:STDOUT:   %AcceptVal.type: type = fn_type @AcceptVal [concrete]
// CHECK:STDOUT:   %AcceptVal: %AcceptVal.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.5de: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %ConvertVal.type: type = fn_type @ConvertVal [concrete]
// CHECK:STDOUT:   %ConvertVal: %ConvertVal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
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
// CHECK:STDOUT: fn @ConvertPtr(%d.param: %ptr.ddb) -> %ptr.fb2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %ptr.ddb = name_ref d, %d
// CHECK:STDOUT:   %.loc8_11.1: ref %Derived = deref %d.ref
// CHECK:STDOUT:   %.loc8_11.2: ref %Base = class_element_access %.loc8_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.fb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %.loc8_11.3: %ptr.fb2 = converted %d.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.e43 = impl_witness_access constants.%Copy.impl_witness.fe1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.07b]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Base) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.fb2 = call %bound_method.loc8_11.2(%.loc8_11.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %.75e: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.c9f: %.75e = struct_value () [concrete]
// CHECK:STDOUT:   %Base.base_fn.type: type = fn_type @Base.base_fn [concrete]
// CHECK:STDOUT:   %Base.base_fn: %Base.base_fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %.809: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.422: %.809 = struct_value () [concrete]
// CHECK:STDOUT:   %.5f9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.510: %.5f9 = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.derived_fn.type: type = fn_type @Derived.derived_fn [concrete]
// CHECK:STDOUT:   %Derived.derived_fn: %Derived.derived_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %.750: %.75e = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.c9f]
// CHECK:STDOUT:   %Base.base_fn.decl: %Base.base_fn.type = fn_decl @Base.base_fn [concrete = constants.%Base.base_fn] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %.22a: %.809 = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.422]
// CHECK:STDOUT:   %.24a: %.5f9 = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.510]
// CHECK:STDOUT:   %Derived.derived_fn.decl: %Derived.derived_fn.type = fn_decl @Derived.derived_fn [concrete = constants.%Derived.derived_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %base_fn.ref.loc8: %.75e = name_ref base_fn, imports.%.750 [concrete = constants.%empty_struct.c9f]
// CHECK:STDOUT:   %Base.base_fn.call.loc8: init %empty_tuple.type = call imports.%Base.base_fn.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Derived.ref.loc9: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %base_fn.ref.loc9: %.809 = name_ref base_fn, imports.%.22a [concrete = constants.%empty_struct.422]
// CHECK:STDOUT:   %Base.base_fn.call.loc9: init %empty_tuple.type = call imports.%Base.base_fn.decl()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Derived.ref.loc10: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %derived_fn.ref: %.5f9 = name_ref derived_fn, imports.%.24a [concrete = constants.%empty_struct.510]
// CHECK:STDOUT:   %Derived.derived_fn.call: init %empty_tuple.type = call imports.%Derived.derived_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.4, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc8_11.2(%.loc8_11.4)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
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
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.4, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.4)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %.5b0: type = cpp_overload_set_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.3f3: %.5b0 = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %Base [concrete]
// CHECK:STDOUT:   %ptr.a97: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %.7c6: type = cpp_overload_set_type @CallQualified [concrete]
// CHECK:STDOUT:   %empty_struct.49e: %.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %g__carbon_thunk.type: type = fn_type @g__carbon_thunk [concrete]
// CHECK:STDOUT:   %g__carbon_thunk: %g__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %.e54: %.5b0 = cpp_overload_set_value @f__carbon_thunk [concrete = constants.%empty_struct.3f3]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.362: %.7c6 = cpp_overload_set_value @CallQualified [concrete = constants.%empty_struct.49e]
// CHECK:STDOUT:   %g__carbon_thunk.decl: %g__carbon_thunk.type = fn_decl @g__carbon_thunk [concrete = constants.%g__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDirect(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %f.ref: %.5b0 = name_ref f, imports.%.e54 [concrete = constants.%empty_struct.3f3]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %f.ref
// CHECK:STDOUT:   %.loc8_3.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc8_3.2: ref %Base = converted %d.ref, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_3.3: %Base = bind_value %.loc8_3.2
// CHECK:STDOUT:   %.loc8_3.4: ref %Base = value_as_ref %.loc8_3.3
// CHECK:STDOUT:   %addr: %ptr.fb2 = addr_of %.loc8_3.4
// CHECK:STDOUT:   %.loc8_7.1: %ptr.a97 = as_compatible %addr
// CHECK:STDOUT:   %.loc8_7.2: %ptr.a97 = converted %addr, %.loc8_7.1
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%.loc8_7.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallQualified(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %g.ref: %.7c6 = name_ref g, imports.%.362 [concrete = constants.%empty_struct.49e]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %g.ref
// CHECK:STDOUT:   %.loc14_3.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc14_3.2: ref %Base = converted %d.ref, %.loc14_3.1
// CHECK:STDOUT:   %.loc14_3.3: %Base = bind_value %.loc14_3.2
// CHECK:STDOUT:   %.loc14_3.4: ref %Base = value_as_ref %.loc14_3.3
// CHECK:STDOUT:   %addr: %ptr.fb2 = addr_of %.loc14_3.4
// CHECK:STDOUT:   %.loc14_18.1: %ptr.a97 = as_compatible %addr
// CHECK:STDOUT:   %.loc14_18.2: %ptr.a97 = converted %addr, %.loc14_18.1
// CHECK:STDOUT:   %g__carbon_thunk.call: init %empty_tuple.type = call imports.%g__carbon_thunk.decl(%.loc14_18.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
