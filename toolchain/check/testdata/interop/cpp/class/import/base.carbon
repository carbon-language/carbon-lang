// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/base.carbon

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

class StaticBase {
 public:
  static auto base_fn() -> void;
};

class StaticDerived : public StaticBase {
 public:
  static auto derived_fn() -> void;
};

// --- use_static_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_member.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.StaticBase.base_fn();
  Cpp.StaticDerived.base_fn();
  Cpp.StaticDerived.derived_fn();
  //@dump-sem-ir-end
}

// --- base_field.h

struct FieldBase {
  int a;
  int b;
};

struct FieldDerived : FieldBase {
  int b;
};

// --- use_base_field.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_field.h";

fn AccessDirect(d: Cpp.FieldDerived) -> i32 {
  //@dump-sem-ir-begin
  return d.a;
  //@dump-sem-ir-end
}

fn AccessQualified(d: Cpp.FieldDerived) -> i32 {
  //@dump-sem-ir-begin
  return d.(Cpp.FieldBase.b);
  //@dump-sem-ir-end
}

// --- base_method.h

struct MethodBase {
  void f() const;
  void g() const;
};

struct MethodDerived : MethodBase {
  void g() const;
};

// --- use_base_method.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_method.h";

fn CallDirect(d: Cpp.MethodDerived) {
  //@dump-sem-ir-begin
  d.f();
  //@dump-sem-ir-end
}

fn CallQualified(d: Cpp.MethodDerived) {
  //@dump-sem-ir-begin
  d.(Cpp.MethodBase.g)();
  //@dump-sem-ir-end
}

fn CallBaseQualifiedByDerived(d: Cpp.MethodDerived, b: Cpp.MethodBase) {
  //@dump-sem-ir-begin
  // `Cpp.Derived.f` names a member of `Cpp.Base`, so we allow that to be used
  // when `self` is a `Cpp.Base` or anything that implicitly converts to
  // `Cpp.Base`, such as `Cpp.Derived`.
  d.(Cpp.MethodDerived.f)();
  b.(Cpp.MethodDerived.f)();
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
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.C*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertB(p: Cpp.C*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.C*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.C*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOneNonEmptyBaseToEmpty1(p: Cpp.OneNonEmptyBase*) -> Cpp.Empty1* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OneNonEmptyBase*` to `Cpp.Empty1*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OneNonEmptyBase*` does not implement interface `Core.ImplicitAs(Cpp.Empty1*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOneNonEmptyBaseToEmpty2(p: Cpp.OneNonEmptyBase*) -> Cpp.Empty2* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OneNonEmptyBase*` to `Cpp.Empty2*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OneNonEmptyBase*` does not implement interface `Core.ImplicitAs(Cpp.Empty2*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoNonEmptyBasesToA(p: Cpp.TwoNonEmptyBases*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoNonEmptyBases*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoNonEmptyBases*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoNonEmptyBasesToB(p: Cpp.TwoNonEmptyBases*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoNonEmptyBases*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoNonEmptyBases*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOnePolymorphicBaseToA(p: Cpp.OnePolymorphicBase*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OnePolymorphicBase*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OnePolymorphicBase*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertOnePolymorphicBaseToB(p: Cpp.OnePolymorphicBase*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.OnePolymorphicBase*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.OnePolymorphicBase*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoPolymorphicBasesToPolymorphic1(p: Cpp.TwoPolymorphicBases*) -> Cpp.Polymorphic1* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoPolymorphicBases*` to `Cpp.Polymorphic1*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoPolymorphicBases*` does not implement interface `Core.ImplicitAs(Cpp.Polymorphic1*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertTwoPolymorphicBasesToPolymorphic2(p: Cpp.TwoPolymorphicBases*) -> Cpp.Polymorphic2* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.TwoPolymorphicBases*` to `Cpp.Polymorphic2*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.TwoPolymorphicBases*` does not implement interface `Core.ImplicitAs(Cpp.Polymorphic2*)` [MissingImplInMemberAccessInContext]
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

struct VirtA { int a; };
struct VirtB : virtual VirtA {};

void UseB(VirtB * _Nonnull p);

// --- use_virtual_inheritance_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "virtual_inheritance.h";

fn Convert(p: Cpp.VirtB*) {
  // OK, doesn't require `Cpp.VirtB` to be a complete type.
  Cpp.UseB(p);
}

// --- fail_todo_use_virtual_inheritance.carbon

library "[[@TEST_NAME]]";

import Cpp library "virtual_inheritance.h";

fn Convert(p: Cpp.VirtB*) -> Cpp.VirtA* {
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.VirtB*` to `Cpp.VirtA*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.VirtB*` does not implement interface `Core.ImplicitAs(Cpp.VirtA*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- diamond.h

struct DiamondA {
  int a;
};

struct DiamondB : virtual DiamondA {
  int b;
};

struct DiamondC : virtual DiamondA {
  int c;
};

struct DiamondD : DiamondB, DiamondC {
  int d;
};

// --- fail_todo_use_diamond.carbon

library "[[@TEST_NAME]]";

import Cpp library "diamond.h";

fn ConvertBA(p: Cpp.DiamondB*) -> Cpp.DiamondA* {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.DiamondB*` to `Cpp.DiamondA*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:3: note: type `Cpp.DiamondB*` does not implement interface `Core.ImplicitAs(Cpp.DiamondA*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertCA(p: Cpp.DiamondC*) -> Cpp.DiamondA* {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.DiamondC*` to `Cpp.DiamondA*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:3: note: type `Cpp.DiamondC*` does not implement interface `Core.ImplicitAs(Cpp.DiamondA*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertDB(p: Cpp.DiamondD*) -> Cpp.DiamondB* {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.DiamondD*` to `Cpp.DiamondB*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:3: note: type `Cpp.DiamondD*` does not implement interface `Core.ImplicitAs(Cpp.DiamondB*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertDC(p: Cpp.DiamondD*) -> Cpp.DiamondC* {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.DiamondD*` to `Cpp.DiamondC*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:3: note: type `Cpp.DiamondD*` does not implement interface `Core.ImplicitAs(Cpp.DiamondC*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn AccessBA(d: Cpp.DiamondD) -> i32 {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `Cpp.DiamondD` to `Cpp.DiamondB` with `as` [ConversionFailure]
  // CHECK:STDERR:   return (d as Cpp.DiamondB).b;
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:11: note: type `Cpp.DiamondD` does not implement interface `Core.As(Cpp.DiamondB)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return (d as Cpp.DiamondB).b;
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return (d as Cpp.DiamondB).b;
}

fn AccessCA(d: Cpp.DiamondD) -> i32 {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `Cpp.DiamondD` to `Cpp.DiamondC` with `as` [ConversionFailure]
  // CHECK:STDERR:   return (d as Cpp.DiamondC).b;
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:11: note: type `Cpp.DiamondD` does not implement interface `Core.As(Cpp.DiamondC)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return (d as Cpp.DiamondC).b;
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return (d as Cpp.DiamondC).b;
}

fn AccessDB(d: Cpp.DiamondD) -> i32 {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `Cpp.DiamondD` to `Cpp.DiamondB` [ConversionFailure]
  // CHECK:STDERR:   return d.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:10: note: type `Cpp.DiamondD` does not implement interface `Core.ImplicitAs(Cpp.DiamondB)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return d.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return d.b;
}

fn AccessDC(d: Cpp.DiamondD) -> i32 {
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `Cpp.DiamondD` to `Cpp.DiamondC` [ConversionFailure]
  // CHECK:STDERR:   return d.c;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_use_diamond.carbon:[[@LINE+4]]:10: note: type `Cpp.DiamondD` does not implement interface `Core.ImplicitAs(Cpp.DiamondC)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return d.c;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return d.c;
}

// --- final.h

struct FinalA final {};

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

import Cpp library "final.h";

class B {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:16: error: deriving from final type `FinalA`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Cpp.FinalA;
  // CHECK:STDERR:                ^~~~~~~~~~
  // CHECK:STDERR:
  extend base: Cpp.FinalA;
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
// CHECK:STDOUT:   %ptr.976: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.d23: type = pattern_type %ptr.976 [concrete]
// CHECK:STDOUT:   %d.param_patt.9b7: %pattern_type.d23 = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt.336: %pattern_type.d23 = wrapper_binding_pattern d, %d.param_patt.9b7 [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.0e0: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %.866: Core.Form = init_form %ptr.0e0 [concrete]
// CHECK:STDOUT:   %pattern_type.cb0: type = pattern_type %ptr.0e0 [concrete]
// CHECK:STDOUT:   %return.param_patt.b43: %pattern_type.cb0 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.363: %pattern_type.cb0 = return_slot_pattern %return.param_patt.b43, %ptr.0e0 [concrete]
// CHECK:STDOUT:   %ConvertPtr.type: type = fn_type @ConvertPtr [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ConvertPtr: %ConvertPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60d: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.136: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.377: %ptr.as.Copy.impl.Op.type.136 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.0e0, (%Copy.impl_witness.60d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ff1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ff1, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.377, @ptr.as.Copy.impl.Op(%Base) [concrete]
// CHECK:STDOUT:   %AcceptVal.type: type = fn_type @AcceptVal [concrete]
// CHECK:STDOUT:   %AcceptVal: %AcceptVal.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.2f5: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %d.param_patt.0f5: %pattern_type.2f5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt.0e6: %pattern_type.2f5 = wrapper_binding_pattern d, %d.param_patt.0f5 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %ConvertPtr.decl: %ConvertPtr.type = fn_decl @ConvertPtr [concrete = constants.%ConvertPtr] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.d23 = value_param_pattern [concrete = constants.%d.param_patt.9b7]
// CHECK:STDOUT:     %d.patt: %pattern_type.d23 = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt.336]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb0 = out_param_pattern [concrete = constants.%return.param_patt.b43]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb0 = return_slot_pattern %return.param_patt, %ptr.loc7_43 [concrete = constants.%return.patt.363]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref.loc7_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %ptr.loc7_43: type = ptr_type %Base.ref [concrete = constants.%ptr.0e0]
// CHECK:STDOUT:     %.loc7_43: Core.Form = init_form %ptr.loc7_43 [concrete = constants.%.866]
// CHECK:STDOUT:     %d.param: %ptr.976 = value_param call_param0
// CHECK:STDOUT:     %.loc7_29: type = splice_block %ptr.loc7_29 [concrete = constants.%ptr.976] {
// CHECK:STDOUT:       %Cpp.ref.loc7_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc7_29: type = ptr_type %Derived.ref [concrete = constants.%ptr.976]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %ptr.976 = wrapper_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %ptr.0e0 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.0e0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertVal.decl: %ConvertVal.type = fn_decl @ConvertVal [concrete = constants.%ConvertVal] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.2f5 = value_param_pattern [concrete = constants.%d.param_patt.0f5]
// CHECK:STDOUT:     %d.patt: %pattern_type.2f5 = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt.0e6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = wrapper_binding d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertPtr(%d.param: %ptr.976) -> out %return.param: %ptr.0e0 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %ptr.976 = name_ref d, %d
// CHECK:STDOUT:   %.loc8_11.1: ref %Derived = deref %d.ref
// CHECK:STDOUT:   %.loc8_11.2: ref %Base = class_element_access %.loc8_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.0e0 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %.loc8_11.3: %ptr.0e0 = converted %d.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.883 = impl_witness_access constants.%Copy.impl_witness.60d, element0 [concrete = constants.%ptr.as.Copy.impl.Op.377]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Base) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.0e0 = call %bound_method.loc8_11.2(%.loc8_11.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertVal(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AcceptVal.ref: %AcceptVal.type = name_ref AcceptVal, file.%AcceptVal.decl [concrete = constants.%AcceptVal]
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc16_13.1: %Base = as_compatible %d.ref
// CHECK:STDOUT:   %.loc16_13.2: ref %Base = class_element_access %.loc16_13.1, element0
// CHECK:STDOUT:   %.loc16_13.3: ref %Base = converted %d.ref, %.loc16_13.2
// CHECK:STDOUT:   %.loc16_13.4: %Base = acquire_value %.loc16_13.3
// CHECK:STDOUT:   %AcceptVal.call: init %empty_tuple.type = call %AcceptVal.ref(%.loc16_13.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_static_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %StaticBase: type = class_type @StaticBase [concrete]
// CHECK:STDOUT:   %StaticBase.base_fn.cpp_overload_set.type: type = cpp_overload_set_type @StaticBase.base_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticBase.base_fn.cpp_overload_set.value: %StaticBase.base_fn.cpp_overload_set.type = cpp_overload_set_value @StaticBase.base_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticBase.base_fn.type: type = fn_type @StaticBase.base_fn [concrete]
// CHECK:STDOUT:   %StaticBase.base_fn: %StaticBase.base_fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %StaticDerived: type = class_type @StaticDerived [concrete]
// CHECK:STDOUT:   %StaticDerived.base_fn.cpp_overload_set.type: type = cpp_overload_set_type @StaticDerived.base_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticDerived.base_fn.cpp_overload_set.value: %StaticDerived.base_fn.cpp_overload_set.type = cpp_overload_set_value @StaticDerived.base_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticDerived.derived_fn.cpp_overload_set.type: type = cpp_overload_set_type @StaticDerived.derived_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticDerived.derived_fn.cpp_overload_set.value: %StaticDerived.derived_fn.cpp_overload_set.type = cpp_overload_set_value @StaticDerived.derived_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticDerived.derived_fn.type: type = fn_type @StaticDerived.derived_fn [concrete]
// CHECK:STDOUT:   %StaticDerived.derived_fn: %StaticDerived.derived_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticBase = %StaticBase.decl
// CHECK:STDOUT:     .StaticDerived = %StaticDerived.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticBase.decl: type = class_decl @StaticBase [concrete = constants.%StaticBase] {} {}
// CHECK:STDOUT:   %StaticBase.base_fn.cpp_overload_set.value: %StaticBase.base_fn.cpp_overload_set.type = cpp_overload_set_value @StaticBase.base_fn.cpp_overload_set [concrete = constants.%StaticBase.base_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticBase.base_fn.decl: %StaticBase.base_fn.type = fn_decl @StaticBase.base_fn [concrete = constants.%StaticBase.base_fn] {} {}
// CHECK:STDOUT:   %StaticDerived.decl: type = class_decl @StaticDerived [concrete = constants.%StaticDerived] {} {}
// CHECK:STDOUT:   %StaticDerived.base_fn.cpp_overload_set.value: %StaticDerived.base_fn.cpp_overload_set.type = cpp_overload_set_value @StaticDerived.base_fn.cpp_overload_set [concrete = constants.%StaticDerived.base_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticDerived.derived_fn.cpp_overload_set.value: %StaticDerived.derived_fn.cpp_overload_set.type = cpp_overload_set_value @StaticDerived.derived_fn.cpp_overload_set [concrete = constants.%StaticDerived.derived_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticDerived.derived_fn.decl: %StaticDerived.derived_fn.type = fn_decl @StaticDerived.derived_fn [concrete = constants.%StaticDerived.derived_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticBase.ref: type = name_ref StaticBase, imports.%StaticBase.decl [concrete = constants.%StaticBase]
// CHECK:STDOUT:   %base_fn.ref.loc8: %StaticBase.base_fn.cpp_overload_set.type = name_ref base_fn, imports.%StaticBase.base_fn.cpp_overload_set.value [concrete = constants.%StaticBase.base_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticBase.base_fn.call.loc8: init %empty_tuple.type = call imports.%StaticBase.base_fn.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticDerived.ref.loc9: type = name_ref StaticDerived, imports.%StaticDerived.decl [concrete = constants.%StaticDerived]
// CHECK:STDOUT:   %base_fn.ref.loc9: %StaticDerived.base_fn.cpp_overload_set.type = name_ref base_fn, imports.%StaticDerived.base_fn.cpp_overload_set.value [concrete = constants.%StaticDerived.base_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticBase.base_fn.call.loc9: init %empty_tuple.type = call imports.%StaticBase.base_fn.decl()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticDerived.ref.loc10: type = name_ref StaticDerived, imports.%StaticDerived.decl [concrete = constants.%StaticDerived]
// CHECK:STDOUT:   %derived_fn.ref: %StaticDerived.derived_fn.cpp_overload_set.type = name_ref derived_fn, imports.%StaticDerived.derived_fn.cpp_overload_set.value [concrete = constants.%StaticDerived.derived_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticDerived.derived_fn.call: init %empty_tuple.type = call imports.%StaticDerived.derived_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FieldDerived: type = class_type @FieldDerived [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %FieldBase: type = class_type @FieldBase [concrete]
// CHECK:STDOUT:   %FieldBase.elem: type = unbound_element_type %FieldBase, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .FieldDerived = %FieldDerived.decl
// CHECK:STDOUT:     .FieldBase = %FieldBase.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FieldDerived.decl: type = class_decl @FieldDerived [concrete = constants.%FieldDerived] {} {}
// CHECK:STDOUT:   %FieldBase.decl: type = class_decl @FieldBase [concrete = constants.%FieldBase] {} {}
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDirect(%d.param: %FieldDerived) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %FieldDerived = name_ref d, %d
// CHECK:STDOUT:   %a.ref: %FieldBase.elem = name_ref a, @FieldBase.%.1 [concrete = @FieldBase.%.1]
// CHECK:STDOUT:   %.loc8_11.1: %FieldBase = as_compatible %d.ref
// CHECK:STDOUT:   %.loc8_11.2: ref %FieldBase = class_element_access %.loc8_11.1, element0
// CHECK:STDOUT:   %.loc8_11.3: ref %FieldBase = converted %d.ref, %.loc8_11.2
// CHECK:STDOUT:   %.loc8_11.4: ref %i32 = class_element_access %.loc8_11.3, element0
// CHECK:STDOUT:   %.loc8_11.5: %i32 = acquire_value %.loc8_11.4
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.5, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc8_11.2(%.loc8_11.5)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessQualified(%d.param: %FieldDerived) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %FieldDerived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %FieldBase.ref: type = name_ref FieldBase, imports.%FieldBase.decl [concrete = constants.%FieldBase]
// CHECK:STDOUT:   %b.ref: %FieldBase.elem = name_ref b, @FieldBase.%.2 [concrete = @FieldBase.%.2]
// CHECK:STDOUT:   %.loc14_11.1: %FieldBase = as_compatible %d.ref
// CHECK:STDOUT:   %.loc14_11.2: ref %FieldBase = class_element_access %.loc14_11.1, element0
// CHECK:STDOUT:   %.loc14_11.3: ref %FieldBase = converted %d.ref, %.loc14_11.2
// CHECK:STDOUT:   %.loc14_11.4: ref %i32 = class_element_access %.loc14_11.3, element1
// CHECK:STDOUT:   %.loc14_11.5: %i32 = acquire_value %.loc14_11.4
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.5, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.5)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MethodDerived: type = class_type @MethodDerived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MethodBase: type = class_type @MethodBase [concrete]
// CHECK:STDOUT:   %MethodDerived.f.cpp_overload_set.type: type = cpp_overload_set_type @MethodDerived.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %MethodDerived.f.cpp_overload_set.value: %MethodDerived.f.cpp_overload_set.type = cpp_overload_set_value @MethodDerived.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %MethodBase.g.cpp_overload_set.type: type = cpp_overload_set_type @MethodBase.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %MethodBase.g.cpp_overload_set.value: %MethodBase.g.cpp_overload_set.type = cpp_overload_set_value @MethodBase.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %g__carbon_thunk.type: type = fn_type @g__carbon_thunk [concrete]
// CHECK:STDOUT:   %g__carbon_thunk: %g__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MethodDerived = %MethodDerived.decl
// CHECK:STDOUT:     .MethodBase = %MethodBase.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MethodDerived.decl: type = class_decl @MethodDerived [concrete = constants.%MethodDerived] {} {}
// CHECK:STDOUT:   %MethodBase.decl: type = class_decl @MethodBase [concrete = constants.%MethodBase] {} {}
// CHECK:STDOUT:   %MethodDerived.f.cpp_overload_set.value: %MethodDerived.f.cpp_overload_set.type = cpp_overload_set_value @MethodDerived.f.cpp_overload_set [concrete = constants.%MethodDerived.f.cpp_overload_set.value]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MethodBase.g.cpp_overload_set.value: %MethodBase.g.cpp_overload_set.type = cpp_overload_set_value @MethodBase.g.cpp_overload_set [concrete = constants.%MethodBase.g.cpp_overload_set.value]
// CHECK:STDOUT:   %g__carbon_thunk.decl: %g__carbon_thunk.type = fn_decl @g__carbon_thunk [concrete = constants.%g__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDirect(%d.param: %MethodDerived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %MethodDerived = name_ref d, %d
// CHECK:STDOUT:   %f.ref: %MethodDerived.f.cpp_overload_set.type = name_ref f, imports.%MethodDerived.f.cpp_overload_set.value [concrete = constants.%MethodDerived.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %f.ref
// CHECK:STDOUT:   %.loc8_3.1: %MethodBase = as_compatible %d.ref
// CHECK:STDOUT:   %.loc8_3.2: ref %MethodBase = class_element_access %.loc8_3.1, element0
// CHECK:STDOUT:   %.loc8_3.3: ref %MethodBase = converted %d.ref, %.loc8_3.2
// CHECK:STDOUT:   %.loc8_3.4: %MethodBase = acquire_value %.loc8_3.3
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%.loc8_3.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallQualified(%d.param: %MethodDerived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %MethodDerived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MethodBase.ref: type = name_ref MethodBase, imports.%MethodBase.decl [concrete = constants.%MethodBase]
// CHECK:STDOUT:   %g.ref: %MethodBase.g.cpp_overload_set.type = name_ref g, imports.%MethodBase.g.cpp_overload_set.value [concrete = constants.%MethodBase.g.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %g.ref
// CHECK:STDOUT:   %.loc14_3.1: %MethodBase = as_compatible %d.ref
// CHECK:STDOUT:   %.loc14_3.2: ref %MethodBase = class_element_access %.loc14_3.1, element0
// CHECK:STDOUT:   %.loc14_3.3: ref %MethodBase = converted %d.ref, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_3.4: %MethodBase = acquire_value %.loc14_3.3
// CHECK:STDOUT:   %g__carbon_thunk.call: init %empty_tuple.type = call imports.%g__carbon_thunk.decl(%.loc14_3.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBaseQualifiedByDerived(%d.param: %MethodDerived, %b.param: %MethodBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %MethodDerived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MethodDerived.ref.loc23: type = name_ref MethodDerived, imports.%MethodDerived.decl [concrete = constants.%MethodDerived]
// CHECK:STDOUT:   %f.ref.loc23: %MethodDerived.f.cpp_overload_set.type = name_ref f, imports.%MethodDerived.f.cpp_overload_set.value [concrete = constants.%MethodDerived.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %d.ref, %f.ref.loc23
// CHECK:STDOUT:   %.loc23_3.1: %MethodBase = as_compatible %d.ref
// CHECK:STDOUT:   %.loc23_3.2: ref %MethodBase = class_element_access %.loc23_3.1, element0
// CHECK:STDOUT:   %.loc23_3.3: ref %MethodBase = converted %d.ref, %.loc23_3.2
// CHECK:STDOUT:   %.loc23_3.4: %MethodBase = acquire_value %.loc23_3.3
// CHECK:STDOUT:   %f__carbon_thunk.call.loc23: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%.loc23_3.4)
// CHECK:STDOUT:   %b.ref: %MethodBase = name_ref b, %b
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MethodDerived.ref.loc24: type = name_ref MethodDerived, imports.%MethodDerived.decl [concrete = constants.%MethodDerived]
// CHECK:STDOUT:   %f.ref.loc24: %MethodDerived.f.cpp_overload_set.type = name_ref f, imports.%MethodDerived.f.cpp_overload_set.value [concrete = constants.%MethodDerived.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %b.ref, %f.ref.loc24
// CHECK:STDOUT:   %f__carbon_thunk.call.loc24: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
