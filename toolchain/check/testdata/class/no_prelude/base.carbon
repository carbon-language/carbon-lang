// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/base.carbon

// --- base.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
base class Base {
  var b: {};
}

class Derived {
  extend base: Base;

  var d: {};
}
//@dump-sem-ir-end

fn Make() -> Derived {
  //@dump-sem-ir-begin
  return {.base = {.b = {}}, .d = {}};
  //@dump-sem-ir-end
}

fn Access(d: Derived) -> ({}, {}) {
  //@dump-sem-ir-begin
  return (d.d, d.base.b);
  //@dump-sem-ir-end
}

// --- base_field.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: {};
  var b: {};
  var c: {};
}

class Derived {
  extend base: Base;

  var d: {};
  var e: {};
}

fn Access(p: Derived*) -> {}* {
  //@dump-sem-ir-begin
  return &(*p).c;
  //@dump-sem-ir-end
}

// --- fail_base_after_field.carbon

library "[[@TEST_NAME]]";

base class Base {
}

class Derived {
  var d: {};

  // CHECK:STDERR: fail_base_after_field.carbon:[[@LINE+4]]:3: error: `base` declaration must appear before field declarations [BaseDeclAfterFieldDecl]
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Base;
}

// --- compound_field.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: {};
  var b: {};
  var c: {};
}

class Derived {
  extend base: Base;

  var d: {};
  var e: {};
}

//@dump-sem-ir-begin
fn AccessDerived(d: Derived) -> {} {
  return d.(Derived.d);
}

fn AccessBase(d: Derived) -> {} {
  return d.(Base.b);
}

fn AccessDerivedIndirect(p: Derived*) -> {}* {
  return &p->(Derived.d);
}

fn AccessBaseIndirect(p: Derived*) -> {}* {
  return &p->(Base.b);
}
//@dump-sem-ir-end

// --- fail_base_unbound.carbon

library "[[@TEST_NAME]]";

base class B {}

class C {
  extend base: B;
}

// CHECK:STDERR: fail_base_unbound.carbon:[[@LINE+4]]:12: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: let b: B = C.base;
// CHECK:STDERR:            ^~~~~~
// CHECK:STDERR:
let b: B = C.base;

// --- fail_base_as_declared_name.carbon

namespace N;

// CHECK:STDERR: fail_base_as_declared_name.carbon:[[@LINE+8]]:6: error: `.` should be followed by a name [ExpectedDeclNameAfterPeriod]
// CHECK:STDERR: fn N.base() {}
// CHECK:STDERR:      ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_base_as_declared_name.carbon:[[@LINE+4]]:6: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn N.base() {}
// CHECK:STDERR:      ^~~~
// CHECK:STDERR:
fn N.base() {}

// --- fail_base_method_define.carbon

library "[[@TEST_NAME]]";

base class B {
  fn F();

  class C {
    fn F();
  }
}

class D {
  extend base: B;
}

// CHECK:STDERR: fail_base_method_define.carbon:[[@LINE+4]]:6: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fn D.F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn D.F() {}

// CHECK:STDERR: fail_base_method_define.carbon:[[@LINE+4]]:6: error: name `C` not found [NameNotFound]
// CHECK:STDERR: fn D.C.F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn D.C.F() {}

// --- fail_base_misplaced.carbon

library "[[@TEST_NAME]]";

base class B {}

// CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+4]]:1: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
// CHECK:STDERR: extend base: B;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
extend base: B;

fn F() {
  // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+4]]:3: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   extend base: B;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: B;
}

class C {
  fn F() {
    // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+4]]:5: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
    // CHECK:STDERR:     extend base: B;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR:
    extend base: B;
  }
}

// --- fail_base_modifiers.carbon

library "[[@TEST_NAME]]";

base class B {}

class C1 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: error: `private` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   private extend base: B;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private extend base: B;
}

class C2 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+8]]:3: error: `abstract` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: error: missing `extend` before `base` declaration [BaseMissingExtend]
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract base: B;
}

class C3 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:10: error: `default` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   extend default base: B;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  extend default base: B;
}

class C4 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+7]]:10: error: `extend` repeated on declaration [ModifierRepeated]
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: note: `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend extend base: B;
}

// --- fail_base_no_extend.carbon

library "[[@TEST_NAME]]";

base class B {}

class C {
  // CHECK:STDERR: fail_base_no_extend.carbon:[[@LINE+4]]:3: error: missing `extend` before `base` declaration [BaseMissingExtend]
  // CHECK:STDERR:   base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  base: B;
}

// --- fail_base_repeated.carbon

library "[[@TEST_NAME]]";

base class B1 {}
base class B2 {}

class C {
  extend base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+7]]:3: error: multiple `base` declarations in class; multiple inheritance is not permitted [BaseDeclRepeated]
  // CHECK:STDERR:   extend base: B2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-4]]:3: note: previous `base` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:   extend base: B1;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: B2;
}

class D {
  // TODO: Consider adding a custom diagnostic for this case.
  extend base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+7]]:3: error: multiple `base` declarations in class; multiple inheritance is not permitted [BaseDeclRepeated]
  // CHECK:STDERR:   extend base: B1;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-4]]:3: note: previous `base` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:   extend base: B1;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: B1;
}

// --- fail_extend_cycle.carbon

library "[[@TEST_NAME]]";

base class A {
}

base class B {
  // This ensures that the compiler treats A as complete.
  extend base: A;
}

// CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+7]]:1: error: redefinition of `class A` [RedeclRedef]
// CHECK:STDERR: base class A {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE-11]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: base class A {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
base class A {
  extend base: A;
  // CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+4]]:10: error: name `C` not found [NameNotFound]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  var c: C;
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.66e: <witness> = complete_type_witness %struct_type.b.347 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.69e: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.ad9: type = unbound_element_type %Derived, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.base.d.f04: type = struct_type {.base: %Base, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.base.d.f04 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.aae: type = struct_type {.base: %struct_type.b.347, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val, %empty_struct) [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc6_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_11.2: type = converted %.loc6_11.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc6_8: %Base.elem = field_decl b, element0 [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete = constants.%struct_type.b.347]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.b [concrete = constants.%complete_type.66e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc10: %Derived.elem.69e = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc12_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_11.2: type = converted %.loc12_11.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc12_8: %Derived.elem.ad9 = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: %Base, .d: %empty_struct_type} [concrete = constants.%struct_type.base.d.f04]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d [concrete = constants.%complete_type.fa5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   .d = %.loc12_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_27.1: %struct_type.b.347 = struct_literal (%.loc18_26.1)
// CHECK:STDOUT:   %.loc18_36.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_37.1: %struct_type.base.d.aae = struct_literal (%.loc18_27.1, %.loc18_36.1)
// CHECK:STDOUT:   %.loc18_37.2: ref %Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc18_27.2: ref %empty_struct_type = class_element_access %.loc18_37.2, element0
// CHECK:STDOUT:   %.loc18_26.2: init %empty_struct_type = struct_init () to %.loc18_27.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18_27.3: init %empty_struct_type = converted %.loc18_26.1, %.loc18_26.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18_27.4: init %Base = class_init (%.loc18_27.3), %.loc18_37.2 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc18_37.3: init %Base = converted %.loc18_27.1, %.loc18_27.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc18_37.4: ref %empty_struct_type = class_element_access %return, element1
// CHECK:STDOUT:   %.loc18_36.2: init %empty_struct_type = struct_init () to %.loc18_37.4 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18_37.5: init %empty_struct_type = converted %.loc18_36.1, %.loc18_36.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18_37.6: init %Derived = class_init (%.loc18_37.3, %.loc18_37.5), %return [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc18_38: init %Derived = converted %.loc18_37.1, %.loc18_37.6 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   return %.loc18_38 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d.param: %Derived) -> %return.param: %tuple.type.b6b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc24_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc24_12: %Derived.elem.ad9 = name_ref d, @Derived.%.loc12_8 [concrete = @Derived.%.loc12_8]
// CHECK:STDOUT:   %.loc24_12.1: ref %empty_struct_type = class_element_access %d.ref.loc24_11, element1
// CHECK:STDOUT:   %empty_struct.loc24_12: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_12.2: %empty_struct_type = converted %.loc24_12.1, %empty_struct.loc24_12 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %d.ref.loc24_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %Derived.elem.69e = name_ref base, @Derived.%.loc10 [concrete = @Derived.%.loc10]
// CHECK:STDOUT:   %.loc24_17.1: ref %Base = class_element_access %d.ref.loc24_16, element0
// CHECK:STDOUT:   %.loc24_17.2: %Base = bind_value %.loc24_17.1
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc6_8 [concrete = @Base.%.loc6_8]
// CHECK:STDOUT:   %.loc24_22.1: ref %empty_struct_type = class_element_access %.loc24_17.2, element0
// CHECK:STDOUT:   %empty_struct.loc24_22: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_22.2: %empty_struct_type = converted %.loc24_22.1, %empty_struct.loc24_22 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_24.1: %tuple.type.b6b = tuple_literal (%.loc24_12.2, %.loc24_22.2)
// CHECK:STDOUT:   %tuple.elem0: ref %empty_struct_type = tuple_access %return, element0
// CHECK:STDOUT:   %.loc24_12.3: init %empty_struct_type = struct_init () to %tuple.elem0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_24.2: init %empty_struct_type = converted %.loc24_12.2, %.loc24_12.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_struct_type = tuple_access %return, element1
// CHECK:STDOUT:   %.loc24_22.3: init %empty_struct_type = struct_init () to %tuple.elem1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_24.3: init %empty_struct_type = converted %.loc24_22.2, %.loc24_22.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_24.4: init %tuple.type.b6b = tuple_init (%.loc24_24.2, %.loc24_24.3) to %return [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc24_25: init %tuple.type.b6b = converted %.loc24_24.1, %.loc24_24.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc24_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p.param: %ptr.404) -> %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %.loc19_12: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %c.ref: %Base.elem = name_ref c, @Base.%.loc7_8 [concrete = @Base.%.loc7_8]
// CHECK:STDOUT:   %.loc19_15.1: ref %Base = class_element_access %.loc19_12, element0
// CHECK:STDOUT:   %.loc19_15.2: ref %Base = converted %.loc19_12, %.loc19_15.1
// CHECK:STDOUT:   %.loc19_15.3: ref %empty_struct_type = class_element_access %.loc19_15.2, element2
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %.loc19_15.3
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.ad9: type = unbound_element_type %Derived, %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [concrete]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [concrete]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %AccessDerivedIndirect.type: type = fn_type @AccessDerivedIndirect [concrete]
// CHECK:STDOUT:   %AccessDerivedIndirect: %AccessDerivedIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessBaseIndirect.type: type = fn_type @AccessBaseIndirect [concrete]
// CHECK:STDOUT:   %AccessBaseIndirect: %AccessBaseIndirect.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [concrete = constants.%AccessDerived] {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc18_34.2: type = converted %.loc18_34.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref.loc18: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [concrete = constants.%AccessBase] {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_31.2: type = converted %.loc22_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerivedIndirect.decl: %AccessDerivedIndirect.type = fn_decl @AccessDerivedIndirect [concrete = constants.%AccessDerivedIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1cc = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1cc = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_43: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc26_44: type = converted %.loc26_43, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc26_44: type = ptr_type %.loc26_44 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc26_36: type = splice_block %ptr.loc26_36 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref.loc26: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc26_36: type = ptr_type %Derived.ref.loc26 [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.c28 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.c28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBaseIndirect.decl: %AccessBaseIndirect.type = fn_decl @AccessBaseIndirect [concrete = constants.%AccessBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1cc = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1cc = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30_40: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc30_41: type = converted %.loc30_40, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc30_41: type = ptr_type %.loc30_41 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc30_33: type = splice_block %ptr.loc30_33 [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc30_33: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.c28 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.c28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerived(%d.param: %Derived) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc19_10: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Derived.ref.loc19: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref.loc19_20: %Derived.elem.ad9 = name_ref d, @Derived.%.loc13_8 [concrete = @Derived.%.loc13_8]
// CHECK:STDOUT:   %.loc19_11.1: ref %empty_struct_type = class_element_access %d.ref.loc19_10, element1
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_11.2: %empty_struct_type = converted %.loc19_11.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc19_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBase(%d.param: %Derived) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc6_8 [concrete = @Base.%.loc6_8]
// CHECK:STDOUT:   %.loc23_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc23_11.2: ref %Base = converted %d.ref, %.loc23_11.1
// CHECK:STDOUT:   %.loc23_11.3: ref %empty_struct_type = class_element_access %.loc23_11.2, element1
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_20: %empty_struct_type = converted %.loc23_11.3, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc23_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerivedIndirect(%p.param: %ptr.404) -> %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Derived.ref.loc27: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref: %Derived.elem.ad9 = name_ref d, @Derived.%.loc13_8 [concrete = @Derived.%.loc13_8]
// CHECK:STDOUT:   %.loc27_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc27_12.2: ref %empty_struct_type = class_element_access %.loc27_12.1, element1
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %.loc27_12.2
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBaseIndirect(%p.param: %ptr.404) -> %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc6_8 [concrete = @Base.%.loc6_8]
// CHECK:STDOUT:   %.loc31_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc31_12.2: ref %Base = class_element_access %.loc31_12.1, element0
// CHECK:STDOUT:   %.loc31_12.3: ref %Base = converted %.loc31_12.1, %.loc31_12.2
// CHECK:STDOUT:   %.loc31_12.4: ref %empty_struct_type = class_element_access %.loc31_12.3, element1
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %.loc31_12.4
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
