// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/member_functions.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/member_functions.carbon

// --- basics.carbon
library "[[@TEST_NAME]]";

class Class {
  //@dump-sem-ir-begin
  fn F[self: Self]() -> {};
  fn G[addr self: Self*]() -> {};

  alias A = F;
  //@dump-sem-ir-end

  var k: {};
}

fn Make() -> Class;

//@dump-sem-ir-begin
fn Class.F[self: Self]() -> {} {
  return self.k;
}

fn Call(c: Class) -> {} {
  // TODO: The sem-ir for this call doesn't distinguish the `self` argument from
  // the explicit arguments.
  return c.F();
}

fn CallAlias(c: Class) -> {} {
  return c.A();
}

fn CallOnConstBoundMethod() -> {} {
  return ({.k = {}} as Class).F();
}

fn CallWithAddr() -> {} {
  var c: Class;
  return c.G();
}

fn CallFThroughPointer(p: Class*) -> {} {
  return (*p).F();
}

fn CallGThroughPointer(p: Class*) -> {} {
  return (*p).G();
}

fn CallFOnInitializingExpr() -> {} {
  return Make().F();
}

fn CallGOnInitializingExpr() -> {} {
  return Make().G();
}
//@dump-sem-ir-end

// --- fail_multiple_bindings.carbon
library "[[@TEST_NAME]]";

class X {
  fn F[self: Self]();
}

fn G(x: X) {
  // TODO: Produce a better diagnostic for this case.
  // CHECK:STDERR: fail_multiple_bindings.carbon:[[@LINE+4]]:3: error: member name of type `<bound method>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   x.(x.F)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.(x.F)();
}

// --- fail_addr_not_self.carbon
library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a `self` parameter [AddrOnNonSelfParam]
  // CHECK:STDERR:   fn F[addr a:! Class*]();
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  fn F[addr a:! Class*]();

  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a `self` parameter [AddrOnNonSelfParam]
  // CHECK:STDERR:   fn G(addr b: Class*);
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  fn G(addr b: Class*);
}

// --- fail_addr_self.carbon

library "[[@TEST_NAME]]";

class Class {
  fn F[addr self: Class*]();
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a binding with a pointer type [AddrOnNonPointerType]
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G[addr self: Class]();
}

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.F();

  c.G();

  // This call is OK.
  (*p).F();

  (*p).G();
}

// CHECK:STDERR: fail_addr_self.carbon:[[@LINE+4]]:4: error: name `D` not found [NameNotFound]
// CHECK:STDERR: fn D.F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn D.F() {}

// CHECK:STDERR: fail_addr_self.carbon:[[@LINE+4]]:4: error: name `D` not found [NameNotFound]
// CHECK:STDERR: fn D.C.F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn D.C.F() {}

// --- fail_method_binding.carbon
library "[[@TEST_NAME]]";

class Class {
  fn NoSelf();
  fn WithSelf[self: Class]();
}

alias A = Class.WithSelf;

fn F(c: Class) {
  c.NoSelf();
  c.WithSelf();

  Class.NoSelf();
  // CHECK:STDERR: fail_method_binding.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   Class.WithSelf();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method_binding.carbon:[[@LINE-13]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf();
  // CHECK:STDERR: fail_method_binding.carbon:[[@LINE+7]]:3: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   Class.WithSelf(c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method_binding.carbon:[[@LINE-21]]:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf(c);

  // CHECK:STDERR: fail_method_binding.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   A();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_method_binding.carbon:[[@LINE-30]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A();
}

// --- fail_method_redefinition.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() {}
  // CHECK:STDERR: fail_method_redefinition.carbon:[[@LINE+7]]:3: error: redefinition of `fn F` [RedeclRedef]
  // CHECK:STDERR:   fn F() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_redefinition.carbon:[[@LINE-4]]:3: note: previously defined here [RedeclPrevDef]
  // CHECK:STDERR:   fn F() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  fn F() {}
}

// --- fail_out_of_line_decl.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_out_of_line_decl.carbon:[[@LINE+4]]:6: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fn C.F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn C.F() {}

// --- class_function.carbon
library "[[@TEST_NAME]]";

class Class {
  //@dump-sem-ir-begin
  fn F() -> ();
  //@dump-sem-ir-end
}

fn Run() -> () {
  var c: Class;
  //@dump-sem-ir-begin
  return c.F();
  //@dump-sem-ir-end
}

// --- fail_member_of_let.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F();
}

// TODO: Use `:!` here once it is available.
let T: type = Class;

// The class name is required to be written in the same way as in the class
// declaration. An expression that evaluates to the class name is not accepted.
// CHECK:STDERR: fail_member_of_let.carbon:[[@LINE+7]]:4: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: fn T.F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR: fail_member_of_let.carbon:[[@LINE-7]]:5: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR: let T: type = Class;
// CHECK:STDERR:     ^
// CHECK:STDERR:
fn T.F() {}

// --- fail_memaccess_category.carbon
library "[[@TEST_NAME]]";

class A {
  fn F[addr self: A*]();
}

class B {
  var a: A;
}

fn F(s: {.a: A}, b: B) {
  // `s` has only a value representation, so this must be invalid.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-23]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  b.a.F();
}

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.k.df9: type = struct_type {.k: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallAlias.type: type = fn_type @CallAlias [concrete]
// CHECK:STDOUT:   %CallAlias: %CallAlias.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallOnConstBoundMethod.type: type = fn_type @CallOnConstBoundMethod [concrete]
// CHECK:STDOUT:   %CallOnConstBoundMethod: %CallOnConstBoundMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %CallWithAddr.type: type = fn_type @CallWithAddr [concrete]
// CHECK:STDOUT:   %CallWithAddr: %CallWithAddr.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFThroughPointer.type: type = fn_type @CallFThroughPointer [concrete]
// CHECK:STDOUT:   %CallFThroughPointer: %CallFThroughPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGThroughPointer.type: type = fn_type @CallGThroughPointer [concrete]
// CHECK:STDOUT:   %CallGThroughPointer: %CallGThroughPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFOnInitializingExpr.type: type = fn_type @CallFOnInitializingExpr [concrete]
// CHECK:STDOUT:   %CallFOnInitializingExpr: %CallFOnInitializingExpr.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGOnInitializingExpr.type: type = fn_type @CallGOnInitializingExpr [concrete]
// CHECK:STDOUT:   %CallGOnInitializingExpr: %CallGOnInitializingExpr.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_30.2: type = converted %.loc17_30.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param.loc17: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc17: %Class = bind_name self, %self.param.loc17
// CHECK:STDOUT:     %return.param.loc17: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return.loc17: ref %empty_struct_type = return_slot %return.param.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.761 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_23.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc21_23.2: type = converted %.loc21_23.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias.decl: %CallAlias.type = fn_decl @CallAlias [concrete = constants.%CallAlias] {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.761 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc27_28.2: type = converted %.loc27_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod.decl: %CallOnConstBoundMethod.type = fn_decl @CallOnConstBoundMethod [concrete = constants.%CallOnConstBoundMethod] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc31_33.2: type = converted %.loc31_33.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithAddr.decl: %CallWithAddr.type = fn_decl @CallWithAddr [concrete = constants.%CallWithAddr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc35_23.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc35_23.2: type = converted %.loc35_23.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer.decl: %CallFThroughPointer.type = fn_decl @CallFThroughPointer [concrete = constants.%CallFThroughPointer] {
// CHECK:STDOUT:     %p.patt: %pattern_type.796 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.796 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc40_39.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc40_39.2: type = converted %.loc40_39.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc40_32: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer.decl: %CallGThroughPointer.type = fn_decl @CallGThroughPointer [concrete = constants.%CallGThroughPointer] {
// CHECK:STDOUT:     %p.patt: %pattern_type.796 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.796 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc44_39.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc44_39.2: type = converted %.loc44_39.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc44_32: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr.decl: %CallFOnInitializingExpr.type = fn_decl @CallFOnInitializingExpr [concrete = constants.%CallFOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc48_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc48_34.2: type = converted %.loc48_34.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr.decl: %CallGOnInitializingExpr.type = fn_decl @CallGOnInitializingExpr [concrete = constants.%CallGOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc52_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc52_34.2: type = converted %.loc52_34.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param.loc5: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc5: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc5: %Class = bind_name self, %self.param.loc5
// CHECK:STDOUT:     %return.param.loc5: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return.loc5: ref %empty_struct_type = return_slot %return.param.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_32.2: type = converted %.loc6_32.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc6_23: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %A: %F.type = bind_alias A, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .A = %A
// CHECK:STDOUT:   .k = %.loc11_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc17: %Class) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc17
// CHECK:STDOUT:   %k.ref: %Class.elem = name_ref k, @Class.%.loc11_8 [concrete = @Class.%.loc11_8]
// CHECK:STDOUT:   %.loc18_14.1: ref %empty_struct_type = class_element_access %self.ref, element0
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18_14.2: %empty_struct_type = converted %.loc18_14.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc18_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%self.param: %ptr.e71) -> %empty_struct_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c.param: %Class) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%c.ref)
// CHECK:STDOUT:   %.loc24_14.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc24_14.2: ref %empty_struct_type = temporary %.loc24_14.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_15: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc24_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c.param: %Class) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: %F.type = name_ref A, @Class.%A [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%c.ref)
// CHECK:STDOUT:   %.loc28_14.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc28_14.2: ref %empty_struct_type = temporary %.loc28_14.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc28_15: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc28_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc32_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc32_19.1: %struct_type.k.df9 = struct_literal (%.loc32_18.1)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc32_19.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc32_19.3: ref %empty_struct_type = class_element_access %.loc32_19.2, element0
// CHECK:STDOUT:   %.loc32_18.2: init %empty_struct_type = struct_init () to %.loc32_19.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_19.4: init %empty_struct_type = converted %.loc32_18.1, %.loc32_18.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_19.5: init %Class = class_init (%.loc32_19.4), %.loc32_19.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc32_19.6: ref %Class = temporary %.loc32_19.2, %.loc32_19.5
// CHECK:STDOUT:   %.loc32_21.1: ref %Class = converted %.loc32_19.1, %.loc32_19.6
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc32_21.1, %F.ref
// CHECK:STDOUT:   %.loc32_21.2: %Class = bind_value %.loc32_21.1
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%.loc32_21.2)
// CHECK:STDOUT:   %.loc32_33.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc32_33.2: ref %empty_struct_type = temporary %.loc32_33.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_34: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc32_34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.761 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Class = var %c.var_patt
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %c.ref
// CHECK:STDOUT:   %G.call: init %empty_struct_type = call %G.bound(%addr)
// CHECK:STDOUT:   %.loc37_14.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc37_14.2: ref %empty_struct_type = temporary %.loc37_14.1, %G.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc37_15: %empty_struct_type = converted %G.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc37_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p.param: %ptr.e71) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc41_11.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc41_11.1, %F.ref
// CHECK:STDOUT:   %.loc41_11.2: %Class = bind_value %.loc41_11.1
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%.loc41_11.2)
// CHECK:STDOUT:   %.loc41_17.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc41_17.2: ref %empty_struct_type = temporary %.loc41_17.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc41_18: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc41_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p.param: %ptr.e71) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc45_11: ref %Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %.loc45_11, %G.ref
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %.loc45_11
// CHECK:STDOUT:   %G.call: init %empty_struct_type = call %G.bound(%addr)
// CHECK:STDOUT:   %.loc45_17.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc45_17.2: ref %empty_struct_type = temporary %.loc45_17.1, %G.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc45_18: %empty_struct_type = converted %G.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc45_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc49_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc49_15.1
// CHECK:STDOUT:   %.loc49_15.2: ref %Class = temporary %.loc49_15.1, %Make.call
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc49_15.2, %F.ref
// CHECK:STDOUT:   %.loc49_15.3: %Class = bind_value %.loc49_15.2
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%.loc49_15.3)
// CHECK:STDOUT:   %.loc49_19.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc49_19.2: ref %empty_struct_type = temporary %.loc49_19.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc49_20: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc49_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc53_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc53_15.1
// CHECK:STDOUT:   %.loc53_15.2: ref %Class = temporary %.loc53_15.1, %Make.call
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %.loc53_15.2, %G.ref
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %.loc53_15.2
// CHECK:STDOUT:   %G.call: init %empty_struct_type = call %G.bound(%addr)
// CHECK:STDOUT:   %.loc53_19.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc53_19.2: ref %empty_struct_type = temporary %.loc53_19.1, %G.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc53_20: %empty_struct_type = converted %G.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc53_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   %.loc12_14.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc12_14.2: ref %empty_tuple.type = temporary %.loc12_14.1, %F.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_15: %empty_tuple.type = converted %F.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc12_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
