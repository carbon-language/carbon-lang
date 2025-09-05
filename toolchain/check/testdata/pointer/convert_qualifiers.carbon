// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/convert_qualifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/convert_qualifiers.carbon

// --- same_or_added_qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn TakeConst(p: const X*);
fn TakeNonConst(p: X*);

fn NonConst(p: X*) {
  //@dump-sem-ir-begin
  TakeNonConst(p);
  TakeConst(p);
  //@dump-sem-ir-end
}
fn Const(p: const X*) {
  //@dump-sem-ir-begin
  TakeConst(p);
  //@dump-sem-ir-end
}

// Same thing, but the pointee type happens to be a pointer.
fn TakeConstConst(p: const (const X*)*);
fn NonConstConst(p: const X**) {
  //@dump-sem-ir-begin
  TakeConstConst(p);
  //@dump-sem-ir-end
}

// --- fail_removed_qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn TakeNonConst(p: X*);

// CHECK:STDERR: fail_removed_qualifiers.carbon:[[@LINE+10]]:38: error: cannot implicitly convert expression of type `const X*` to `X*` [ConversionFailure]
// CHECK:STDERR: fn Const(p: const X*) { TakeNonConst(p); }
// CHECK:STDERR:                                      ^
// CHECK:STDERR: fail_removed_qualifiers.carbon:[[@LINE+7]]:38: note: type `const X*` does not implement interface `Core.ImplicitAs(X*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn Const(p: const X*) { TakeNonConst(p); }
// CHECK:STDERR:                                      ^
// CHECK:STDERR: fail_removed_qualifiers.carbon:[[@LINE-8]]:17: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn TakeNonConst(p: X*);
// CHECK:STDERR:                 ^~~~~
// CHECK:STDERR:
fn Const(p: const X*) { TakeNonConst(p); }

// --- fail_todo_nested_qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn TakeConstConst(p: const (const X*)*);

// TODO: We should allow these conversions.

fn NonConstNonConst(p: X**) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+10]]:18: error: cannot implicitly convert expression of type `X**` to `const (const X*)*` [ConversionFailure]
  // CHECK:STDERR:   TakeConstConst(p);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+7]]:18: note: type `X**` does not implement interface `Core.ImplicitAs(const (const X*)*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   TakeConstConst(p);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE-12]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeConstConst(p: const (const X*)*);
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakeConstConst(p);
  //@dump-sem-ir-end
}

fn ConstNonConst(p: const (X*)*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+10]]:18: error: cannot implicitly convert expression of type `const (X*)*` to `const (const X*)*` [ConversionFailure]
  // CHECK:STDERR:   TakeConstConst(p);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+7]]:18: note: type `const (X*)*` does not implement interface `Core.ImplicitAs(const (const X*)*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   TakeConstConst(p);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE-28]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeConstConst(p: const (const X*)*);
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakeConstConst(p);
  //@dump-sem-ir-end
}

// --- fail_nested_added_qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn TakeNonConstConst(p: (const X*)*);

// CHECK:STDERR: fail_nested_added_qualifiers.carbon:[[@LINE+10]]:49: error: cannot implicitly convert expression of type `X**` to `const X**` [ConversionFailure]
// CHECK:STDERR: fn NonConstNonConst(p: X**) { TakeNonConstConst(p); }
// CHECK:STDERR:                                                 ^
// CHECK:STDERR: fail_nested_added_qualifiers.carbon:[[@LINE+7]]:49: note: type `X**` does not implement interface `Core.ImplicitAs(const X**)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn NonConstNonConst(p: X**) { TakeNonConstConst(p); }
// CHECK:STDERR:                                                 ^
// CHECK:STDERR: fail_nested_added_qualifiers.carbon:[[@LINE-8]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn TakeNonConstConst(p: (const X*)*);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn NonConstNonConst(p: X**) { TakeNonConstConst(p); }

// CHECK:STDOUT: --- same_or_added_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const.dde: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.cbd: type = ptr_type %const.dde [concrete]
// CHECK:STDOUT:   %TakeConst.type: type = fn_type @TakeConst [concrete]
// CHECK:STDOUT:   %TakeConst: %TakeConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeNonConst.type: type = fn_type @TakeNonConst [concrete]
// CHECK:STDOUT:   %TakeNonConst: %TakeNonConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.5df: type = const_type %ptr.cbd [concrete]
// CHECK:STDOUT:   %ptr.20a: type = ptr_type %const.5df [concrete]
// CHECK:STDOUT:   %TakeConstConst.type: type = fn_type @TakeConstConst [concrete]
// CHECK:STDOUT:   %TakeConstConst: %TakeConstConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dd0: type = ptr_type %ptr.cbd [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConst(%p.param: %ptr.d17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeNonConst.ref: %TakeNonConst.type = name_ref TakeNonConst, file.%TakeNonConst.decl [concrete = constants.%TakeNonConst]
// CHECK:STDOUT:   %p.ref.loc11: %ptr.d17 = name_ref p, %p
// CHECK:STDOUT:   %TakeNonConst.call: init %empty_tuple.type = call %TakeNonConst.ref(%p.ref.loc11)
// CHECK:STDOUT:   %TakeConst.ref: %TakeConst.type = name_ref TakeConst, file.%TakeConst.decl [concrete = constants.%TakeConst]
// CHECK:STDOUT:   %p.ref.loc12: %ptr.d17 = name_ref p, %p
// CHECK:STDOUT:   %.loc12_13.1: %ptr.cbd = as_compatible %p.ref.loc12
// CHECK:STDOUT:   %.loc12_13.2: %ptr.cbd = converted %p.ref.loc12, %.loc12_13.1
// CHECK:STDOUT:   %TakeConst.call: init %empty_tuple.type = call %TakeConst.ref(%.loc12_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Const(%p.param: %ptr.cbd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConst.ref: %TakeConst.type = name_ref TakeConst, file.%TakeConst.decl [concrete = constants.%TakeConst]
// CHECK:STDOUT:   %p.ref: %ptr.cbd = name_ref p, %p
// CHECK:STDOUT:   %TakeConst.call: init %empty_tuple.type = call %TakeConst.ref(%p.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConst(%p.param: %ptr.dd0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstConst.ref: %TakeConstConst.type = name_ref TakeConstConst, file.%TakeConstConst.decl [concrete = constants.%TakeConstConst]
// CHECK:STDOUT:   %p.ref: %ptr.dd0 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_18.1: %ptr.20a = as_compatible %p.ref
// CHECK:STDOUT:   %.loc25_18.2: %ptr.20a = converted %p.ref, %.loc25_18.1
// CHECK:STDOUT:   %TakeConstConst.call: init %empty_tuple.type = call %TakeConstConst.ref(%.loc25_18.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_nested_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const.dde: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.cbd: type = ptr_type %const.dde [concrete]
// CHECK:STDOUT:   %const.5df: type = const_type %ptr.cbd [concrete]
// CHECK:STDOUT:   %ptr.20a: type = ptr_type %const.5df [concrete]
// CHECK:STDOUT:   %TakeConstConst.type: type = fn_type @TakeConstConst [concrete]
// CHECK:STDOUT:   %TakeConstConst: %TakeConstConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b1b: type = ptr_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %const.2bc: type = const_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %ptr.297: type = ptr_type %const.2bc [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstNonConst(%p.param: %ptr.b1b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstConst.ref: %TakeConstConst.type = name_ref TakeConstConst, file.%TakeConstConst.decl [concrete = constants.%TakeConstConst]
// CHECK:STDOUT:   %p.ref: %ptr.b1b = name_ref p, %p
// CHECK:STDOUT:   %.loc22: %ptr.20a = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstConst.call: init %empty_tuple.type = call %TakeConstConst.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstNonConst(%p.param: %ptr.297) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstConst.ref: %TakeConstConst.type = name_ref TakeConstConst, file.%TakeConstConst.decl [concrete = constants.%TakeConstConst]
// CHECK:STDOUT:   %p.ref: %ptr.297 = name_ref p, %p
// CHECK:STDOUT:   %.loc38: %ptr.20a = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstConst.call: init %empty_tuple.type = call %TakeConstConst.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
