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

// CHECK:STDERR: fail_removed_qualifiers.carbon:[[@LINE+4]]:38: error: cannot implicitly convert expression of type `const X*` to `X*` [ConversionFailure]
// CHECK:STDERR: fail_removed_qualifiers.carbon:[[@LINE+3]]:38: note: type `const X*` does not implement interface `Core.ImplicitAs(X*)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fail_removed_qualifiers.carbon:[[@LINE-4]]:17-21: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:
fn Const(p: const X*) { TakeNonConst(p); }

// --- fail_todo_nested_qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn TakeConstConst(p: const (const X*)*);

// TODO: We should allow these conversions.

fn NonConstNonConst(p: X**) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+4]]:18: error: cannot implicitly convert expression of type `X**` to `const (const X*)*` [ConversionFailure]
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+3]]:18: note: type `X**` does not implement interface `Core.ImplicitAs(const (const X*)*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE-8]]:19-38: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  TakeConstConst(p);
  //@dump-sem-ir-end
}

fn ConstNonConst(p: const (X*)*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+4]]:18: error: cannot implicitly convert expression of type `const (X*)*` to `const (const X*)*` [ConversionFailure]
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE+3]]:18: note: type `const (X*)*` does not implement interface `Core.ImplicitAs(const (const X*)*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: fail_todo_nested_qualifiers.carbon:[[@LINE-18]]:19-38: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  TakeConstConst(p);
  //@dump-sem-ir-end
}

// --- fail_nested_added_qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn TakeNonConstConst(p: (const X*)*);

// CHECK:STDERR: fail_nested_added_qualifiers.carbon:[[@LINE+4]]:49: error: cannot implicitly convert expression of type `X**` to `const X**` [ConversionFailure]
// CHECK:STDERR: fail_nested_added_qualifiers.carbon:[[@LINE+3]]:49: note: type `X**` does not implement interface `Core.ImplicitAs(const X**)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fail_nested_added_qualifiers.carbon:[[@LINE-4]]:22-35: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:
fn NonConstNonConst(p: X**) { TakeNonConstConst(p); }

// CHECK:STDOUT: --- same_or_added_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %const.435: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.faf: type = ptr_type %const.435 [concrete]
// CHECK:STDOUT:   %TakeConst.type: type = fn_type @TakeConst [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakeConst: %TakeConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %TakeNonConst.type: type = fn_type @TakeNonConst [concrete]
// CHECK:STDOUT:   %TakeNonConst: %TakeNonConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.95a: type = const_type %ptr.faf [concrete]
// CHECK:STDOUT:   %ptr.b53: type = ptr_type %const.95a [concrete]
// CHECK:STDOUT:   %TakeConstConst.type: type = fn_type @TakeConstConst [concrete]
// CHECK:STDOUT:   %TakeConstConst: %TakeConstConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.845: type = ptr_type %ptr.faf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConst(%p.param: %ptr.db8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeNonConst.ref: %TakeNonConst.type = name_ref TakeNonConst, file.%TakeNonConst.decl [concrete = constants.%TakeNonConst]
// CHECK:STDOUT:   %p.ref.loc11: %ptr.db8 = name_ref p, %p
// CHECK:STDOUT:   %TakeNonConst.call: init %empty_tuple.type = call %TakeNonConst.ref(%p.ref.loc11)
// CHECK:STDOUT:   %TakeConst.ref: %TakeConst.type = name_ref TakeConst, file.%TakeConst.decl [concrete = constants.%TakeConst]
// CHECK:STDOUT:   %p.ref.loc12: %ptr.db8 = name_ref p, %p
// CHECK:STDOUT:   %.loc12_13.1: %ptr.faf = as_compatible %p.ref.loc12
// CHECK:STDOUT:   %.loc12_13.2: %ptr.faf = converted %p.ref.loc12, %.loc12_13.1
// CHECK:STDOUT:   %TakeConst.call: init %empty_tuple.type = call %TakeConst.ref(%.loc12_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Const(%p.param: %ptr.faf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConst.ref: %TakeConst.type = name_ref TakeConst, file.%TakeConst.decl [concrete = constants.%TakeConst]
// CHECK:STDOUT:   %p.ref: %ptr.faf = name_ref p, %p
// CHECK:STDOUT:   %TakeConst.call: init %empty_tuple.type = call %TakeConst.ref(%p.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConst(%p.param: %ptr.845) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstConst.ref: %TakeConstConst.type = name_ref TakeConstConst, file.%TakeConstConst.decl [concrete = constants.%TakeConstConst]
// CHECK:STDOUT:   %p.ref: %ptr.845 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_18.1: %ptr.b53 = as_compatible %p.ref
// CHECK:STDOUT:   %.loc25_18.2: %ptr.b53 = converted %p.ref, %.loc25_18.1
// CHECK:STDOUT:   %TakeConstConst.call: init %empty_tuple.type = call %TakeConstConst.ref(%.loc25_18.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_nested_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %const.435: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.faf: type = ptr_type %const.435 [concrete]
// CHECK:STDOUT:   %const.95a: type = const_type %ptr.faf [concrete]
// CHECK:STDOUT:   %ptr.b53: type = ptr_type %const.95a [concrete]
// CHECK:STDOUT:   %TakeConstConst.type: type = fn_type @TakeConstConst [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakeConstConst: %TakeConstConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %ptr.9ae: type = ptr_type %ptr.db8 [concrete]
// CHECK:STDOUT:   %const.d43: type = const_type %ptr.db8 [concrete]
// CHECK:STDOUT:   %ptr.060: type = ptr_type %const.d43 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstNonConst(%p.param: %ptr.9ae) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstConst.ref: %TakeConstConst.type = name_ref TakeConstConst, file.%TakeConstConst.decl [concrete = constants.%TakeConstConst]
// CHECK:STDOUT:   %p.ref: %ptr.9ae = name_ref p, %p
// CHECK:STDOUT:   %.loc16: %ptr.b53 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstConst.call: init %empty_tuple.type = call %TakeConstConst.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstNonConst(%p.param: %ptr.060) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstConst.ref: %TakeConstConst.type = name_ref TakeConstConst, file.%TakeConstConst.decl [concrete = constants.%TakeConstConst]
// CHECK:STDOUT:   %p.ref: %ptr.060 = name_ref p, %p
// CHECK:STDOUT:   %.loc26: %ptr.b53 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstConst.call: init %empty_tuple.type = call %TakeConstConst.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
