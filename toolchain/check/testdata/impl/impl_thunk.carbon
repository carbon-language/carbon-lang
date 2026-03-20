// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_thunk.carbon

// --- no_thunk_param_name_differs.carbon

library "[[@TEST_NAME]]";

interface I { fn F(x: Self); }

// This is OK, and does not require a thunk despite the parameter name differing.
class C {
  impl as I {
    fn F(not_x: C);
  }
}

// --- struct_conversion.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(x: {.a: (), .b: {}}) -> {.c: (), .d: {}};
}

impl () as I {
  //@dump-sem-ir-begin
  fn F(y: {.b: {}, .a: ()}) -> {.d: {}, .c: ()};
  //@dump-sem-ir-end
}

// --- inheritance_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F[ref self: Self](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[ref self: A](other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F[self: Self](other: Self);
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[self: A](other: A);
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion_pointer.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F[ref self: Self](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[ref self: A](other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion_pointer_addr.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F[ref self: Self](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[ref self: A](other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- fail_inheritance_value_conversion_copy_return.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F() -> Self;
}

impl A as X {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE+10]]:3: error: cannot copy value of type `A` [CopyOfUncopyableType]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE+7]]:3: note: type `A` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE-11]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F() -> Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F() -> B;
  //@dump-sem-ir-end
}

// --- fail_param_type_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+6]]:8: error: cannot implicitly convert expression of type `A` to `B` [ConversionFailure]
  // CHECK:STDERR:   fn F(a: Self);
  // CHECK:STDERR:        ^~~~~~~
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+3]]:8: note: type `A` does not implement interface `Core.ImplicitAs(B)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   fn F(a: Self);
  // CHECK:STDERR:        ^~~~~~~
  fn F(a: Self);
}

class A {}
class B {}

impl A as I {
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+7]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F(a: B);
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE-10]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F(a: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(a: B);
}

// --- fail_return_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F() -> Self;
}

class A {}
class B {}

impl A as I {
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `B` to `A` [ConversionFailure]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE+7]]:3: note: type `B` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE-13]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F() -> Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F() -> B;
}

// --- return_empty_tuple_mismatch_allowed.carbon

library "[[@TEST_NAME]]";

interface I {
  fn HasReturn() -> Self;
  fn NoReturn();
  fn EmptyTupleReturn() -> ();
}

impl () as I {
  // OK, can `return HasReturn();` in thunk.
  fn HasReturn();

  // OK, exact match.
  fn NoReturn();

  // OK, same as `HasReturn`.
  fn EmptyTupleReturn();
}

// --- fail_return_empty_tuple_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  fn NoReturn();
}

impl () as I {
  // TODO: The proposal says to reject this. But should we really do so?
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE+7]]:3: error: function redeclaration differs because return type is `()` [FunctionRedeclReturnTypeDiffers]
  // CHECK:STDERR:   fn NoReturn() -> ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE-8]]:3: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
  // CHECK:STDERR:   fn NoReturn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn NoReturn() -> ();
}

class C {}

impl C as I {
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE+7]]:3: error: function redeclaration differs because return type is `C` [FunctionRedeclReturnTypeDiffers]
  // CHECK:STDERR:   fn NoReturn() -> C;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE-21]]:3: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
  // CHECK:STDERR:   fn NoReturn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn NoReturn() -> C;
}

// --- fail_param_type_incomplete.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+3]]:8: error: parameter has incomplete type `B` in function definition [IncompleteTypeInFunctionParam]
  // CHECK:STDERR:   fn F(a: T);
  // CHECK:STDERR:        ^~~~
  fn F(a: T);
}

// CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+3]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
class B;

class C {
  impl as I(B) {
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+17]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-14]]:8: error: cannot implicitly convert expression of type `B` to `C` [ConversionFailure]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:        ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-17]]:8: note: type `B` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:        ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-23]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:   ^~~~~~~~~~~
    // CHECK:STDERR:
    fn F(c: C);
  }
}

// --- generic_function.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[T:! type](x: T*) -> T*;
}

impl () as I {
  //@dump-sem-ir-begin
  fn F[U:! Core.Copy](x: U) -> U { return x; }
  //@dump-sem-ir-end
}

// --- generic_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[self: Self, T:! type](x: T*) -> T*;
}

impl () as I {
  //@dump-sem-ir-begin
  fn F[unused self: (), U:! Core.Copy](x: U) -> U { return x; }
  //@dump-sem-ir-end
}

// --- generic_interface.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F() -> {.a: T, .b: T};
}

impl () as I({}) {
  //@dump-sem-ir-begin
  fn F() -> {.b: {}, .a: {}};
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- struct_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.391: type = struct_type {.a: %empty_tuple.type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.c.d.15a: type = struct_type {.c: %empty_tuple.type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b.a.40c: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.231: type = pattern_type %struct_type.b.a.40c [concrete]
// CHECK:STDOUT:   %struct_type.d.c.b36: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %.4b1: Core.Form = init_form %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %pattern_type.844: type = pattern_type %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc81.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc81.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.c.d.15a = struct_value (%empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc44.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc81.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc81.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1] {
// CHECK:STDOUT:     %y.param_patt: %pattern_type.231 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.231 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.844 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.844 = return_slot_pattern %return.param_patt, %struct_type.d.c [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc73.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc73.2: type = converted %.loc73.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc79.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc79.2: type = converted %.loc79.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.c: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete = constants.%struct_type.d.c.b36]
// CHECK:STDOUT:     %.loc80: Core.Form = init_form %struct_type.d.c [concrete = constants.%.4b1]
// CHECK:STDOUT:     %y.param: %struct_type.b.a.40c = value_param call_param0
// CHECK:STDOUT:     %.loc65: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a.40c] {
// CHECK:STDOUT:       %.loc58.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc58.2: type = converted %.loc58.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc64.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc64.2: type = converted %.loc64.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a.40c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %struct_type.b.a.40c = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref %struct_type.d.c.b36 = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.d.c.b36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc81.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc81.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc81.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc81.1(%y.param: %struct_type.b.a.40c) -> out %return.param: %struct_type.d.c.b36;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc81.2(%x.param: %struct_type.a.b.391) -> out %return.param: %struct_type.c.d.15a [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc81.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc81.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc81.1: ref %struct_type.d.c.b36 = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.d.c.b36 to %.loc81.1 = call %F.ref(%.loc11.3)
// CHECK:STDOUT:   %.loc81.2: ref %struct_type.d.c.b36 = temporary %.loc81.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc81.3: ref %empty_tuple.type = struct_access %.loc81.2, element1
// CHECK:STDOUT:   %.loc81.4: ref %empty_tuple.type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc81.5: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc81.6: init %empty_tuple.type = converted %.loc81.3, %.loc81.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc81.7: ref %empty_struct_type = struct_access %.loc81.2, element0
// CHECK:STDOUT:   %.loc81.8: ref %empty_struct_type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc81.9: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc81.10: init %empty_struct_type = converted %.loc81.7, %.loc81.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc81.11: init %struct_type.c.d.15a to %return.param = struct_init (%.loc81.6, %.loc81.10) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc81.12: init %struct_type.c.d.15a = converted %empty_tuple.type.as.I.impl.F.call, %.loc81.11 [concrete = constants.%struct]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc81.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc81.2)
// CHECK:STDOUT:   return %.loc81.12 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %struct_type.d.c.b36) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ptr.643: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.f29: type = pattern_type %ptr.643 [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.de8: Core.Form = init_form %ptr.31e [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc72.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc72.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.672: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0fc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.20b: %ptr.as.Copy.impl.Op.type.0fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.27c, (%Copy.impl_witness.672) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d34: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cc8: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d34, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.20b, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc72.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc72.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.f29 = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.f29 = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.506 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.506 = return_slot_pattern %return.param_patt, %ptr.loc71 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc71: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %.loc71: Core.Form = init_form %ptr.loc71 [concrete = constants.%.de8]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc61: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = ref_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.643 = value_param call_param1
// CHECK:STDOUT:     %.loc67: type = splice_block %ptr.loc67 [concrete = constants.%ptr.643] {
// CHECK:STDOUT:       %A.ref.loc66: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc67: type = ptr_type %A.ref.loc66 [concrete = constants.%ptr.643]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.643 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.31e = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.31e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc72.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc72.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc72.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc72.1(%self.param: ref %A, %other.param: %ptr.643) -> out %return.param: %ptr.31e;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc72.2(%self.param: ref %B, %other.param: %ptr.27c) -> out %return.param: %ptr.27c [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc72.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc72.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.31e = call %B.as.X.impl.F.bound(%.loc36.2, %.loc41.3)
// CHECK:STDOUT:   %.loc72.1: %ptr.31e = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc72.2: %ptr.31e = converted %B.as.X.impl.F.call, %.loc72.1
// CHECK:STDOUT:   %.loc72.3: ref %C = deref %.loc72.2
// CHECK:STDOUT:   %.loc72.4: ref %B = class_element_access %.loc72.3, element0
// CHECK:STDOUT:   %addr.loc72: %ptr.27c = addr_of %.loc72.4
// CHECK:STDOUT:   %.loc72.5: %ptr.27c = converted %B.as.X.impl.F.call, %addr.loc72
// CHECK:STDOUT:   %impl.elem0: %.cc8 = impl_witness_access constants.%Copy.impl_witness.672, element0 [concrete = constants.%ptr.as.Copy.impl.Op.20b]
// CHECK:STDOUT:   %bound_method.loc72.1: <bound method> = bound_method %.loc72.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72.2: <bound method> = bound_method %.loc72.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.27c = call %bound_method.loc72.2(%.loc72.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc52.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc52.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc52.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc52.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.1ab = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.1ab = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc45: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %A = value_param call_param1
// CHECK:STDOUT:     %A.ref.loc50: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %other: %A = value_binding other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc52.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc52.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc52.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc52.1(%self.param: %A, %other.param: %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc52.2(%self.param: %B, %other.param: %B) [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc52.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc52.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %empty_tuple.type = call %B.as.X.impl.F.bound(%.loc25.3, %.loc30.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ptr.643: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.f29: type = pattern_type %ptr.643 [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.de8: Core.Form = init_form %ptr.31e [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc72.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc72.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.672: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0fc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.20b: %ptr.as.Copy.impl.Op.type.0fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.27c, (%Copy.impl_witness.672) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d34: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cc8: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d34, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.20b, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc72.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc72.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.f29 = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.f29 = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.506 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.506 = return_slot_pattern %return.param_patt, %ptr.loc71 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc71: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %.loc71: Core.Form = init_form %ptr.loc71 [concrete = constants.%.de8]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc61: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = ref_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.643 = value_param call_param1
// CHECK:STDOUT:     %.loc67: type = splice_block %ptr.loc67 [concrete = constants.%ptr.643] {
// CHECK:STDOUT:       %A.ref.loc66: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc67: type = ptr_type %A.ref.loc66 [concrete = constants.%ptr.643]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.643 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.31e = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.31e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc72.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc72.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc72.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc72.1(%self.param: ref %A, %other.param: %ptr.643) -> out %return.param: %ptr.31e;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc72.2(%self.param: ref %B, %other.param: %ptr.27c) -> out %return.param: %ptr.27c [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc72.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc72.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.31e = call %B.as.X.impl.F.bound(%.loc36.2, %.loc41.3)
// CHECK:STDOUT:   %.loc72.1: %ptr.31e = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc72.2: %ptr.31e = converted %B.as.X.impl.F.call, %.loc72.1
// CHECK:STDOUT:   %.loc72.3: ref %C = deref %.loc72.2
// CHECK:STDOUT:   %.loc72.4: ref %B = class_element_access %.loc72.3, element0
// CHECK:STDOUT:   %addr.loc72: %ptr.27c = addr_of %.loc72.4
// CHECK:STDOUT:   %.loc72.5: %ptr.27c = converted %B.as.X.impl.F.call, %addr.loc72
// CHECK:STDOUT:   %impl.elem0: %.cc8 = impl_witness_access constants.%Copy.impl_witness.672, element0 [concrete = constants.%ptr.as.Copy.impl.Op.20b]
// CHECK:STDOUT:   %bound_method.loc72.1: <bound method> = bound_method %.loc72.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72.2: <bound method> = bound_method %.loc72.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.27c = call %bound_method.loc72.2(%.loc72.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer_addr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ptr.643: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.f29: type = pattern_type %ptr.643 [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.de8: Core.Form = init_form %ptr.31e [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc72.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc72.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.672: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0fc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.20b: %ptr.as.Copy.impl.Op.type.0fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.27c, (%Copy.impl_witness.672) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d34: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cc8: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d34, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.20b, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc72.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc72.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.f29 = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.f29 = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.506 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.506 = return_slot_pattern %return.param_patt, %ptr.loc71 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc71: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %.loc71: Core.Form = init_form %ptr.loc71 [concrete = constants.%.de8]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc61: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = ref_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.643 = value_param call_param1
// CHECK:STDOUT:     %.loc67: type = splice_block %ptr.loc67 [concrete = constants.%ptr.643] {
// CHECK:STDOUT:       %A.ref.loc66: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc67: type = ptr_type %A.ref.loc66 [concrete = constants.%ptr.643]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.643 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.31e = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.31e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc72.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc72.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc72.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc72.1(%self.param: ref %A, %other.param: %ptr.643) -> out %return.param: %ptr.31e;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc72.2(%self.param: ref %B, %other.param: %ptr.27c) -> out %return.param: %ptr.27c [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc72.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc72.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.31e = call %B.as.X.impl.F.bound(%.loc36.2, %.loc41.3)
// CHECK:STDOUT:   %.loc72.1: %ptr.31e = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc72.2: %ptr.31e = converted %B.as.X.impl.F.call, %.loc72.1
// CHECK:STDOUT:   %.loc72.3: ref %C = deref %.loc72.2
// CHECK:STDOUT:   %.loc72.4: ref %B = class_element_access %.loc72.3, element0
// CHECK:STDOUT:   %addr.loc72: %ptr.27c = addr_of %.loc72.4
// CHECK:STDOUT:   %.loc72.5: %ptr.27c = converted %B.as.X.impl.F.call, %addr.loc72
// CHECK:STDOUT:   %impl.elem0: %.cc8 = impl_witness_access constants.%Copy.impl_witness.672, element0 [concrete = constants.%ptr.as.Copy.impl.Op.20b]
// CHECK:STDOUT:   %bound_method.loc72.1: <bound method> = bound_method %.loc72.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72.2: <bound method> = bound_method %.loc72.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.27c = call %bound_method.loc72.2(%.loc72.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inheritance_value_conversion_copy_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.7df: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.1f4: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.170a02.1: type = fn_type @A.as.X.impl.F.loc40.1 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.e6cf46.1: %A.as.X.impl.F.type.170a02.1 = struct_value () [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.170a02.2: type = fn_type @A.as.X.impl.F.loc40.2 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.e6cf46.2: %A.as.X.impl.F.type.170a02.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.X.impl: %A.ref as %X.ref {
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc40.1: %A.as.X.impl.F.type.170a02.1 = fn_decl @A.as.X.impl.F.loc40.1 [concrete = constants.%A.as.X.impl.F.e6cf46.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1f4 = return_slot_pattern %return.param_patt, %B.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc39: Core.Form = init_form %B.ref [concrete = constants.%.7df]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc40.2: %A.as.X.impl.F.type.170a02.2 = fn_decl @A.as.X.impl.F.loc40.2 [concrete = constants.%A.as.X.impl.F.e6cf46.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %A.as.X.impl.F.decl.loc40.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc40.1() -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc40.2() -> out %return.param: %A [thunk @A.as.X.impl.%A.as.X.impl.F.decl.loc40.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %A.as.X.impl.F.type.170a02.1 = name_ref F, @A.as.X.impl.%A.as.X.impl.F.decl.loc40.1 [concrete = constants.%A.as.X.impl.F.e6cf46.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc40.1: ref %B = temporary_storage
// CHECK:STDOUT:   %A.as.X.impl.F.call: init %B to %.loc40.1 = call %F.ref()
// CHECK:STDOUT:   %.loc40.2: ref %B = temporary %.loc40.1, %A.as.X.impl.F.call
// CHECK:STDOUT:   %.loc40.3: ref %A = class_element_access %.loc40.2, element0
// CHECK:STDOUT:   %.loc40.4: ref %A = converted %A.as.X.impl.F.call, %.loc40.3
// CHECK:STDOUT:   %.loc40.5: %A = acquire_value %.loc40.4
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc40.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc40.2)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.035: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.14b: type = symbolic_binding_type U, 0, %U.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc47.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc47.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67ca8d.1: <witness> = require_complete_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58dce0.2: <witness> = lookup_impl_witness %U.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.035) [symbolic]
// CHECK:STDOUT:   %.023143.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.3, %U.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594c59.2: %.023143.2 = impl_witness_access %Copy.lookup_impl_witness.58dce0.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdce5c.2: <specific function> = specific_impl_function %impl.elem0.594c59.2, @Copy.WithSelf.Op(%U.035) [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc47.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc27.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc47.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc47.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1] {
// CHECK:STDOUT:     %U.patt: %pattern_type.ce2 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @empty_tuple.type.as.I.impl.F.loc47.1.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @empty_tuple.type.as.I.impl.F.loc47.1.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @empty_tuple.type.as.I.impl.F.loc47.1.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @empty_tuple.type.as.I.impl.F.loc47.1.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc46.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc46: %Copy.type = name_ref U, %U.loc34.2 [symbolic = %U.loc34.1 (constants.%U.035)]
// CHECK:STDOUT:     %U.as_type.loc46: type = facet_access_type %U.ref.loc46 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc46.3: type = converted %U.ref.loc46, %U.as_type.loc46 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc46.4: Core.Form = init_form %.loc46.3 [symbolic = %.loc46.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %.loc37: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc34.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc34.1 (constants.%U.035)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) = value_param call_param0
// CHECK:STDOUT:     %.loc43.1: type = splice_block %.loc43.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)] {
// CHECK:STDOUT:       %U.ref.loc43: %Copy.type = name_ref U, %U.loc34.2 [symbolic = %U.loc34.1 (constants.%U.035)]
// CHECK:STDOUT:       %U.as_type.loc43: type = facet_access_type %U.ref.loc43 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:       %.loc43.2: type = converted %U.ref.loc43, %U.as_type.loc43 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc47.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc47.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc47.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc47.1(%U.loc34.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc34.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc34.1 (constants.%U.035)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc34.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc46.2: Core.Form = init_form %U.binding.as_type [symbolic = %.loc46.2 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.67ca8d.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc34.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58dce0.2)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.loc34.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.3)]
// CHECK:STDOUT:   %.loc49: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %U.loc34.1 [symbolic = %.loc49 (constants.%.023143.2)]
// CHECK:STDOUT:   %impl.elem0.loc49.2: @empty_tuple.type.as.I.impl.F.loc47.1.%.loc49 (%.023143.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc49.2 (constants.%impl.elem0.594c59.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc49.2: <specific function> = specific_impl_function %impl.elem0.loc49.2, @Copy.WithSelf.Op(%U.loc34.1) [symbolic = %specific_impl_fn.loc49.2 (constants.%specific_impl_fn.bdce5c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc49.1: @empty_tuple.type.as.I.impl.F.loc47.1.%.loc49 (%.023143.2) = impl_witness_access constants.%Copy.lookup_impl_witness.58dce0.2, element0 [symbolic = %impl.elem0.loc49.2 (constants.%impl.elem0.594c59.2)]
// CHECK:STDOUT:     %bound_method.loc49.1: <bound method> = bound_method %x.ref, %impl.elem0.loc49.1
// CHECK:STDOUT:     %specific_impl_fn.loc49.1: <specific function> = specific_impl_function %impl.elem0.loc49.1, @Copy.WithSelf.Op(constants.%U.035) [symbolic = %specific_impl_fn.loc49.2 (constants.%specific_impl_fn.bdce5c.2)]
// CHECK:STDOUT:     %bound_method.loc49.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc49.1
// CHECK:STDOUT:     %.loc46.1: ref @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @empty_tuple.type.as.I.impl.F.loc47.1.%U.binding.as_type (%U.binding.as_type.14b) to %.loc46.1 = call %bound_method.loc49.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc47.2(%T.loc10.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc47.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc10.1) [symbolic = %.loc47.2 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet.loc47.2: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc47.2 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc47.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc47.1(%Copy.facet.loc47.2) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc47.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc47.2.%ptr (%ptr.e8f)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc47.2.%ptr (%ptr.e8f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc47.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc47.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.facet.loc47.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc47.2 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc47.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc47.1 [symbolic = %Copy.facet.loc47.2 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc47.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc47.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc47.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc47.2.%ptr (%ptr.e8f) = call %empty_tuple.type.as.I.impl.F.specific_fn.loc47.1(%x.ref)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc47.1(constants.%U.035) {
// CHECK:STDOUT:   %U.loc34.1 => constants.%U.035
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.14b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc46.2 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc47.2(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc10.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc22 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc47.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.loc34.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.binding.as_type => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc46.2 => constants.%.cb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.2e6
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d82
// CHECK:STDOUT:   %.loc49 => constants.%.299
// CHECK:STDOUT:   %impl.elem0.loc49.2 => constants.%impl.elem0.1c7
// CHECK:STDOUT:   %specific_impl_fn.loc49.2 => constants.%specific_impl_fn.366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.035: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.14b: type = symbolic_binding_type U, 0, %U.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc57.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc57.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67ca8d.1: <witness> = require_complete_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58dce0.2: <witness> = lookup_impl_witness %U.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.035) [symbolic]
// CHECK:STDOUT:   %.023143.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.3, %U.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594c59.2: %.023143.2 = impl_witness_access %Copy.lookup_impl_witness.58dce0.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdce5c.2: <specific function> = specific_impl_function %impl.elem0.594c59.2, @Copy.WithSelf.Op(%U.035) [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc57.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc31.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc57.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc57.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.ce2 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @empty_tuple.type.as.I.impl.F.loc57.1.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @empty_tuple.type.as.I.impl.F.loc57.1.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @empty_tuple.type.as.I.impl.F.loc57.1.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @empty_tuple.type.as.I.impl.F.loc57.1.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc56.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc56: %Copy.type = name_ref U, %U.loc44.2 [symbolic = %U.loc44.1 (constants.%U.035)]
// CHECK:STDOUT:     %U.as_type.loc56: type = facet_access_type %U.ref.loc56 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc56.3: type = converted %U.ref.loc56, %U.as_type.loc56 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc56.4: Core.Form = init_form %.loc56.3 [symbolic = %.loc56.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc42.1: type = splice_block %.loc42.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc42.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc42.3: type = converted %.loc42.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %.loc47: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc44.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc44.1 (constants.%U.035)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) = value_param call_param1
// CHECK:STDOUT:     %.loc53.1: type = splice_block %.loc53.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)] {
// CHECK:STDOUT:       %U.ref.loc53: %Copy.type = name_ref U, %U.loc44.2 [symbolic = %U.loc44.1 (constants.%U.035)]
// CHECK:STDOUT:       %U.as_type.loc53: type = facet_access_type %U.ref.loc53 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:       %.loc53.2: type = converted %U.ref.loc53, %U.as_type.loc53 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) = out_param call_param2
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc57.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc57.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc57.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc57.1(%U.loc44.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc44.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc44.1 (constants.%U.035)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc44.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc56.2: Core.Form = init_form %U.binding.as_type [symbolic = %.loc56.2 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.67ca8d.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc44.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58dce0.2)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.loc44.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.3)]
// CHECK:STDOUT:   %.loc59: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %U.loc44.1 [symbolic = %.loc59 (constants.%.023143.2)]
// CHECK:STDOUT:   %impl.elem0.loc59.2: @empty_tuple.type.as.I.impl.F.loc57.1.%.loc59 (%.023143.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc59.2 (constants.%impl.elem0.594c59.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc59.2: <specific function> = specific_impl_function %impl.elem0.loc59.2, @Copy.WithSelf.Op(%U.loc44.1) [symbolic = %specific_impl_fn.loc59.2 (constants.%specific_impl_fn.bdce5c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc59.1: @empty_tuple.type.as.I.impl.F.loc57.1.%.loc59 (%.023143.2) = impl_witness_access constants.%Copy.lookup_impl_witness.58dce0.2, element0 [symbolic = %impl.elem0.loc59.2 (constants.%impl.elem0.594c59.2)]
// CHECK:STDOUT:     %bound_method.loc59.1: <bound method> = bound_method %x.ref, %impl.elem0.loc59.1
// CHECK:STDOUT:     %specific_impl_fn.loc59.1: <specific function> = specific_impl_function %impl.elem0.loc59.1, @Copy.WithSelf.Op(constants.%U.035) [symbolic = %specific_impl_fn.loc59.2 (constants.%specific_impl_fn.bdce5c.2)]
// CHECK:STDOUT:     %bound_method.loc59.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc59.1
// CHECK:STDOUT:     %.loc56.1: ref @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @empty_tuple.type.as.I.impl.F.loc57.1.%U.binding.as_type (%U.binding.as_type.14b) to %.loc56.1 = call %bound_method.loc59.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc57.2(%T.loc14.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc57.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc14.1) [symbolic = %.loc57.2 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet.loc57.2: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc57.2 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc57.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc57.1(%Copy.facet.loc57.2) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc57.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc57.2.%ptr (%ptr.e8f)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc57.2.%ptr (%ptr.e8f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc57.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc57.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:     %Copy.facet.loc57.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc57.2 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc57.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc57.1 [symbolic = %Copy.facet.loc57.2 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc57.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc57.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc57.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %self.ref, %empty_tuple.type.as.I.impl.F.specific_fn.loc57.1
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc57.2.%ptr (%ptr.e8f) = call %bound_method(%self.ref, %x.ref)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc57.1(constants.%U.035) {
// CHECK:STDOUT:   %U.loc44.1 => constants.%U.035
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.14b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc56.2 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc57.2(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc14.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc26 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc57.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.loc44.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.binding.as_type => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc56.2 => constants.%.cb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.2e6
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d82
// CHECK:STDOUT:   %.loc59 => constants.%.299
// CHECK:STDOUT:   %impl.elem0.loc59.2 => constants.%impl.elem0.1c7
// CHECK:STDOUT:   %specific_impl_fn.loc59.2 => constants.%specific_impl_fn.366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.a.1b0: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %.3fc: Core.Form = init_form %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %pattern_type.914: type = pattern_type %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e82c13.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc58.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.5f0cef.1: %empty_tuple.type.as.I.impl.F.type.e82c13.1 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f95: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e82c13.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc58.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.5f0cef.2: %empty_tuple.type.as.I.impl.F.type.e82c13.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.f95 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc32.2 as %I.type {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc58.1: %empty_tuple.type.as.I.impl.F.type.e82c13.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc58.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.5f0cef.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.914 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.914 = return_slot_pattern %return.param_patt, %struct_type.b.a [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc50.2: type = converted %.loc50.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc56.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc56.2: type = converted %.loc56.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete = constants.%struct_type.b.a.1b0]
// CHECK:STDOUT:     %.loc57: Core.Form = init_form %struct_type.b.a [concrete = constants.%.3fc]
// CHECK:STDOUT:     %return.param: ref %struct_type.b.a.1b0 = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.b.a.1b0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc58.2: %empty_tuple.type.as.I.impl.F.type.e82c13.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc58.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.5f0cef.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc58.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc58.1() -> out %return.param: %struct_type.b.a.1b0;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc58.2() -> out %return.param: %struct_type.a.b.f95 [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc58.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.e82c13.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc58.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.5f0cef.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc58.1: ref %struct_type.b.a.1b0 = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.b.a.1b0 to %.loc58.1 = call %F.ref()
// CHECK:STDOUT:   %.loc58.2: ref %struct_type.b.a.1b0 = temporary %.loc58.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc58.3: ref %empty_struct_type = struct_access %.loc58.2, element1
// CHECK:STDOUT:   %.loc58.4: ref %empty_struct_type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc58.5: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58.6: init %empty_struct_type = converted %.loc58.3, %.loc58.5 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58.7: ref %empty_struct_type = struct_access %.loc58.2, element0
// CHECK:STDOUT:   %.loc58.8: ref %empty_struct_type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc58.9: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58.10: init %empty_struct_type = converted %.loc58.7, %.loc58.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58.11: init %struct_type.a.b.f95 to %return.param = struct_init (%.loc58.6, %.loc58.10) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc58.12: init %struct_type.a.b.f95 = converted %empty_tuple.type.as.I.impl.F.call, %.loc58.11 [concrete = constants.%struct]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc58.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc58.2)
// CHECK:STDOUT:   return %.loc58.12 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %struct_type.b.a.1b0) = "no_op";
// CHECK:STDOUT:
