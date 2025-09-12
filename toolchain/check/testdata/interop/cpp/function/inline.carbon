// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/inline.carbon

// ============================================================================
// With definition
// ============================================================================

// --- with_definition.h

inline void foo() {}

// --- import_with_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Without definition
// ============================================================================

// --- without_definition.h

inline void foo();

// --- todo_fail_import_without_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "without_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  // TODO: Error on using an inline function without definition.
  Cpp.foo();

  // Don't error on repeated calls.
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// automatic return type
// ============================================================================

// --- automatic_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
inline auto WithoutThunk(int) -> auto { return 0; }
inline auto ThunkOnArg(short) -> auto { return 0; }
inline auto ThunkOnReturn(int) -> auto { short x = 0; return x; }
inline auto ThunkOnBoth(short) -> auto { short x = 0; return x; }
''';

fn MyF() {
  //@dump-sem-ir-begin
  let r1: i32 = Cpp.WithoutThunk(1);
  let r2: i32 = Cpp.ThunkOnArg(1);
  let r3: i16 = Cpp.ThunkOnReturn(1);
  let r4: i16 = Cpp.ThunkOnBoth(1);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_with_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_without_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc9: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call.loc9: init %empty_tuple.type = call %foo.ref.loc9()
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc12: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call.loc12: init %empty_tuple.type = call %foo.ref.loc12()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- automatic_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %WithoutThunk.type: type = fn_type @WithoutThunk [concrete]
// CHECK:STDOUT:   %WithoutThunk: %WithoutThunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e4d: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.e4d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.ab7: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b59: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.ab7 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ThunkOnArg.type: type = fn_type @ThunkOnArg [concrete]
// CHECK:STDOUT:   %ThunkOnArg: %ThunkOnArg.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.type: type = fn_type @ThunkOnArg__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk: %ThunkOnArg__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.fa6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i16) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.346: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.346 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fa2: %ImplicitAs.type.9fb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %.8df: type = fn_type_with_self_type %ImplicitAs.Convert.type.fa6, %ImplicitAs.facet.fa2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.34d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e3b: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.dd7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e3b [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.efc: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.type: type = fn_type @ThunkOnReturn [concrete]
// CHECK:STDOUT:   %ThunkOnReturn: %ThunkOnReturn.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.type: type = fn_type @ThunkOnReturn__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk: %ThunkOnReturn__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.type: type = fn_type @ThunkOnBoth [concrete]
// CHECK:STDOUT:   %ThunkOnBoth: %ThunkOnBoth.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.type: type = fn_type @ThunkOnBoth__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk: %ThunkOnBoth__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .WithoutThunk = %WithoutThunk.decl
// CHECK:STDOUT:     .ThunkOnArg = %ThunkOnArg.decl
// CHECK:STDOUT:     .ThunkOnReturn = %ThunkOnReturn.decl
// CHECK:STDOUT:     .ThunkOnBoth = %ThunkOnBoth.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithoutThunk.decl: %WithoutThunk.type = fn_decl @WithoutThunk [concrete = constants.%WithoutThunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ThunkOnArg.decl: %ThunkOnArg.type = fn_decl @ThunkOnArg [concrete = constants.%ThunkOnArg] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.decl: %ThunkOnArg__carbon_thunk.type = fn_decl @ThunkOnArg__carbon_thunk [concrete = constants.%ThunkOnArg__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.decl: %ThunkOnReturn.type = fn_decl @ThunkOnReturn [concrete = constants.%ThunkOnReturn] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.decl: %ThunkOnReturn__carbon_thunk.type = fn_decl @ThunkOnReturn__carbon_thunk [concrete = constants.%ThunkOnReturn__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnBoth.decl: %ThunkOnBoth.type = fn_decl @ThunkOnBoth [concrete = constants.%ThunkOnBoth] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.decl: %ThunkOnBoth__carbon_thunk.type = fn_decl @ThunkOnBoth__carbon_thunk [concrete = constants.%ThunkOnBoth__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r1.patt: %pattern_type.7ce = binding_pattern r1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %WithoutThunk.ref: %WithoutThunk.type = name_ref WithoutThunk, imports.%WithoutThunk.decl [concrete = constants.%WithoutThunk]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc13: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.ab7]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_1.loc13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_1.loc13, %.loc13_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %WithoutThunk.call: init %i32 = call %WithoutThunk.ref(%.loc13_34.2)
// CHECK:STDOUT:   %.loc13_11: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_35.1: %i32 = value_of_initializer %WithoutThunk.call
// CHECK:STDOUT:   %.loc13_35.2: %i32 = converted %WithoutThunk.call, %.loc13_35.1
// CHECK:STDOUT:   %r1: %i32 = bind_name r1, %.loc13_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r2.patt: %pattern_type.7ce = binding_pattern r2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnArg.ref: %ThunkOnArg.type = name_ref ThunkOnArg, imports.%ThunkOnArg.decl [concrete = constants.%ThunkOnArg]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc14_32.1: %.8df = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1]
// CHECK:STDOUT:   %bound_method.loc14_32.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_32.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.34d]
// CHECK:STDOUT:   %specific_fn.loc14_32.1: <specific function> = specific_function %impl.elem0.loc14_32.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e3b]
// CHECK:STDOUT:   %bound_method.loc14_32.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_32.1 [concrete = constants.%bound_method.dd7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i16 = call %bound_method.loc14_32.2(%int_1.loc14) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_32.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_32.2: %i16 = converted %int_1.loc14, %.loc14_32.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc14_32.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc14_32.3: <bound method> = bound_method %.loc14_32.2, %impl.elem0.loc14_32.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc14_32.2: <specific function> = specific_function %impl.elem0.loc14_32.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_32.4: <bound method> = bound_method %.loc14_32.2, %specific_fn.loc14_32.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14: init %i16 = call %bound_method.loc14_32.4(%.loc14_32.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_32.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc14_32.4: ref %i16 = temporary %.loc14_32.3, %Int.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %addr.loc14_33: %ptr.251 = addr_of %.loc14_32.4
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.call: init %i32 = call imports.%ThunkOnArg__carbon_thunk.decl(%addr.loc14_33)
// CHECK:STDOUT:   %.loc14_11: type = splice_block %i32.loc14 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_33.1: %i32 = value_of_initializer %ThunkOnArg__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_33.2: %i32 = converted %ThunkOnArg__carbon_thunk.call, %.loc14_33.1
// CHECK:STDOUT:   %r2: %i32 = bind_name r2, %.loc14_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r3.patt: %pattern_type.2f8 = binding_pattern r3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnReturn.ref: %ThunkOnReturn.type = name_ref ThunkOnReturn, imports.%ThunkOnReturn.decl [concrete = constants.%ThunkOnReturn]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc15_35.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.ab7]
// CHECK:STDOUT:   %bound_method.loc15_35.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_35.2(%int_1.loc15) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_35.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_35.2: %i32 = converted %int_1.loc15, %.loc15_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_36.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc15: %ptr.251 = addr_of %.loc15_36.1
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnReturn__carbon_thunk.decl(%.loc15_35.2, %addr.loc15)
// CHECK:STDOUT:   %.loc15_36.2: init %i16 = in_place_init %ThunkOnReturn__carbon_thunk.call, %.loc15_36.1
// CHECK:STDOUT:   %.loc15_11: type = splice_block %i16.loc15 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16.loc15: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16.loc15: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_36.3: %i16 = value_of_initializer %.loc15_36.2
// CHECK:STDOUT:   %.loc15_36.4: %i16 = converted %.loc15_36.2, %.loc15_36.3
// CHECK:STDOUT:   %r3: %i16 = bind_name r3, %.loc15_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r4.patt: %pattern_type.2f8 = binding_pattern r4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnBoth.ref: %ThunkOnBoth.type = name_ref ThunkOnBoth, imports.%ThunkOnBoth.decl [concrete = constants.%ThunkOnBoth]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16_33.1: %.8df = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1]
// CHECK:STDOUT:   %bound_method.loc16_33.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_33.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.34d]
// CHECK:STDOUT:   %specific_fn.loc16_33.1: <specific function> = specific_function %impl.elem0.loc16_33.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e3b]
// CHECK:STDOUT:   %bound_method.loc16_33.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_33.1 [concrete = constants.%bound_method.dd7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i16 = call %bound_method.loc16_33.2(%int_1.loc16) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_33.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_33.2: %i16 = converted %int_1.loc16, %.loc16_33.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc16_33.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc16_33.3: <bound method> = bound_method %.loc16_33.2, %impl.elem0.loc16_33.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc16_33.2: <specific function> = specific_function %impl.elem0.loc16_33.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_33.4: <bound method> = bound_method %.loc16_33.2, %specific_fn.loc16_33.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i16 = call %bound_method.loc16_33.4(%.loc16_33.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_33.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc16_33.4: ref %i16 = temporary %.loc16_33.3, %Int.as.Copy.impl.Op.call.loc16
// CHECK:STDOUT:   %addr.loc16_34.1: %ptr.251 = addr_of %.loc16_33.4
// CHECK:STDOUT:   %.loc16_34.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc16_34.2: %ptr.251 = addr_of %.loc16_34.1
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnBoth__carbon_thunk.decl(%addr.loc16_34.1, %addr.loc16_34.2)
// CHECK:STDOUT:   %.loc16_34.2: init %i16 = in_place_init %ThunkOnBoth__carbon_thunk.call, %.loc16_34.1
// CHECK:STDOUT:   %.loc16_11: type = splice_block %i16.loc16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16.loc16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16.loc16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_34.3: %i16 = value_of_initializer %.loc16_34.2
// CHECK:STDOUT:   %.loc16_34.4: %i16 = converted %.loc16_34.2, %.loc16_34.3
// CHECK:STDOUT:   %r4: %i16 = bind_name r4, %.loc16_34.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound.loc16: <bound method> = bound_method %.loc16_33.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc16_33.5: <bound method> = bound_method %.loc16_33.4, %Int.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc16_33: %ptr.251 = addr_of %.loc16_33.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call.loc16: init %empty_tuple.type = call %bound_method.loc16_33.5(%addr.loc16_33)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound.loc14: <bound method> = bound_method %.loc14_32.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14_32.5: <bound method> = bound_method %.loc14_32.4, %Int.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc14_32: %ptr.251 = addr_of %.loc14_32.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call.loc14: init %empty_tuple.type = call %bound_method.loc14_32.5(%addr.loc14_32)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
