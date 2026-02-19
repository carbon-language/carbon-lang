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

// TODO: Promote this warning to an error by default.
// CHECK:STDERR: todo_fail_import_without_definition.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./without_definition.h:2:13: warning: inline function 'foo' is not defined [CppInteropParseWarning]
// CHECK:STDERR:     2 | inline void foo();
// CHECK:STDERR:       |             ^
import Cpp library "without_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: todo_fail_import_without_definition.carbon:[[@LINE+4]]:11: note: used here [CppInteropParseNote]
  // CHECK:STDERR:    17 |   Cpp.foo();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
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
  let unused r1: i32 = Cpp.WithoutThunk(1);
  let unused r2: i32 = Cpp.ThunkOnArg(1);
  let unused r3: i16 = Cpp.ThunkOnReturn(1);
  let unused r4: i16 = Cpp.ThunkOnBoth(1);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_with_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_without_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc17: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc17: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc20: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc20: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- automatic_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.type: type = cpp_overload_set_type @WithoutThunk.cpp_overload_set [concrete]
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.value: %WithoutThunk.cpp_overload_set.type = cpp_overload_set_value @WithoutThunk.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %WithoutThunk.type: type = fn_type @WithoutThunk [concrete]
// CHECK:STDOUT:   %WithoutThunk: %WithoutThunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnArg.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.value: %ThunkOnArg.cpp_overload_set.type = cpp_overload_set_value @ThunkOnArg.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.type: type = fn_type @ThunkOnArg__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk: %ThunkOnArg__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.882: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c54: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8d9: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c54 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.44d: %ImplicitAs.type.9fb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.882) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.89b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i16, %ImplicitAs.facet.44d) [concrete]
// CHECK:STDOUT:   %.3ce: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.89b, %ImplicitAs.facet.44d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8d9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.a76: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8d9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.576: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.a76 [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e99: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.97f: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.0da: %Int.as.Copy.impl.Op.type.97f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.e99) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d1f: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.eb3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d1f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.0da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.0da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.a48: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.value: %ThunkOnReturn.cpp_overload_set.type = cpp_overload_set_value @ThunkOnReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.type: type = fn_type @ThunkOnReturn__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk: %ThunkOnReturn__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnBoth.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.value: %ThunkOnBoth.cpp_overload_set.type = cpp_overload_set_value @ThunkOnBoth.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.type: type = fn_type @ThunkOnBoth__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk: %ThunkOnBoth__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .WithoutThunk = %WithoutThunk.cpp_overload_set.value
// CHECK:STDOUT:     .ThunkOnArg = %ThunkOnArg.cpp_overload_set.value
// CHECK:STDOUT:     .ThunkOnReturn = %ThunkOnReturn.cpp_overload_set.value
// CHECK:STDOUT:     .ThunkOnBoth = %ThunkOnBoth.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.value: %WithoutThunk.cpp_overload_set.type = cpp_overload_set_value @WithoutThunk.cpp_overload_set [concrete = constants.%WithoutThunk.cpp_overload_set.value]
// CHECK:STDOUT:   %WithoutThunk.decl: %WithoutThunk.type = fn_decl @WithoutThunk [concrete = constants.%WithoutThunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.value: %ThunkOnArg.cpp_overload_set.type = cpp_overload_set_value @ThunkOnArg.cpp_overload_set [concrete = constants.%ThunkOnArg.cpp_overload_set.value]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.decl: %ThunkOnArg__carbon_thunk.type = fn_decl @ThunkOnArg__carbon_thunk [concrete = constants.%ThunkOnArg__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.value: %ThunkOnReturn.cpp_overload_set.type = cpp_overload_set_value @ThunkOnReturn.cpp_overload_set [concrete = constants.%ThunkOnReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.decl: %ThunkOnReturn__carbon_thunk.type = fn_decl @ThunkOnReturn__carbon_thunk [concrete = constants.%ThunkOnReturn__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.value: %ThunkOnBoth.cpp_overload_set.type = cpp_overload_set_value @ThunkOnBoth.cpp_overload_set [concrete = constants.%ThunkOnBoth.cpp_overload_set.value]
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
// CHECK:STDOUT:     %r1.patt: %pattern_type.7ce = value_binding_pattern r1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %WithoutThunk.ref: %WithoutThunk.cpp_overload_set.type = name_ref WithoutThunk, imports.%WithoutThunk.cpp_overload_set.value [concrete = constants.%WithoutThunk.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc13: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc13_41.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709]
// CHECK:STDOUT:   %bound_method.loc13_41.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_41.2(%int_1.loc13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_41.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_41.2: %i32 = converted %int_1.loc13, %.loc13_41.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %WithoutThunk.call: init %i32 = call imports.%WithoutThunk.decl(%.loc13_41.2)
// CHECK:STDOUT:   %.loc13_18: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %WithoutThunk.call
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %WithoutThunk.call, %.loc13_42.1
// CHECK:STDOUT:   %r1: %i32 = value_binding r1, %.loc13_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r2.patt: %pattern_type.7ce = value_binding_pattern r2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnArg.ref: %ThunkOnArg.cpp_overload_set.type = name_ref ThunkOnArg, imports.%ThunkOnArg.cpp_overload_set.value [concrete = constants.%ThunkOnArg.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc14_39.1: %.3ce = impl_witness_access constants.%ImplicitAs.impl_witness.882, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8d9]
// CHECK:STDOUT:   %bound_method.loc14_39.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_39.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e8]
// CHECK:STDOUT:   %specific_fn.loc14_39.1: <specific function> = specific_function %impl.elem0.loc14_39.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.a76]
// CHECK:STDOUT:   %bound_method.loc14_39.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_39.1 [concrete = constants.%bound_method.576]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i16 = call %bound_method.loc14_39.2(%int_1.loc14) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_39.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_39.2: %i16 = converted %int_1.loc14, %.loc14_39.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_39.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc14_39.2: %.eb3 = impl_witness_access constants.%Copy.impl_witness.e99, element0 [concrete = constants.%Int.as.Copy.impl.Op.0da]
// CHECK:STDOUT:   %bound_method.loc14_39.3: <bound method> = bound_method %.loc14_39.2, %impl.elem0.loc14_39.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc14_39.2: <specific function> = specific_function %impl.elem0.loc14_39.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_39.4: <bound method> = bound_method %.loc14_39.2, %specific_fn.loc14_39.2 [concrete = constants.%bound_method.a48]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14: init %i16 = call %bound_method.loc14_39.4(%.loc14_39.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc14_39.4: ref %i16 = temporary %.loc14_39.3, %Int.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %addr.loc14: %ptr.251 = addr_of %.loc14_39.4
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.call: init %i32 = call imports.%ThunkOnArg__carbon_thunk.decl(%addr.loc14)
// CHECK:STDOUT:   %.loc14_18: type = splice_block %i32.loc14 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_40.1: %i32 = value_of_initializer %ThunkOnArg__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_40.2: %i32 = converted %ThunkOnArg__carbon_thunk.call, %.loc14_40.1
// CHECK:STDOUT:   %r2: %i32 = value_binding r2, %.loc14_40.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r3.patt: %pattern_type.2f8 = value_binding_pattern r3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnReturn.ref: %ThunkOnReturn.cpp_overload_set.type = name_ref ThunkOnReturn, imports.%ThunkOnReturn.cpp_overload_set.value [concrete = constants.%ThunkOnReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc15_42.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709]
// CHECK:STDOUT:   %bound_method.loc15_42.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_42.2(%int_1.loc15) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_42.2: %i32 = converted %int_1.loc15, %.loc15_42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_43.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc15: %ptr.251 = addr_of %.loc15_43.1
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnReturn__carbon_thunk.decl(%.loc15_42.2, %addr.loc15)
// CHECK:STDOUT:   %.loc15_43.2: init %i16 to %.loc15_43.1 = mark_in_place_init %ThunkOnReturn__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_18: type = splice_block %i16.loc15 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16.loc15: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16.loc15: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_43.3: ref %i16 = temporary %.loc15_43.1, %.loc15_43.2
// CHECK:STDOUT:   %.loc15_43.4: %i16 = acquire_value %.loc15_43.3
// CHECK:STDOUT:   %r3: %i16 = value_binding r3, %.loc15_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r4.patt: %pattern_type.2f8 = value_binding_pattern r4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnBoth.ref: %ThunkOnBoth.cpp_overload_set.type = name_ref ThunkOnBoth, imports.%ThunkOnBoth.cpp_overload_set.value [concrete = constants.%ThunkOnBoth.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16_40.1: %.3ce = impl_witness_access constants.%ImplicitAs.impl_witness.882, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8d9]
// CHECK:STDOUT:   %bound_method.loc16_40.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7e8]
// CHECK:STDOUT:   %specific_fn.loc16_40.1: <specific function> = specific_function %impl.elem0.loc16_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.a76]
// CHECK:STDOUT:   %bound_method.loc16_40.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_40.1 [concrete = constants.%bound_method.576]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i16 = call %bound_method.loc16_40.2(%int_1.loc16) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_40.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_40.2: %i16 = converted %int_1.loc16, %.loc16_40.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_40.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc16_40.2: %.eb3 = impl_witness_access constants.%Copy.impl_witness.e99, element0 [concrete = constants.%Int.as.Copy.impl.Op.0da]
// CHECK:STDOUT:   %bound_method.loc16_40.3: <bound method> = bound_method %.loc16_40.2, %impl.elem0.loc16_40.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc16_40.2: <specific function> = specific_function %impl.elem0.loc16_40.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_40.4: <bound method> = bound_method %.loc16_40.2, %specific_fn.loc16_40.2 [concrete = constants.%bound_method.a48]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i16 = call %bound_method.loc16_40.4(%.loc16_40.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_40.4: ref %i16 = temporary %.loc16_40.3, %Int.as.Copy.impl.Op.call.loc16
// CHECK:STDOUT:   %addr.loc16_41.1: %ptr.251 = addr_of %.loc16_40.4
// CHECK:STDOUT:   %.loc16_41.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc16_41.2: %ptr.251 = addr_of %.loc16_41.1
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnBoth__carbon_thunk.decl(%addr.loc16_41.1, %addr.loc16_41.2)
// CHECK:STDOUT:   %.loc16_41.2: init %i16 to %.loc16_41.1 = mark_in_place_init %ThunkOnBoth__carbon_thunk.call
// CHECK:STDOUT:   %.loc16_18: type = splice_block %i16.loc16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16.loc16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16.loc16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_41.3: ref %i16 = temporary %.loc16_41.1, %.loc16_41.2
// CHECK:STDOUT:   %.loc16_41.4: %i16 = acquire_value %.loc16_41.3
// CHECK:STDOUT:   %r4: %i16 = value_binding r4, %.loc16_41.4
// CHECK:STDOUT:   %DestroyOp.bound.loc16_41: <bound method> = bound_method %.loc16_41.3, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc16_41: init %empty_tuple.type = call %DestroyOp.bound.loc16_41(%.loc16_41.3)
// CHECK:STDOUT:   %DestroyOp.bound.loc16_40: <bound method> = bound_method %.loc16_40.4, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc16_40: init %empty_tuple.type = call %DestroyOp.bound.loc16_40(%.loc16_40.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc15: <bound method> = bound_method %.loc15_43.3, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc15: init %empty_tuple.type = call %DestroyOp.bound.loc15(%.loc15_43.3)
// CHECK:STDOUT:   %DestroyOp.bound.loc14: <bound method> = bound_method %.loc14_39.4, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc14: init %empty_tuple.type = call %DestroyOp.bound.loc14(%.loc14_39.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
