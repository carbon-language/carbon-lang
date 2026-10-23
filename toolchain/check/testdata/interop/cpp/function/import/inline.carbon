// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/inline.carbon

// ============================================================================
// With definition
// ============================================================================

// --- with_definition.h

inline void inline_with_def() {}

// --- import_with_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.inline_with_def();
  //@dump-sem-ir-end
}

// ============================================================================
// Without definition
// ============================================================================

// --- without_definition.h

// TODO: Promote this warning to an error by default.
// CHECK:STDERR: ./without_definition.h:[[@LINE+1]]:13: warning: inline function 'inline_without_def' is not defined [CppInteropParseWarning]
inline void inline_without_def();

// --- todo_fail_import_without_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "without_definition.h";

fn MyF() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: todo_fail_import_without_definition.carbon:[[@LINE+2]]:26: note: used here [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.inline_without_def();

  // Don't error on repeated calls.
  Cpp.inline_without_def();
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %inline_with_def.cpp_overload_set.type: type = cpp_overload_set_type @inline_with_def.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inline_with_def.cpp_overload_set.value: %inline_with_def.cpp_overload_set.type = cpp_overload_set_value @inline_with_def.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inline_with_def.type: type = fn_type @inline_with_def [concrete]
// CHECK:STDOUT:   %inline_with_def: %inline_with_def.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .inline_with_def = %inline_with_def.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inline_with_def.cpp_overload_set.value: %inline_with_def.cpp_overload_set.type = cpp_overload_set_value @inline_with_def.cpp_overload_set [concrete = constants.%inline_with_def.cpp_overload_set.value]
// CHECK:STDOUT:   %inline_with_def.decl: %inline_with_def.type = fn_decl @inline_with_def [concrete = constants.%inline_with_def] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %inline_with_def.ref: %inline_with_def.cpp_overload_set.type = name_ref inline_with_def, imports.%inline_with_def.cpp_overload_set.value [concrete = constants.%inline_with_def.cpp_overload_set.value]
// CHECK:STDOUT:   %inline_with_def.call: init %empty_tuple.type = call imports.%inline_with_def.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_without_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %inline_without_def.cpp_overload_set.type: type = cpp_overload_set_type @inline_without_def.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inline_without_def.cpp_overload_set.value: %inline_without_def.cpp_overload_set.type = cpp_overload_set_value @inline_without_def.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inline_without_def.type: type = fn_type @inline_without_def [concrete]
// CHECK:STDOUT:   %inline_without_def: %inline_without_def.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .inline_without_def = %inline_without_def.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inline_without_def.cpp_overload_set.value: %inline_without_def.cpp_overload_set.type = cpp_overload_set_value @inline_without_def.cpp_overload_set [concrete = constants.%inline_without_def.cpp_overload_set.value]
// CHECK:STDOUT:   %inline_without_def.decl: %inline_without_def.type = fn_decl @inline_without_def [concrete = constants.%inline_without_def] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %inline_without_def.ref.loc10: %inline_without_def.cpp_overload_set.type = name_ref inline_without_def, imports.%inline_without_def.cpp_overload_set.value [concrete = constants.%inline_without_def.cpp_overload_set.value]
// CHECK:STDOUT:   %inline_without_def.call.loc10: init %empty_tuple.type = call imports.%inline_without_def.decl()
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %inline_without_def.ref.loc13: %inline_without_def.cpp_overload_set.type = name_ref inline_without_def, imports.%inline_without_def.cpp_overload_set.value [concrete = constants.%inline_without_def.cpp_overload_set.value]
// CHECK:STDOUT:   %inline_without_def.call.loc13: init %empty_tuple.type = call imports.%inline_without_def.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- automatic_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %r1.patt: %pattern_type.d94 = value_binding_pattern r1 [concrete]
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.type: type = cpp_overload_set_type @WithoutThunk.cpp_overload_set [concrete]
// CHECK:STDOUT:   %WithoutThunk.cpp_overload_set.value: %WithoutThunk.cpp_overload_set.type = cpp_overload_set_value @WithoutThunk.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %WithoutThunk.type: type = fn_type @WithoutThunk [concrete]
// CHECK:STDOUT:   %WithoutThunk: %WithoutThunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8 [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %r2.patt: %pattern_type.d94 = value_binding_pattern r2 [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnArg.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.value: %ThunkOnArg.cpp_overload_set.type = cpp_overload_set_value @ThunkOnArg.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9bc: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.e14: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.type: type = fn_type @ThunkOnArg__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk: %ThunkOnArg__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.887: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.610: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9e0: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e71: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9e0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b42: %ImplicitAs.type.887 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.610) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.14b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet.b42) [concrete]
// CHECK:STDOUT:   %.425: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.14b, %ImplicitAs.facet.b42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d9f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e71 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.91f: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e71, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.f03: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.91f [concrete]
// CHECK:STDOUT:   %int_1.6e1: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.21d: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.6f7: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5fe: %Int.as.Copy.impl.Op.type.6f7 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.21d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.64d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.376: type = fn_type_with_self_type %Copy.WithSelf.Op.type.64d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.6e1, %Int.as.Copy.impl.Op.5fe [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.5fe, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.96e: <bound method> = bound_method %int_1.6e1, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.c52: ref %i16 = temporary invalid, %int_1.6e1 [concrete]
// CHECK:STDOUT:   %addr: %ptr.e14 = addr_of %.c52 [concrete]
// CHECK:STDOUT:   %r3.patt: %pattern_type.9bc = value_binding_pattern r3 [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnReturn.cpp_overload_set.value: %ThunkOnReturn.cpp_overload_set.type = cpp_overload_set_value @ThunkOnReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.type: type = fn_type @ThunkOnReturn__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk: %ThunkOnReturn__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %r4.patt: %pattern_type.9bc = value_binding_pattern r4 [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.type: type = cpp_overload_set_type @ThunkOnBoth.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnBoth.cpp_overload_set.value: %ThunkOnBoth.cpp_overload_set.type = cpp_overload_set_value @ThunkOnBoth.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.type: type = fn_type @ThunkOnBoth__carbon_thunk [concrete]
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk: %ThunkOnBoth__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_41.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c5e: %SubobjectDestroy.type = facet_value %i16, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dff: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c5e) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ThunkOnArg.cpp_overload_set.value: %ThunkOnArg.cpp_overload_set.type = cpp_overload_set_value @ThunkOnArg.cpp_overload_set [concrete = constants.%ThunkOnArg.cpp_overload_set.value]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.decl: %ThunkOnArg__carbon_thunk.type = fn_decl @ThunkOnArg__carbon_thunk [concrete = constants.%ThunkOnArg__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %WithoutThunk.ref: %WithoutThunk.cpp_overload_set.type = name_ref WithoutThunk, imports.%WithoutThunk.cpp_overload_set.value [concrete = constants.%WithoutThunk.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc13: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_41.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8]
// CHECK:STDOUT:   %bound_method.loc13_41.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_41.2(%int_1.loc13) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_41.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_41.2: %i32 = converted %int_1.loc13, %.loc13_41.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %WithoutThunk.call: init %i32 = call imports.%WithoutThunk.decl(%.loc13_41.2)
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %WithoutThunk.call
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %WithoutThunk.call, %.loc13_42.1
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %r1: %i32 = wrapper_binding r1, %.loc13_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r1.patt: %pattern_type.d94 = value_binding_pattern r1 [concrete = constants.%r1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnArg.ref: %ThunkOnArg.cpp_overload_set.type = name_ref ThunkOnArg, imports.%ThunkOnArg.cpp_overload_set.value [concrete = constants.%ThunkOnArg.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc14_39.1: %.425 = impl_witness_access constants.%ImplicitAs.impl_witness.610, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e71]
// CHECK:STDOUT:   %bound_method.loc14_39.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_39.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d9f]
// CHECK:STDOUT:   %specific_fn.loc14_39.1: <specific function> = specific_function %impl.elem0.loc14_39.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.91f]
// CHECK:STDOUT:   %bound_method.loc14_39.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_39.1 [concrete = constants.%bound_method.f03]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i16 = call %bound_method.loc14_39.2(%int_1.loc14) [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc14_39.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc14_39.2: %i16 = converted %int_1.loc14, %.loc14_39.1 [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %impl.elem0.loc14_39.2: %.376 = impl_witness_access constants.%Copy.impl_witness.21d, element0 [concrete = constants.%Int.as.Copy.impl.Op.5fe]
// CHECK:STDOUT:   %bound_method.loc14_39.3: <bound method> = bound_method %.loc14_39.2, %impl.elem0.loc14_39.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc14_39.2: <specific function> = specific_function %impl.elem0.loc14_39.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_39.4: <bound method> = bound_method %.loc14_39.2, %specific_fn.loc14_39.2 [concrete = constants.%bound_method.96e]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14: init %i16 = call %bound_method.loc14_39.4(%.loc14_39.2) [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc14_39.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc14_39.4: ref %i16 = temporary %.loc14_39.3, %Int.as.Copy.impl.Op.call.loc14 [concrete = constants.%.c52]
// CHECK:STDOUT:   %addr.loc14: %ptr.e14 = addr_of %.loc14_39.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %ThunkOnArg__carbon_thunk.call: init %i32 = call imports.%ThunkOnArg__carbon_thunk.decl(%addr.loc14)
// CHECK:STDOUT:   %.loc14_40.1: %i32 = value_of_initializer %ThunkOnArg__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_40.2: %i32 = converted %ThunkOnArg__carbon_thunk.call, %.loc14_40.1
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %r2: %i32 = wrapper_binding r2, %.loc14_40.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r2.patt: %pattern_type.d94 = value_binding_pattern r2 [concrete = constants.%r2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnReturn.ref: %ThunkOnReturn.cpp_overload_set.type = name_ref ThunkOnReturn, imports.%ThunkOnReturn.cpp_overload_set.value [concrete = constants.%ThunkOnReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc15: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc15_42.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8]
// CHECK:STDOUT:   %bound_method.loc15_42.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_42.2(%int_1.loc15) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc15_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc15_42.2: %i32 = converted %int_1.loc15, %.loc15_42.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc15_43.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc15: %ptr.e14 = addr_of %.loc15_43.1
// CHECK:STDOUT:   %ThunkOnReturn__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnReturn__carbon_thunk.decl(%.loc15_42.2, %addr.loc15)
// CHECK:STDOUT:   %.loc15_43.2: init %i16 to %.loc15_43.1 = mark_in_place_init %ThunkOnReturn__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_43.3: ref %i16 = temporary %.loc15_43.1, %.loc15_43.2
// CHECK:STDOUT:   %.loc15_43.4: %i16 = acquire_value %.loc15_43.3
// CHECK:STDOUT:   %i16.loc15: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %r3: %i16 = wrapper_binding r3, %.loc15_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r3.patt: %pattern_type.9bc = value_binding_pattern r3 [concrete = constants.%r3.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ThunkOnBoth.ref: %ThunkOnBoth.cpp_overload_set.type = name_ref ThunkOnBoth, imports.%ThunkOnBoth.cpp_overload_set.value [concrete = constants.%ThunkOnBoth.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16_40.1: %.425 = impl_witness_access constants.%ImplicitAs.impl_witness.610, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e71]
// CHECK:STDOUT:   %bound_method.loc16_40.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d9f]
// CHECK:STDOUT:   %specific_fn.loc16_40.1: <specific function> = specific_function %impl.elem0.loc16_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.91f]
// CHECK:STDOUT:   %bound_method.loc16_40.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_40.1 [concrete = constants.%bound_method.f03]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i16 = call %bound_method.loc16_40.2(%int_1.loc16) [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc16_40.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc16_40.2: %i16 = converted %int_1.loc16, %.loc16_40.1 [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %impl.elem0.loc16_40.2: %.376 = impl_witness_access constants.%Copy.impl_witness.21d, element0 [concrete = constants.%Int.as.Copy.impl.Op.5fe]
// CHECK:STDOUT:   %bound_method.loc16_40.3: <bound method> = bound_method %.loc16_40.2, %impl.elem0.loc16_40.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc16_40.2: <specific function> = specific_function %impl.elem0.loc16_40.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_40.4: <bound method> = bound_method %.loc16_40.2, %specific_fn.loc16_40.2 [concrete = constants.%bound_method.96e]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i16 = call %bound_method.loc16_40.4(%.loc16_40.2) [concrete = constants.%int_1.6e1]
// CHECK:STDOUT:   %.loc16_40.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc16_40.4: ref %i16 = temporary %.loc16_40.3, %Int.as.Copy.impl.Op.call.loc16 [concrete = constants.%.c52]
// CHECK:STDOUT:   %addr.loc16_41.1: %ptr.e14 = addr_of %.loc16_40.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %.loc16_41.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr.loc16_41.2: %ptr.e14 = addr_of %.loc16_41.1
// CHECK:STDOUT:   %ThunkOnBoth__carbon_thunk.call: init %empty_tuple.type = call imports.%ThunkOnBoth__carbon_thunk.decl(%addr.loc16_41.1, %addr.loc16_41.2)
// CHECK:STDOUT:   %.loc16_41.2: init %i16 to %.loc16_41.1 = mark_in_place_init %ThunkOnBoth__carbon_thunk.call
// CHECK:STDOUT:   %.loc16_41.3: ref %i16 = temporary %.loc16_41.1, %.loc16_41.2
// CHECK:STDOUT:   %.loc16_41.4: %i16 = acquire_value %.loc16_41.3
// CHECK:STDOUT:   %i16.loc16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %r4: %i16 = wrapper_binding r4, %.loc16_41.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r4.patt: %pattern_type.9bc = value_binding_pattern r4 [concrete = constants.%r4.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc16_41.1: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc16_41.5: %facet_type = converted constants.%i16, %facet_value.loc16_41.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc16_41.2: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc16_41.6: %facet_type = converted constants.%i16, %facet_value.loc16_41.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_41: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_41.3)
// CHECK:STDOUT:   %facet_value.loc16_40.1: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc16_40.5: %facet_type = converted constants.%i16, %facet_value.loc16_40.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc16_40.2: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc16_40.6: %facet_type = converted constants.%i16, %facet_value.loc16_40.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_40: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_40.4)
// CHECK:STDOUT:   %facet_value.loc15_43.1: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc15_43.5: %facet_type = converted constants.%i16, %facet_value.loc15_43.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc15_43.2: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc15_43.6: %facet_type = converted constants.%i16, %facet_value.loc15_43.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc15_43.3)
// CHECK:STDOUT:   %facet_value.loc14_39.1: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc14_39.5: %facet_type = converted constants.%i16, %facet_value.loc14_39.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc14_39.2: %facet_type = facet_value constants.%i16, (constants.%DestroyImpl.impl_witness.dff, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc14_39.6: %facet_type = converted constants.%i16, %facet_value.loc14_39.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc14_39.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_41.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_41.2(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
