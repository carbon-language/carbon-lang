// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-pc-windows-msvc --clang-arg=-fno-PIE
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/builtins.llp64.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/builtins.llp64.carbon

// --- supported_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline
'''
class LongResult {};
auto PassLong(long x) -> LongResult;

class IntResult {};
auto PassLong(int x) -> IntResult;
''';

fn F() {
  //@dump-sem-ir-begin
  let cpp_long_long : Cpp.long_long = 1 as i64;
  let carbon_long_long: i64 = cpp_long_long;

  let cpp_long: Cpp.long = 1 as i32;
  let cpp_long_result: Cpp.LongResult = Cpp.PassLong(cpp_long);

  let cpp_compat_long: Core.CppCompat.Long32 = cpp_long;
  let cpp_compat_long_long_result: Cpp.LongResult = Cpp.PassLong(cpp_long);

  let carbon_i32: i32 = cpp_long;
  let carbon_i32_result: Cpp.IntResult = Cpp.PassLong(carbon_i32);

  let cpp_unsigned_long_long : Cpp.unsigned_long_long = 1 as u64;
  let carbon_unsigned_long_long: u64 = cpp_unsigned_long_long;
  //@dump-sem-ir-end
}

// --- fail_todo_unsupported_types.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+7]]:27: error: semantics TODO: `Unsupported: builtin type: unsigned long` [SemanticsTodo]
  // CHECK:STDERR:   let cpp_unsigned_long : Cpp.unsigned_long = 1 as u64;
  // CHECK:STDERR:                           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_unsupported_types.carbon:[[@LINE+4]]:27: note: in `Cpp` name lookup for `unsigned_long` [InCppNameLookup]
  // CHECK:STDERR:   let cpp_unsigned_long : Cpp.unsigned_long = 1 as u64;
  // CHECK:STDERR:                           ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let cpp_unsigned_long : Cpp.unsigned_long = 1 as u64;
  let carbon_unsigned_long: u64 = cpp_unsigned_long;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- supported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.bbb: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %To.586: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%To.586) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c40: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl.863(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4f0: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.108: %Core.IntLiteral.as.As.impl.Convert.type.4f0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.2cd: %As.type.bbb = facet_value Core.IntLiteral, (%As.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %.277: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet.2cd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.373: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.108 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.83e: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.108, @Core.IntLiteral.as.As.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c84: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.83e [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %As.impl_witness.bf0: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl.863(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.11b: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.863(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.035: %Core.IntLiteral.as.As.impl.Convert.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.1f3: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.bf0) [concrete]
// CHECK:STDOUT:   %.edb: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet.1f3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.17b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.035 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.d03: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.035, @Core.IntLiteral.as.As.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6b2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.d03 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.c65: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9db: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bd5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.687: %ImplicitAs.type.c65 = facet_value %i32, (%ImplicitAs.impl_witness.9db) [concrete]
// CHECK:STDOUT:   %.1bd: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.687 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5d2, %i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.cd9: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.305: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.9d9430.1: type = fn_type @PassLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.58a8f3.1: %PassLong__carbon_thunk.type.9d9430.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.90c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a39: %ImplicitAs.type.d14 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.c39) [concrete]
// CHECK:STDOUT:   %.261: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.a39 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntResult: type = class_type @IntResult [concrete]
// CHECK:STDOUT:   %pattern_type.454: type = pattern_type %IntResult [concrete]
// CHECK:STDOUT:   %ptr.3cb: type = ptr_type %IntResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.9d9430.2: type = fn_type @PassLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.58a8f3.2: %PassLong__carbon_thunk.type.9d9430.2 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %As.type.465: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.56b: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.38a(%To.586) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.02d: %Core.IntLiteral.as.As.impl.Convert.type.56b = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ed2d: <witness> = impl_witness imports.%As.impl_witness_table.a7d, @Core.IntLiteral.as.As.impl.38a(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.422: type = fn_type @Core.IntLiteral.as.As.impl.Convert.2, @Core.IntLiteral.as.As.impl.38a(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a09: %Core.IntLiteral.as.As.impl.Convert.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.bf5: %As.type.465 = facet_value Core.IntLiteral, (%As.impl_witness.ed2d) [concrete]
// CHECK:STDOUT:   %.5b8: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet.bf5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.eb8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.009: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a09, @Core.IntLiteral.as.As.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.bb9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.009 [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.743: %type_where = facet_value %IntResult, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.3b7: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.743) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.515: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.3b7 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.1e3: %type_where = facet_value %LongResult, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6c9: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.1e3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.418: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6c9 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .CppCompat = %CppCompat.c59
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.1
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     .LongResult = %LongResult.decl
// CHECK:STDOUT:     .PassLong = %PassLong.cpp_overload_set.value
// CHECK:STDOUT:     .IntResult = %IntResult.decl
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.863.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.863.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl.863 [concrete]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.c59: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .Long32 = %Core.Long32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Long32: type = import_ref Core//prelude/types/cpp/int, Long32, loaded [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e63: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bd5 = impl_witness_table (%Core.import_ref.e63), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLong__carbon_thunk.decl.d16229.1: %PassLong__carbon_thunk.type.9d9430.1 = fn_decl @PassLong__carbon_thunk.1 [concrete = constants.%PassLong__carbon_thunk.58a8f3.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1a1: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.90c = impl_witness_table (%Core.import_ref.1a1), @Cpp.long.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %IntResult.decl: type = class_decl @IntResult [concrete = constants.%IntResult] {} {}
// CHECK:STDOUT:   %PassLong__carbon_thunk.decl.d16229.2: %PassLong__carbon_thunk.type.9d9430.2 = fn_decl @PassLong__carbon_thunk.2 [concrete = constants.%PassLong__carbon_thunk.58a8f3.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.As.impl.38a.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.56b) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.38a.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.02d)]
// CHECK:STDOUT:   %As.impl_witness_table.a7d = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.As.impl.38a [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.95b = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %i64.loc15: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc15: %.277 = impl_witness_access constants.%As.impl_witness.c40, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.108]
// CHECK:STDOUT:   %bound_method.loc15_41.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.373]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.83e]
// CHECK:STDOUT:   %bound_method.loc15_41.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.c84]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_41.2(%int_1.loc15) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_41.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc15 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_41.2: %i64 = converted %int_1.loc15, %.loc15_41.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_26: type = splice_block %long_long.ref [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc15_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc16: type = splice_block %i64.loc16 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc16: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc16: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.68c = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc18_30.1: %.edb = impl_witness_access constants.%As.impl_witness.bf0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.035]
// CHECK:STDOUT:   %bound_method.loc18_30.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.17b]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_30.1, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc18_30.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.6b2]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc18: init %i32 = call %bound_method.loc18_30.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc18 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_30.2: %i32 = converted %int_1.loc18, %.loc18_30.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_20: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc18_30.2: %.1bd = impl_witness_access constants.%ImplicitAs.impl_witness.9db, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_30.3: <bound method> = bound_method %.loc18_30.2, %impl.elem0.loc18_30.2 [concrete = constants.%i32.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc18_30.3(%.loc18_30.2) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_30.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_30.4: %Cpp.long = converted %.loc18_30.2, %.loc18_30.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc18_30.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc19: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc19: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc19_62.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.305 = addr_of %.loc19_62.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_long.ref.loc19, %addr.loc19)
// CHECK:STDOUT:   %.loc19_62.2: init %LongResult = in_place_init %PassLong__carbon_thunk.call.loc19, %.loc19_62.1
// CHECK:STDOUT:   %.loc19_27: type = splice_block %LongResult.ref.loc19 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc19: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_62.3: ref %LongResult = temporary %.loc19_62.1, %.loc19_62.2
// CHECK:STDOUT:   %.loc19_62.4: %LongResult = acquire_value %.loc19_62.3
// CHECK:STDOUT:   %cpp_long_result: %LongResult = value_binding cpp_long_result, %.loc19_62.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long.patt: %pattern_type.68c = value_binding_pattern cpp_compat_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc21: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc21: type = splice_block %Long32.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.c59 [concrete = imports.%CppCompat.c59]
// CHECK:STDOUT:     %Long32.ref: type = name_ref Long32, imports.%Core.Long32 [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long: %Cpp.long = value_binding cpp_compat_long, %cpp_long.ref.loc21
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_compat_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc22: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc22: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc22_74.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.305 = addr_of %.loc22_74.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_long.ref.loc22, %addr.loc22)
// CHECK:STDOUT:   %.loc22_74.2: init %LongResult = in_place_init %PassLong__carbon_thunk.call.loc22, %.loc22_74.1
// CHECK:STDOUT:   %.loc22_39: type = splice_block %LongResult.ref.loc22 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc22: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_74.3: ref %LongResult = temporary %.loc22_74.1, %.loc22_74.2
// CHECK:STDOUT:   %.loc22_74.4: %LongResult = acquire_value %.loc22_74.3
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongResult = value_binding cpp_compat_long_long_result, %.loc22_74.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.7ce = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc24: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc24_19: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc24: %.261 = impl_witness_access constants.%ImplicitAs.impl_witness.c39, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %cpp_long.ref.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc24(%cpp_long.ref.loc24)
// CHECK:STDOUT:   %.loc24_25.1: %i32 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc24_25.2: %i32 = converted %cpp_long.ref.loc24, %.loc24_25.1
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc24_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.454 = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc25_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc25: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc25_65.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc25: %ptr.3cb = addr_of %.loc25_65.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc25: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.2(%carbon_i32.ref, %addr.loc25)
// CHECK:STDOUT:   %.loc25_65.2: init %IntResult = in_place_init %PassLong__carbon_thunk.call.loc25, %.loc25_65.1
// CHECK:STDOUT:   %.loc25_29: type = splice_block %IntResult.ref [concrete = constants.%IntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc25_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntResult.ref: type = name_ref IntResult, imports.%IntResult.decl [concrete = constants.%IntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_65.3: ref %IntResult = temporary %.loc25_65.1, %.loc25_65.2
// CHECK:STDOUT:   %.loc25_65.4: %IntResult = acquire_value %.loc25_65.3
// CHECK:STDOUT:   %carbon_i32_result: %IntResult = value_binding carbon_i32_result, %.loc25_65.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc27: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %u64.loc27: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc27: %.5b8 = impl_witness_access constants.%As.impl_witness.ed2d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a09]
// CHECK:STDOUT:   %bound_method.loc27_59.1: <bound method> = bound_method %int_1.loc27, %impl.elem0.loc27 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.eb8]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Core.IntLiteral.as.As.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.009]
// CHECK:STDOUT:   %bound_method.loc27_59.2: <bound method> = bound_method %int_1.loc27, %specific_fn.loc27 [concrete = constants.%bound_method.bb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc27: init %u64 = call %bound_method.loc27_59.2(%int_1.loc27) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc27_59.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc27 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc27_59.2: %u64 = converted %int_1.loc27, %.loc27_59.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc27_35: type = splice_block %unsigned_long_long.ref [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref: type = name_ref unsigned_long_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc27_59.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc28: type = splice_block %u64.loc28 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc28: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc28: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc25: <bound method> = bound_method %.loc25_65.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.515
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc25: <bound method> = bound_method %.loc25_65.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25(%.loc25_65.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %.loc22_74.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.418
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %.loc22_74.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%.loc22_74.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc19: <bound method> = bound_method %.loc19_62.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.418
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %.loc19_62.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc19: init %empty_tuple.type = call %bound_method.loc19(%.loc19_62.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unsupported_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.465: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %To.586: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.56b: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.586) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.02d: %Core.IntLiteral.as.As.impl.Convert.type.56b = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ed2d: <witness> = impl_witness imports.%As.impl_witness_table.a7d, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.422: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a09: %Core.IntLiteral.as.As.impl.Convert.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.465 = facet_value Core.IntLiteral, (%As.impl_witness.ed2d) [concrete]
// CHECK:STDOUT:   %.5b8: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a09 [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a09, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.56b) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.02d)]
// CHECK:STDOUT:   %As.impl_witness_table.a7d = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: <error> = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_64.loc15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %u64.loc15: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0: %.5b8 = impl_witness_access constants.%As.impl_witness.ed2d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a09]
// CHECK:STDOUT:   %bound_method.loc15_49.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_49.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc15_49.2(%int_1) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc15_49.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc15_49.2: %u64 = converted %int_1, %.loc15_49.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %cpp_unsigned_long: <error> = value_binding cpp_unsigned_long, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: <error> = name_ref cpp_unsigned_long, %cpp_unsigned_long [concrete = <error>]
// CHECK:STDOUT:   %.loc16: type = splice_block %u64.loc16 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc16: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc16: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
