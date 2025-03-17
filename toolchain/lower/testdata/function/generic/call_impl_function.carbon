// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_impl_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_impl_function.carbon

import Core library "io";

interface SomeInterface(T:! type) {
  fn F(x: T) -> T;
}

class SomeParam {}

class ImplsSomeInterface {}
impl ImplsSomeInterface as SomeInterface(i32) {
  fn F(x: i32) -> i32 {
    return x + x;
  }
}

// TODO: This asserts with "Mangled name collision".
// impl ImplsSomeInterface as SomeInterface(SomeParam) {
  // fn F(x: SomeParam) -> SomeParam {
    // return {};
  // }
// }


fn CallGenericMethod(T:! type, U:! SomeInterface(T), x: T) -> T {
  return U.F(x);
}

fn G() {
  var out : i32 = CallGenericMethod(i32, ImplsSomeInterface, 10);
  Core.Print(out);

  // TODO: Fix above
  // CallGenericMethod(SomeParam, ImplsSomeInterface, {});
}


// CHECK:STDOUT: ; ModuleID = 'call_impl_function.carbon'
// CHECK:STDOUT: source_filename = "call_impl_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.Main"(i32 %x) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %x, %x, !dbg !7
// CHECK:STDOUT:   ret i32 %int.sadd, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %out.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %out.var), !dbg !10
// CHECK:STDOUT:   %CallGenericMethod.call = call i32 @_CCallGenericMethod.Main.2b28d1a7fa793cbc(i32 10), !dbg !11
// CHECK:STDOUT:   store i32 %CallGenericMethod.call, ptr %out.var, align 4, !dbg !10
// CHECK:STDOUT:   %.loc40 = load i32, ptr %out.var, align 4, !dbg !12
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %.loc40), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CCallGenericMethod.Main.2b28d1a7fa793cbc(i32 %x) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_15 = call i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.Main"(i32 %x), !dbg !16
// CHECK:STDOUT:   ret i32 %.loc35_15, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_impl_function.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.ImplsSomeInterface.Main:SomeInterface.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 22, column: 12, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 39, column: 3, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 39, column: 19, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 40, column: 14, scope: !9)
// CHECK:STDOUT: !13 = !DILocation(line: 40, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 38, column: 1, scope: !9)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "CallGenericMethod", linkageName: "_CCallGenericMethod.Main.2b28d1a7fa793cbc", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 35, column: 10, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 35, column: 3, scope: !15)
