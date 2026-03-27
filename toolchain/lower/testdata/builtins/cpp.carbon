// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/cpp.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/cpp.carbon

// --- std_initializer_list_make.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T>
struct InitializerListPointerPointer {
  T* begin;
  T* end;
};

template<typename T>
struct InitializerListPointerSize {
  T* begin;
  decltype(sizeof(T)) size;
};
''';

fn MakePointerPointer(a: array(i32, 4)) -> Cpp.InitializerListPointerPointer(i32) = "cpp.std.initializer_list.make";
fn MakePointerSize(a: array(i32, 4)) -> Cpp.InitializerListPointerSize(i32) = "cpp.std.initializer_list.make";

fn TakePointerPointer(a: Cpp.InitializerListPointerPointer(i32));
fn TakePointerSize(a: Cpp.InitializerListPointerSize(i32));

fn Test() {
  TakePointerPointer(MakePointerPointer((1, 2, 3, 4)));
  TakePointerSize(MakePointerSize((1, 2, 3, 4)));
}

// CHECK:STDOUT: ; ModuleID = 'std_initializer_list_make.carbon'
// CHECK:STDOUT: source_filename = "std_initializer_list_make.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CTakePointerPointer.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CTakePointerSize.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_53.1.temp = alloca [16 x i8], align 1, !dbg !14
// CHECK:STDOUT:   %.loc25_52.3.temp = alloca [4 x i32], align 4, !dbg !15
// CHECK:STDOUT:   %.loc26_47.1.temp = alloca [16 x i8], align 1, !dbg !16
// CHECK:STDOUT:   %.loc26_46.3.temp = alloca [4 x i32], align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_53.1.temp), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_52.3.temp), !dbg !15
// CHECK:STDOUT:   %.loc25_52.4.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 0, !dbg !15
// CHECK:STDOUT:   %.loc25_52.7.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 1, !dbg !15
// CHECK:STDOUT:   %.loc25_52.10.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 2, !dbg !15
// CHECK:STDOUT:   %.loc25_52.13.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 3, !dbg !15
// CHECK:STDOUT:   %MakePointerPointer.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %.loc25_53.1.temp, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   store ptr @array, ptr %MakePointerPointer.call.init_list.begin, align 8, !dbg !14
// CHECK:STDOUT:   %MakePointerPointer.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %.loc25_53.1.temp, i32 0, i32 8, !dbg !14
// CHECK:STDOUT:   store ptr getelementptr inbounds ([4 x i32], ptr @array, i32 1), ptr %MakePointerPointer.call.init_list.end, align 8, !dbg !14
// CHECK:STDOUT:   call void @_CTakePointerPointer.Main(ptr %.loc25_53.1.temp), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_47.1.temp), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_46.3.temp), !dbg !17
// CHECK:STDOUT:   %.loc26_46.4.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 0, !dbg !17
// CHECK:STDOUT:   %.loc26_46.7.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 1, !dbg !17
// CHECK:STDOUT:   %.loc26_46.10.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 2, !dbg !17
// CHECK:STDOUT:   %.loc26_46.13.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 3, !dbg !17
// CHECK:STDOUT:   %MakePointerSize.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %.loc26_47.1.temp, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store ptr @array, ptr %MakePointerSize.call.init_list.begin, align 8, !dbg !16
// CHECK:STDOUT:   %MakePointerSize.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %.loc26_47.1.temp, i32 0, i32 8, !dbg !16
// CHECK:STDOUT:   store i64 4, ptr %MakePointerSize.call.init_list.size, align 8, !dbg !16
// CHECK:STDOUT:   call void @_CTakePointerSize.Main(ptr %.loc26_47.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "std_initializer_list_make.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !6, line: 24, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 25, column: 22, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 25, column: 41, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 26, column: 19, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 26, column: 35, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 25, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 26, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 24, column: 1, scope: !11)
