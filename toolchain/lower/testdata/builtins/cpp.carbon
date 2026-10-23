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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'std_initializer_list_make.carbon'
// CHECK:STDOUT: source_filename = "std_initializer_list_make.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
// CHECK:STDOUT: @array.loc25_52.18 = internal constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CTakePointerPointer.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CTakePointerSize.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_53.1.temp = alloca [16 x i8], align 8, !dbg !14
// CHECK:STDOUT:   %.loc25_52.3.temp = alloca [4 x i32], align 4, !dbg !15
// CHECK:STDOUT:   %.loc26_47.1.temp = alloca [16 x i8], align 8, !dbg !17
// CHECK:STDOUT:   %.loc26_46.3.temp = alloca [4 x i32], align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_53.1.temp), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_52.3.temp), !dbg !15
// CHECK:STDOUT:   %.loc25_52.4.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 0, !dbg !15
// CHECK:STDOUT:   %.loc25_52.7.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 1, !dbg !15
// CHECK:STDOUT:   %.loc25_52.10.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 2, !dbg !15
// CHECK:STDOUT:   %.loc25_52.13.array.index = getelementptr inbounds [4 x i32], ptr %.loc25_52.3.temp, i32 0, i64 3, !dbg !15
// CHECK:STDOUT:   %MakePointerPointer.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %.loc25_53.1.temp, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   store ptr @array.loc25_52.18, ptr %MakePointerPointer.call.init_list.begin, align 8, !dbg !14
// CHECK:STDOUT:   %MakePointerPointer.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %.loc25_53.1.temp, i32 0, i32 8, !dbg !14
// CHECK:STDOUT:   store ptr getelementptr inbounds (i8, ptr @array.loc25_52.18, i64 16), ptr %MakePointerPointer.call.init_list.end, align 8, !dbg !14
// CHECK:STDOUT:   call void @_CTakePointerPointer.Main(ptr %.loc25_53.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.eddcd8182dbffd26(ptr %.loc25_53.1.temp), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.74a1f892059422e4(ptr @array), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_47.1.temp), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_46.3.temp), !dbg !18
// CHECK:STDOUT:   %.loc26_46.4.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 0, !dbg !18
// CHECK:STDOUT:   %.loc26_46.7.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 1, !dbg !18
// CHECK:STDOUT:   %.loc26_46.10.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 2, !dbg !18
// CHECK:STDOUT:   %.loc26_46.13.array.index = getelementptr inbounds [4 x i32], ptr %.loc26_46.3.temp, i32 0, i64 3, !dbg !18
// CHECK:STDOUT:   %MakePointerSize.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %.loc26_47.1.temp, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   store ptr @array.loc25_52.18, ptr %MakePointerSize.call.init_list.begin, align 8, !dbg !17
// CHECK:STDOUT:   %MakePointerSize.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %.loc26_47.1.temp, i32 0, i32 8, !dbg !17
// CHECK:STDOUT:   store i64 4, ptr %MakePointerSize.call.init_list.size, align 8, !dbg !17
// CHECK:STDOUT:   call void @_CTakePointerSize.Main(ptr %.loc26_47.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbfb6e6199068909(ptr %.loc26_47.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.74a1f892059422e4(ptr @array), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.eddcd8182dbffd26(ptr %t) #0 !dbg !22 {
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.74a1f892059422e4(ptr %t) #0 !dbg !29 {
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bbfb6e6199068909(ptr %t) #0 !dbg !36 {
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.74a1f892059422e4, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "std_initializer_list_make.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !{null}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !4, line: 24, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !14 = !DILocation(line: 25, column: 22, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 25, column: 41, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 3, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 26, column: 19, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 26, column: 35, scope: !13)
// CHECK:STDOUT: !19 = !DILocation(line: 26, column: 3, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 24, column: 1, scope: !13)
// CHECK:STDOUT: !21 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.eddcd8182dbffd26", scope: null, file: !21, line: 35, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !22, type: !8)
// CHECK:STDOUT: !27 = !DILocation(line: 35, column: 1, scope: !22)
// CHECK:STDOUT: !28 = !{!23}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.74a1f892059422e4", scope: null, file: !21, line: 35, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !8)
// CHECK:STDOUT: !34 = !DILocation(line: 35, column: 1, scope: !29)
// CHECK:STDOUT: !35 = !{!30}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bbfb6e6199068909", scope: null, file: !21, line: 35, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !8)
// CHECK:STDOUT: !41 = !DILocation(line: 35, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT:
