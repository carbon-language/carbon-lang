// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/layout.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/layout.carbon

// --- element_tail_padding.carbon
library "[[@TEST_NAME]]";

fn Check(p: i32*);

fn Test() {
  // We insert tail padding after each element (including the last) to keep the
  // array elements properly aligned and to ensure that a past-the-end address
  // points within the overall object.
  var v: array({.a: ()*, .b: i32}, 3);
  Check(&v[0].b);
  Check(&v[1].b);
  Check(&v[2].b);
}

// --- constant_with_padding.carbon
library "[[@TEST_NAME]]";

var c: array(Core.Int(257), 5) = (1, 2, 3, 4, 5);

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'element_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "element_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCheck.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !29
// CHECK:STDOUT:   %.loc10_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc10_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc10_13.array.index, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc10_14.b), !dbg !33
// CHECK:STDOUT:   %.loc11_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   %.loc11_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc11_13.array.index, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc11_14.b), !dbg !36
// CHECK:STDOUT:   %.loc12_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 2, !dbg !37
// CHECK:STDOUT:   %.loc12_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc12_13.array.index, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc12_14.b), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.95cbbc5c97be04bc(ptr %v.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.9e05708f03266b13:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.95cbbc5c97be04bc(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.9e05708f03266b13:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "element_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !18, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 9, column: 3, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 10, column: 10, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 3, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 11, column: 10, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 12, column: 10, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 3, scope: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 1, scope: !28)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 9, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bcbe81284b6bcb5:core.Destroy.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 9, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.9e05708f03266b13:core.Destroy.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 9, column: 3, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.95cbbc5c97be04bc", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 5, scope: !53)
// CHECK:STDOUT: !58 = !DILocation(line: 24, column: 3, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'constant_with_padding.carbon'
// CHECK:STDOUT: source_filename = "constant_with_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cc.Main = global [5 x <{ i257, [28 x i8] }>] zeroinitializer
// CHECK:STDOUT: @array.loc3_1 = internal constant [5 x <{ i257, [28 x i8] }>] [<{ i257, [28 x i8] }> <{ i257 1, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 2, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 3, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 4, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 5, [28 x i8] poison }>]
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 32 @_Cc.Main, ptr align 32 @array.loc3_1, i64 320, i1 false), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !60 = !DIFile(filename: "constant_with_padding.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !60, type: !25, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 3, column: 1, scope: !63)
// CHECK:STDOUT: !66 = !DILocation(line: 0, scope: !63)
// CHECK:STDOUT:
