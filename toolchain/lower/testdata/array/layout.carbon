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

// CHECK:STDOUT: ; ModuleID = 'element_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "element_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCheck.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc10_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc10_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc10_13.array.index, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc10_14.b), !dbg !9
// CHECK:STDOUT:   %.loc11_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   %.loc11_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc11_13.array.index, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc11_14.b), !dbg !11
// CHECK:STDOUT:   %.loc12_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 2, !dbg !12
// CHECK:STDOUT:   %.loc12_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc12_13.array.index, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc12_14.b), !dbg !13
// CHECK:STDOUT:   call void @"_COp.0154c643da095258:core.Destroy.Core"(ptr %v.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %self) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b5e6b9ba1f3e5e4d:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0154c643da095258:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "element_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 10, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1fa36cd3e044ac37:core.Destroy.Core", scope: null, file: !3, line: 9, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !15, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 3, scope: !15)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b5e6b9ba1f3e5e4d:core.Destroy.Core", scope: null, file: !3, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 3, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0154c643da095258:core.Destroy.Core", scope: null, file: !3, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 3, scope: !29)
// CHECK:STDOUT: ; ModuleID = 'constant_with_padding.carbon'
// CHECK:STDOUT: source_filename = "constant_with_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cc.Main = global [5 x <{ i257, [28 x i8] }>] zeroinitializer
// CHECK:STDOUT: @array.loc3_1 = internal constant [5 x <{ i257, [28 x i8] }>] [<{ i257, [28 x i8] }> <{ i257 1, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 2, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 3, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 4, [28 x i8] poison }>, <{ i257, [28 x i8] }> <{ i257 5, [28 x i8] poison }>]
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 32 @_Cc.Main, ptr align 32 @array.loc3_1, i64 320, i1 false), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "constant_with_padding.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 3, column: 1, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 0, scope: !4)
