// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c20bd9e093f68e6f(ptr %v.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0bcbe81284b6bcb5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc9_3.1.b), !dbg !43
// CHECK:STDOUT:   %.loc9_3.4.a = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %.loc9_3.4.a), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2bf9274c89d5ec9f:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %self, i32 0, i64 2, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1692a9e18a371c25(ptr %.loc9_3.1.array.index), !dbg !49
// CHECK:STDOUT:   %.loc9_3.4.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %self, i32 0, i64 1, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1692a9e18a371c25(ptr %.loc9_3.4.array.index), !dbg !49
// CHECK:STDOUT:   %.loc9_3.7.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %self, i32 0, i64 0, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1692a9e18a371c25(ptr %.loc9_3.7.array.index), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c20bd9e093f68e6f(ptr %t) #0 !dbg !51 {
// CHECK:STDOUT:   call void @"_COp.2bf9274c89d5ec9f:core.SubobjectDestroy.Core"(ptr %t), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %t) #0 !dbg !65 {
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1692a9e18a371c25(ptr %t) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_COp.0bcbe81284b6bcb5:core.SubobjectDestroy.Core"(ptr %t), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.1692a9e18a371c25, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIFile(filename: "element_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !20, line: 5, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 9, column: 3, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 10, column: 10, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 3, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 11, column: 10, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 12, column: 10, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 3, scope: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 1, scope: !28)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0bcbe81284b6bcb5:core.SubobjectDestroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !46)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !23)
// CHECK:STDOUT: !43 = !DILocation(line: 9, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{null, !22}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !{!42}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2bf9274c89d5ec9f:core.SubobjectDestroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 3, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c20bd9e093f68e6f", scope: null, file: !14, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !23)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 5, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 1, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !14, line: 26, type: !45, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !22)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.437d3b42b67f7a2e", scope: null, file: !14, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !71)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !23)
// CHECK:STDOUT: !70 = !DILocation(line: 26, column: 1, scope: !65)
// CHECK:STDOUT: !71 = !{!66}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1692a9e18a371c25", scope: null, file: !14, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !23)
// CHECK:STDOUT: !76 = !DILocation(line: 28, column: 5, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 1, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
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
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 32 @_Cc.Main, ptr align 32 @array.loc3_1, i64 320, i1 false), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !79 = !DIFile(filename: "constant_with_padding.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !79, type: !27, spFlags: DISPFlagDefinition, unit: !80)
// CHECK:STDOUT: !84 = !DILocation(line: 3, column: 1, scope: !82)
// CHECK:STDOUT: !85 = !DILocation(line: 0, scope: !82)
// CHECK:STDOUT:
