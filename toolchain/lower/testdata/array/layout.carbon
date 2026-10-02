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
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], align 8, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !31
// CHECK:STDOUT:   %.loc10_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc10_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc10_13.array.index, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc10_14.b), !dbg !35
// CHECK:STDOUT:   %.loc11_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 1, !dbg !36
// CHECK:STDOUT:   %.loc11_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc11_13.array.index, i32 0, i32 1, !dbg !36
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc11_14.b), !dbg !38
// CHECK:STDOUT:   %.loc12_13.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %v.var, i32 0, i32 2, !dbg !39
// CHECK:STDOUT:   %.loc12_14.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc12_13.array.index, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   call void @_CCheck.Main(ptr %.loc12_14.b), !dbg !41
// CHECK:STDOUT:   call void @"_CSelfDestruct.2bf9274c89d5ec9f:core.Destroy.Core"(ptr %v.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc9_3.1.b), !dbg !53
// CHECK:STDOUT:   %.loc9_3.2.a = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %.loc9_3.2.a), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2bf9274c89d5ec9f:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %self, i32 0, i64 2, !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %.loc9_3.1.array.index), !dbg !61
// CHECK:STDOUT:   %.loc9_3.2.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %self, i32 0, i64 1, !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %.loc9_3.2.array.index), !dbg !61
// CHECK:STDOUT:   %.loc9_3.3.array.index = getelementptr inbounds [3 x <{ <{ ptr, i32 }>, [4 x i8] }>], ptr %self, i32 0, i64 0, !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.0bcbe81284b6bcb5:core.Destroy.Core"(ptr %.loc9_3.3.array.index), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2bf9274c89d5ec9f:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2bf9274c89d5ec9f:core.Destroy.Core"(ptr %self), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "element_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !22}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !20, line: 5, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 3, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 10, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 3, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 10, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 11, column: 3, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 10, scope: !30)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 3, scope: !30)
// CHECK:STDOUT: !42 = !DILocation(line: 5, column: 1, scope: !30)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8436f795915f0543:core.Destroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !45 = !DILocation(line: 9, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !20, line: 9, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !22)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 3, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bcbe81284b6bcb5:core.Destroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !23)
// CHECK:STDOUT: !53 = !DILocation(line: 9, column: 3, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0bcbe81284b6bcb5:core.Destroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !23)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2bf9274c89d5ec9f:core.Destroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 9, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2bf9274c89d5ec9f:core.Destroy.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !23)
// CHECK:STDOUT: !65 = !DILocation(line: 9, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
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
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 32 @_Cc.Main, ptr align 32 @array.loc3_1, i64 320, i1 false), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!68}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !67 = !DIFile(filename: "constant_with_padding.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !67, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !67, type: !27, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 3, column: 1, scope: !70)
// CHECK:STDOUT: !73 = !DILocation(line: 0, scope: !70)
// CHECK:STDOUT:
