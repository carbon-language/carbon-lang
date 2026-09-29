// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/param.carbon

// --- with_cleanup.carbon

library "[[@TEST_NAME]]";

class C {
  // TODO: Change to Core.Destructor when possible.
  impl as Core.Destroy {
    fn Op(ref self);
  }

  fn Make() -> Self;
}

fn F(var c: C);

fn G() {
  F(C.Make());
}

// --- struct_literal_cleanup.carbon

library "[[@TEST_NAME]]";

class C {
  // TODO: Change to Core.Destructor when possible.
  impl as Core.Destroy {
    fn Op(ref self);
  }
}

fn F(var c: C);

fn G() {
  F({});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'with_cleanup.carbon'
// CHECK:STDOUT: source_filename = "with_cleanup.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.C.Main:Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.C.Main(ptr sret({}))
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !16
// CHECK:STDOUT:   call void @_CMake.C.Main(ptr %c.var), !dbg !17
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !18
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "with_cleanup.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !6, line: 15, type: !14, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 13, column: 6, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 16, column: 5, scope: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 3, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 15, column: 1, scope: !15)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !6, line: 13, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !23)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !20, type: !8)
// CHECK:STDOUT: !22 = !DILocation(line: 13, column: 6, scope: !20)
// CHECK:STDOUT: !23 = !{!21}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !6, line: 13, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !27)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !8)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 6, scope: !24)
// CHECK:STDOUT: !27 = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'struct_literal_cleanup.carbon'
// CHECK:STDOUT: source_filename = "struct_literal_cleanup.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc11 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.C.Main:Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !31
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc11, i64 0, i1 false), !dbg !31
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !32
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!29}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !28 = !DIFile(filename: "struct_literal_cleanup.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !28, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !28, line: 13, type: !14, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 11, column: 6, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 3, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 13, column: 1, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !28, line: 11, type: !10, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !8)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 6, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !28, line: 11, type: !10, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !8)
// CHECK:STDOUT: !40 = !DILocation(line: 11, column: 6, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT:
