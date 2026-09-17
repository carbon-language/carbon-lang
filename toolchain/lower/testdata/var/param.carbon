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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %c.var), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !6, line: 13, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !23)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !20, type: !8)
// CHECK:STDOUT: !22 = !DILocation(line: 13, column: 6, scope: !20)
// CHECK:STDOUT: !23 = !{!21}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !4, line: 24, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !30)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !8)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 5, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 24, column: 3, scope: !24)
// CHECK:STDOUT: !30 = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'struct_literal_cleanup.carbon'
// CHECK:STDOUT: source_filename = "struct_literal_cleanup.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc11_6.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.C.Main:Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !34
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc11_6.1, i64 0, i1 false), !dbg !34
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %c.var), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!32}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !31 = !DIFile(filename: "struct_literal_cleanup.carbon", directory: "")
// CHECK:STDOUT: !32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !31, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !31, line: 13, type: !14, spFlags: DISPFlagDefinition, unit: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 11, column: 6, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 14, column: 3, scope: !33)
// CHECK:STDOUT: !36 = !DILocation(line: 13, column: 1, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !31, line: 11, type: !10, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !8)
// CHECK:STDOUT: !39 = !DILocation(line: 11, column: 6, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !4, line: 24, type: !10, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !47)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !8)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 5, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 3, scope: !41)
// CHECK:STDOUT: !47 = !{!42}
// CHECK:STDOUT:
