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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !7
// CHECK:STDOUT:   call void @_CMake.C.Main(ptr %c.var), !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %c.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "with_cleanup.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 13, column: 6, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 5, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !1, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 6, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !19, line: 24, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !19 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !18, type: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 26, column: 5, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 24, column: 3, scope: !18)
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc11_6.1, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %c.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "struct_literal_cleanup.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 6, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !1, line: 11, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{null, !13}
// CHECK:STDOUT: !13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !10, type: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 11, column: 6, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !18, line: 24, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !18 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !17, type: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 26, column: 5, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 3, scope: !17)
// CHECK:STDOUT:
