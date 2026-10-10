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
  impl as Core.DestroyImpl {
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
  impl as Core.DestroyImpl {
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
// CHECK:STDOUT: declare void @"_COp.C.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.C.Main(ptr sret({}))
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !22
// CHECK:STDOUT:   call void @_CMake.C.Main(ptr %c.var), !dbg !23
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %c.var), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !26 {
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!13}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !12 = !DIFile(filename: "with_cleanup.carbon", directory: "")
// CHECK:STDOUT: !13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !12, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !14}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !12, line: 14, type: !20, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !22 = !DILocation(line: 12, column: 6, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 15, column: 5, scope: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 15, column: 3, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 14, column: 1, scope: !21)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !4, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !32)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 26, column: 1, scope: !26)
// CHECK:STDOUT: !32 = !{!27}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'struct_literal_cleanup.carbon'
// CHECK:STDOUT: source_filename = "struct_literal_cleanup.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc11_6.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.C.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !36
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc11_6.1, i64 0, i1 false), !dbg !36
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %c.var), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !39 {
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!34}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !14}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !33 = !DIFile(filename: "struct_literal_cleanup.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !33, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 6, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 14, column: 3, scope: !35)
// CHECK:STDOUT: !38 = !DILocation(line: 13, column: 1, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !4, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !45)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 1, scope: !39)
// CHECK:STDOUT: !45 = !{!40}
// CHECK:STDOUT:
