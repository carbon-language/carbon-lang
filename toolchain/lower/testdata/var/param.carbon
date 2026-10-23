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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !24
// CHECK:STDOUT:   call void @_CMake.C.Main(ptr %c.var), !dbg !25
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !26
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5d59e052b8dad42e(ptr %c.var), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5d59e052b8dad42e(ptr %t) #0 !dbg !28 {
// CHECK:STDOUT:   call void @"_COp.C.Main:DestroyImpl.Core"(ptr %t), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "with_cleanup.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !14, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 12, column: 6, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 15, column: 5, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 3, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 14, column: 1, scope: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5d59e052b8dad42e", scope: null, file: !4, line: 32, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !34)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !16)
// CHECK:STDOUT: !31 = !DILocation(line: 33, column: 5, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 32, column: 1, scope: !28)
// CHECK:STDOUT: !34 = !{!29}
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !38
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc11_6.1, i64 0, i1 false), !dbg !38
// CHECK:STDOUT:   call void @_CF.Main(ptr %c.var), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5d59e052b8dad42e(ptr %c.var), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5d59e052b8dad42e(ptr %t) #0 !dbg !41 {
// CHECK:STDOUT:   call void @"_COp.C.Main:DestroyImpl.Core"(ptr %t), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !35 = !DIFile(filename: "struct_literal_cleanup.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !35, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 11, column: 6, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 3, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 13, column: 1, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5d59e052b8dad42e", scope: null, file: !4, line: 32, type: !18, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !47)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !16)
// CHECK:STDOUT: !44 = !DILocation(line: 33, column: 5, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 32, column: 1, scope: !41)
// CHECK:STDOUT: !47 = !{!42}
// CHECK:STDOUT:
