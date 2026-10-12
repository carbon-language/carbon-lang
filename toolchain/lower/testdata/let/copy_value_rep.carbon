// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/let/copy_value_rep.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/let/copy_value_rep.carbon

class X {
  var a: i32;
}

fn Run() -> i32 {
  var x: X = {.a = 1};
  let n: i32 = x.a;
  x.a = 2;
  // Should return 1, not 2.
  return n;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_value_rep.carbon'
// CHECK:STDOUT: source_filename = "copy_value_rep.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @X.val.loc18_3.1 = internal constant { i32 } { i32 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32 }, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !29
// CHECK:STDOUT:   %.loc18_21.3.a = getelementptr inbounds nuw { i32 }, ptr %x.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x.var, ptr align 4 @X.val.loc18_3.1, i64 4, i1 false), !dbg !29
// CHECK:STDOUT:   %.loc19_17.1.a = getelementptr inbounds nuw { i32 }, ptr %x.var, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %.loc19_17.2 = load i32, ptr %.loc19_17.1.a, align 4, !dbg !32
// CHECK:STDOUT:   %.loc20_4.a = getelementptr inbounds nuw { i32 }, ptr %x.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store i32 2, ptr %.loc20_4.a, align 4, !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9060399a99d8613(ptr %x.var), !dbg !29
// CHECK:STDOUT:   ret i32 %.loc19_17.2, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5c911ef3833157e3:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc18_3.1.a), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8b0bf151036f045a:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc18_3.1.a), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e9060399a99d8613(ptr %t) #0 !dbg !48 {
// CHECK:STDOUT:   call void @"_COp.8b0bf151036f045a:core.SubobjectDestroy.Core"(ptr %t), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !55 {
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "copy_value_rep.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 18, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 18, column: 14, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 16, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 20, column: 3, scope: !26)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 3, scope: !26)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5c911ef3833157e3:core.SubobjectDestroy.Core", scope: null, file: !18, line: 18, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !43)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !23)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !{null, !20}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = !{!39}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8b0bf151036f045a:core.SubobjectDestroy.Core", scope: null, file: !18, line: 18, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !23)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e9060399a99d8613", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !23)
// CHECK:STDOUT: !52 = !DILocation(line: 28, column: 5, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 1, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !42, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT:
