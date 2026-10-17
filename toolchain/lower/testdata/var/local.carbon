// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/local.carbon

// --- simple.carbon

library "[[@TEST_NAME]]";

fn Run() -> i32 {
  var x: i32 = 1;
  return x;
}

// --- eval_init_in_place.carbon

library "[[@TEST_NAME]]";

eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }

fn G() -> (i32, i32, i32) {
  var a: (i32, i32, i32) = F();
  return a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !26
// CHECK:STDOUT:   store i32 1, ptr %x.var, align 4, !dbg !26
// CHECK:STDOUT:   %.loc6 = load i32, ptr %x.var, align 4, !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %x.var), !dbg !26
// CHECK:STDOUT:   ret i32 %.loc6, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !32 {
// CHECK:STDOUT:   ret void, !dbg !37
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 5, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 10, scope: !23)
// CHECK:STDOUT: !29 = !{null, !20}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 6, column: 3, scope: !23)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !10, line: 30, type: !30, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !38)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !37 = !DILocation(line: 30, column: 1, scope: !32)
// CHECK:STDOUT: !38 = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'eval_init_in_place.carbon'
// CHECK:STDOUT: source_filename = "eval_init_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.80e.loc4_50 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !47
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.80e.loc4_50, i64 12, i1 false), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !52
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.80e.loc4_50, i64 12, i1 false), !dbg !52
// CHECK:STDOUT:   %tuple.elem0.loc8_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !54
// CHECK:STDOUT:   %.loc8_10.1 = load i32, ptr %tuple.elem0.loc8_10.1.tuple.elem, align 4, !dbg !54
// CHECK:STDOUT:   %tuple.elem0.loc8_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !54
// CHECK:STDOUT:   store i32 %.loc8_10.1, ptr %tuple.elem0.loc8_10.2.tuple.elem, align 4, !dbg !54
// CHECK:STDOUT:   %tuple.elem1.loc8_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !54
// CHECK:STDOUT:   %.loc8_10.3 = load i32, ptr %tuple.elem1.loc8_10.1.tuple.elem, align 4, !dbg !54
// CHECK:STDOUT:   %tuple.elem1.loc8_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !54
// CHECK:STDOUT:   store i32 %.loc8_10.3, ptr %tuple.elem1.loc8_10.2.tuple.elem, align 4, !dbg !54
// CHECK:STDOUT:   %tuple.elem2.loc8_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !54
// CHECK:STDOUT:   %.loc8_10.5 = load i32, ptr %tuple.elem2.loc8_10.1.tuple.elem, align 4, !dbg !54
// CHECK:STDOUT:   %tuple.elem2.loc8_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !54
// CHECK:STDOUT:   store i32 %.loc8_10.5, ptr %tuple.elem2.loc8_10.2.tuple.elem, align 4, !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d03c56e034d264ab(ptr %a.var), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d03c56e034d264ab(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!40}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !39 = !DIFile(filename: "eval_init_in_place.carbon", directory: "")
// CHECK:STDOUT: !40 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !39, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{!41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !39, line: 4, type: !43, spFlags: DISPFlagDefinition, unit: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 4, column: 41, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 4, column: 34, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !39, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !40)
// CHECK:STDOUT: !52 = !DILocation(line: 7, column: 3, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 8, column: 10, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 8, column: 3, scope: !49)
// CHECK:STDOUT: !56 = !{null, !41}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d03c56e034d264ab", scope: null, file: !10, line: 30, type: !57, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !41)
// CHECK:STDOUT: !63 = !DILocation(line: 30, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT:
