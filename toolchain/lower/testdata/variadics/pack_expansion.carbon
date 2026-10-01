// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/variadics/pack_expansion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/variadics/pack_expansion.carbon

// --- homogeneous.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

fn F(x: (i32, i32, i32)) {
  ... G(expand x);
}

// --- heterogeneous.carbon
library "[[@TEST_NAME]]";

fn G[T: type](unused t: T) {}

fn F(x: (i32, bool), y: (bool, i32)) {
  ... if (true) {
    G(expand x);
    G(expand y);
  }
}

// --- generic.carbon
library "[[@TEST_NAME]]";

fn G[T: type](unused t: T) {}

fn F[T: type](x: (T, T)) {
  ... G(expand x);
}

fn Call(a: (i32, i32)) {
  F(a);
}

// --- multiple_expands.carbon
library "[[@TEST_NAME]]";

fn G(a: i32, b: bool);

fn F(x: (i32, i32), y: (bool, bool)) {
  ... G(expand x, expand y);
}

// --- empty.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

fn F(x: ()) {
  ... G(expand x);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'homogeneous.carbon'
// CHECK:STDOUT: source_filename = "homogeneous.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %x) #0 !dbg !40 {
// CHECK:STDOUT: pack.body:
// CHECK:STDOUT:   %.loc6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %x, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   %.loc6.tuple.elem.load = load i32, ptr %.loc6.tuple.elem, align 4, !dbg !45
// CHECK:STDOUT:   call void @_CG.Main(i32 %.loc6.tuple.elem.load), !dbg !46
// CHECK:STDOUT:   br label %pack.body2, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: pack.body2:                                       ; preds = %pack.body
// CHECK:STDOUT:   %.loc6.tuple.elem3 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %x, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   %.loc6.tuple.elem.load4 = load i32, ptr %.loc6.tuple.elem3, align 4, !dbg !45
// CHECK:STDOUT:   call void @_CG.Main(i32 %.loc6.tuple.elem.load4), !dbg !46
// CHECK:STDOUT:   br label %pack.body5, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: pack.body5:                                       ; preds = %pack.body2
// CHECK:STDOUT:   %.loc6.tuple.elem6 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %x, i32 0, i32 2, !dbg !45
// CHECK:STDOUT:   %.loc6.tuple.elem.load7 = load i32, ptr %.loc6.tuple.elem6, align 4, !dbg !45
// CHECK:STDOUT:   call void @_CG.Main(i32 %.loc6.tuple.elem.load7), !dbg !46
// CHECK:STDOUT:   br label %pack.done, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: pack.done:                                        ; preds = %pack.body5
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "homogeneous.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !32, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !48)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 6, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 6, column: 9, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 6, column: 7, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 5, column: 1, scope: !40)
// CHECK:STDOUT: !48 = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'heterogeneous.carbon'
// CHECK:STDOUT: source_filename = "heterogeneous.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %x, ptr %y) #0 !dbg !53 {
// CHECK:STDOUT: pack.body:
// CHECK:STDOUT:   br i1 true, label %if.then, label %if.else, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %pack.body
// CHECK:STDOUT:   %.loc7_7.3.tuple.elem = getelementptr inbounds nuw <{ i32, i1 }>, ptr %x, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   %.loc7_7.3.tuple.elem.load = load i32, ptr %.loc7_7.3.tuple.elem, align 4, !dbg !61
// CHECK:STDOUT:   call void @_CG.Main.84588f41d61dafba(i32 %.loc7_7.3.tuple.elem.load), !dbg !62
// CHECK:STDOUT:   %.loc8_7.3.tuple.elem = getelementptr inbounds nuw { i1, i32 }, ptr %y, i32 0, i32 0, !dbg !64
// CHECK:STDOUT:   %.loc8_7.3.tuple.elem.load = load i8, ptr %.loc8_7.3.tuple.elem, align 1, !dbg !64
// CHECK:STDOUT:   %.loc8_7.3 = trunc i8 %.loc8_7.3.tuple.elem.load to i1, !dbg !64
// CHECK:STDOUT:   call void @_CG.Main.a4c7fafe8a72cac4(i1 %.loc8_7.3), !dbg !65
// CHECK:STDOUT:   br label %if.else, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %pack.body
// CHECK:STDOUT:   br label %pack.body1, !dbg !58
// CHECK:STDOUT:
// CHECK:STDOUT: pack.body1:                                       ; preds = %if.else
// CHECK:STDOUT:   br i1 true, label %if.then2, label %if.else3, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.then2:                                         ; preds = %pack.body1
// CHECK:STDOUT:   %.loc7_7.3.tuple.elem4 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %x, i32 0, i32 1, !dbg !61
// CHECK:STDOUT:   %.loc7_7.3.tuple.elem.load5 = load i8, ptr %.loc7_7.3.tuple.elem4, align 1, !dbg !61
// CHECK:STDOUT:   %.loc7_7.3 = trunc i8 %.loc7_7.3.tuple.elem.load5 to i1, !dbg !61
// CHECK:STDOUT:   call void @_CG.Main.a4c7fafe8a72cac4(i1 %.loc7_7.3), !dbg !62
// CHECK:STDOUT:   %.loc8_7.3.tuple.elem6 = getelementptr inbounds nuw { i1, i32 }, ptr %y, i32 0, i32 1, !dbg !64
// CHECK:STDOUT:   %.loc8_7.3.tuple.elem.load7 = load i32, ptr %.loc8_7.3.tuple.elem6, align 4, !dbg !64
// CHECK:STDOUT:   call void @_CG.Main.84588f41d61dafba(i32 %.loc8_7.3.tuple.elem.load7), !dbg !65
// CHECK:STDOUT:   br label %if.else3, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.else3:                                         ; preds = %if.then2, %pack.body1
// CHECK:STDOUT:   br label %pack.done, !dbg !58
// CHECK:STDOUT:
// CHECK:STDOUT: pack.done:                                        ; preds = %if.else3
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.84588f41d61dafba(i32 %t) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.a4c7fafe8a72cac4(i1 %t) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CG.Main.84588f41d61dafba, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.a4c7fafe8a72cac4, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!50}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !49 = !DIFile(filename: "heterogeneous.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !49, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !51 = !{null, !37, !37}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !49, line: 5, type: !52, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !68)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !37)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !53, type: !37)
// CHECK:STDOUT: !58 = !DILocation(line: 6, column: 3, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 6, column: 10, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 7, column: 7, scope: !53)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 5, scope: !53)
// CHECK:STDOUT: !64 = !DILocation(line: 8, column: 7, scope: !53)
// CHECK:STDOUT: !65 = !DILocation(line: 8, column: 5, scope: !53)
// CHECK:STDOUT: !66 = !DILocation(line: 6, column: 7, scope: !53)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 1, scope: !53)
// CHECK:STDOUT: !68 = !{!54, !55}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.84588f41d61dafba", scope: null, file: !49, line: 3, type: !36, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !73)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !34)
// CHECK:STDOUT: !72 = !DILocation(line: 3, column: 1, scope: !69)
// CHECK:STDOUT: !73 = !{!70}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a4c7fafe8a72cac4", scope: null, file: !49, line: 3, type: !39, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !78)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !37)
// CHECK:STDOUT: !77 = !DILocation(line: 3, column: 1, scope: !74)
// CHECK:STDOUT: !78 = !{!75}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic.carbon'
// CHECK:STDOUT: source_filename = "generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %a) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(ptr %a), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(ptr %x) #0 !dbg !88 {
// CHECK:STDOUT: pack.body:
// CHECK:STDOUT:   %.loc6.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x, i32 0, i32 0, !dbg !93
// CHECK:STDOUT:   %.loc6.tuple.elem.load = load i32, ptr %.loc6.tuple.elem, align 4, !dbg !93
// CHECK:STDOUT:   call void @_CG.Main.84588f41d61dafba(i32 %.loc6.tuple.elem.load), !dbg !94
// CHECK:STDOUT:   br label %pack.body1, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: pack.body1:                                       ; preds = %pack.body
// CHECK:STDOUT:   %.loc6.tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %x, i32 0, i32 1, !dbg !93
// CHECK:STDOUT:   %.loc6.tuple.elem.load3 = load i32, ptr %.loc6.tuple.elem2, align 4, !dbg !93
// CHECK:STDOUT:   call void @_CG.Main.84588f41d61dafba(i32 %.loc6.tuple.elem.load3), !dbg !94
// CHECK:STDOUT:   br label %pack.done, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: pack.done:                                        ; preds = %pack.body1
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.84588f41d61dafba(i32 %t) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CG.Main.84588f41d61dafba, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !79 = !DIFile(filename: "generic.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !79, line: 9, type: !39, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !37)
// CHECK:STDOUT: !85 = !DILocation(line: 10, column: 3, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 9, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !79, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !96)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !37)
// CHECK:STDOUT: !91 = !DILocation(line: 6, column: 3, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 6, column: 9, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 6, column: 7, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 5, column: 1, scope: !88)
// CHECK:STDOUT: !96 = !{!89}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.84588f41d61dafba", scope: null, file: !79, line: 3, type: !36, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !101)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !34)
// CHECK:STDOUT: !100 = !DILocation(line: 3, column: 1, scope: !97)
// CHECK:STDOUT: !101 = !{!98}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'multiple_expands.carbon'
// CHECK:STDOUT: source_filename = "multiple_expands.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32, i1)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %x, ptr %y) #0 !dbg !106 {
// CHECK:STDOUT: pack.body:
// CHECK:STDOUT:   %.loc6_9.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x, i32 0, i32 0, !dbg !113
// CHECK:STDOUT:   %.loc6_9.tuple.elem.load = load i32, ptr %.loc6_9.tuple.elem, align 4, !dbg !113
// CHECK:STDOUT:   %.loc6_19.tuple.elem = getelementptr inbounds nuw { i1, i1 }, ptr %y, i32 0, i32 0, !dbg !115
// CHECK:STDOUT:   %.loc6_19.tuple.elem.load = load i8, ptr %.loc6_19.tuple.elem, align 1, !dbg !115
// CHECK:STDOUT:   %.loc6_19 = trunc i8 %.loc6_19.tuple.elem.load to i1, !dbg !115
// CHECK:STDOUT:   call void @_CG.Main(i32 %.loc6_9.tuple.elem.load, i1 %.loc6_19), !dbg !116
// CHECK:STDOUT:   br label %pack.body1, !dbg !111
// CHECK:STDOUT:
// CHECK:STDOUT: pack.body1:                                       ; preds = %pack.body
// CHECK:STDOUT:   %.loc6_9.tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %x, i32 0, i32 1, !dbg !113
// CHECK:STDOUT:   %.loc6_9.tuple.elem.load3 = load i32, ptr %.loc6_9.tuple.elem2, align 4, !dbg !113
// CHECK:STDOUT:   %.loc6_19.tuple.elem4 = getelementptr inbounds nuw { i1, i1 }, ptr %y, i32 0, i32 1, !dbg !115
// CHECK:STDOUT:   %.loc6_19.tuple.elem.load5 = load i8, ptr %.loc6_19.tuple.elem4, align 1, !dbg !115
// CHECK:STDOUT:   %.loc6_196 = trunc i8 %.loc6_19.tuple.elem.load5 to i1, !dbg !115
// CHECK:STDOUT:   call void @_CG.Main(i32 %.loc6_9.tuple.elem.load3, i1 %.loc6_196), !dbg !116
// CHECK:STDOUT:   br label %pack.done, !dbg !111
// CHECK:STDOUT:
// CHECK:STDOUT: pack.done:                                        ; preds = %pack.body1
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!103}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !51 = !{null, !37, !37}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !102 = !DIFile(filename: "multiple_expands.carbon", directory: "")
// CHECK:STDOUT: !103 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !102, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !102, line: 5, type: !52, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !118)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !37)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 2, scope: !106, type: !37)
// CHECK:STDOUT: !111 = !DILocation(line: 6, column: 3, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 6, column: 9, scope: !106)
// CHECK:STDOUT: !115 = !DILocation(line: 6, column: 19, scope: !106)
// CHECK:STDOUT: !116 = !DILocation(line: 6, column: 7, scope: !106)
// CHECK:STDOUT: !117 = !DILocation(line: 5, column: 1, scope: !106)
// CHECK:STDOUT: !118 = !{!107, !108}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'empty.carbon'
// CHECK:STDOUT: source_filename = "empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %pack.done, !dbg !125
// CHECK:STDOUT:
// CHECK:STDOUT: pack.done:                                        ; preds = %entry
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!120}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !119 = !DIFile(filename: "empty.carbon", directory: "")
// CHECK:STDOUT: !120 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !119, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !121 = !{null}
// CHECK:STDOUT: !122 = !DISubroutineType(types: !121)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !119, line: 5, type: !122, spFlags: DISPFlagDefinition, unit: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 6, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !DILocation(line: 5, column: 1, scope: !123)
// CHECK:STDOUT:
