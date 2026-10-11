// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/field.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> i32 {
  return (*c.b).a;
}

fn Run() -> i32 {
  var c: C;
  c.a = 1;
  c.b = &c;
  return F(c);
}

// --- inplace_init_with_nonconst.carbon

library "[[@TEST_NAME]]";

class Other { }

class Use {
  var a: Other*;
  var b: Other;
}

fn Run() {
  var o: Other;
  // .a is initialized with a non-constant expression
  // .b is initialized with in-place class_init
  var _: Use = {.a = &o, .b = {}};
}

// --- implicit_init_with_nonempty_nonconst.carbon

library "[[@TEST_NAME]]";

class Other {
  var v: i32;
}

class Use {
  var a: Other*;
  var b: Other;
}

fn Run() {
  var o: Other;
  // .a is initialized with a non-constant expression
  // .b is initialized with in-place class_init
  var _: Use = {.a = &o, .b = {.v = 42}};
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %c) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_13.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %c, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   %.loc10_13.2 = load ptr, ptr %.loc10_13.1.b, align 8, !dbg !33
// CHECK:STDOUT:   %.loc10_16.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %.loc10_13.2, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc10_16.2 = load i32, ptr %.loc10_16.1.a, align 4, !dbg !35
// CHECK:STDOUT:   ret i32 %.loc10_16.2, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !41
// CHECK:STDOUT:   %.loc15_4.a = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   store i32 1, ptr %.loc15_4.a, align 4, !dbg !43
// CHECK:STDOUT:   %.loc16.b = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 1, !dbg !44
// CHECK:STDOUT:   store ptr %c.var, ptr %.loc16.b, align 8, !dbg !44
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr %c.var), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %c.var), !dbg !41
// CHECK:STDOUT:   ret i32 %F.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.09861f44a9b84abf:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.766b861e1946716c(ptr %.loc14_3.1.b), !dbg !52
// CHECK:STDOUT:   %.loc14_3.4.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc14_3.4.a), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.766b861e1946716c(ptr %.loc14_3.1.b), !dbg !58
// CHECK:STDOUT:   %.loc14_3.4.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc14_3.4.a), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !60 {
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.SubobjectDestroy.Core"(ptr %t), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.766b861e1946716c(ptr %t) #0 !dbg !67 {
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.766b861e1946716c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !37)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 12, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 10, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 10, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !{!29}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !41 = !DILocation(line: 14, column: 3, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 15, column: 3, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 16, column: 3, scope: !38)
// CHECK:STDOUT: !48 = !DILocation(line: 17, column: 10, scope: !38)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 3, scope: !38)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.09861f44a9b84abf:core.SubobjectDestroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !21)
// CHECK:STDOUT: !52 = !DILocation(line: 14, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{null, !20}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = !{!51}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.SubobjectDestroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 14, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !21)
// CHECK:STDOUT: !64 = !DILocation(line: 28, column: 5, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 1, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.766b861e1946716c", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 26, column: 1, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !54, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !20)
// CHECK:STDOUT: !79 = !DILocation(line: 26, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'inplace_init_with_nonconst.carbon'
// CHECK:STDOUT: source_filename = "inplace_init_with_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc15_33.5 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca {}, align 1, !dbg !84
// CHECK:STDOUT:   %_.var = alloca { ptr, {} }, align 8, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !86
// CHECK:STDOUT:   %.loc15_33.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 0, !dbg !89
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc15_33.2.a, align 8, !dbg !89
// CHECK:STDOUT:   %.loc15_33.4.b = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 1, !dbg !89
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc15_33.4.b, ptr align 1 @Other.val.loc15_33.5, i64 0, i1 false), !dbg !89
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.44ca5843c95a3b8e(ptr %_.var), !dbg !86
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.edec27683d878fd6(ptr %o.var), !dbg !84
// CHECK:STDOUT:   ret i32 0, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fa5d2a8a45149476:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.b = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.edec27683d878fd6(ptr %.loc15_3.1.b), !dbg !94
// CHECK:STDOUT:   %.loc15_3.4.a = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbd3ab9dc2e81384(ptr %.loc15_3.4.a), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2fbae80a61a11ec7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.b = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.edec27683d878fd6(ptr %.loc15_3.1.b), !dbg !98
// CHECK:STDOUT:   %.loc15_3.4.a = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbd3ab9dc2e81384(ptr %.loc15_3.4.a), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.44ca5843c95a3b8e(ptr %t) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_COp.2fbae80a61a11ec7:core.SubobjectDestroy.Core"(ptr %t), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.edec27683d878fd6(ptr %t) #0 !dbg !107 {
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bbd3ab9dc2e81384(ptr %t) #0 !dbg !114 {
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.edec27683d878fd6, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.bbd3ab9dc2e81384, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!82}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !81 = !DIFile(filename: "inplace_init_with_nonconst.carbon", directory: "")
// CHECK:STDOUT: !82 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !81, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !81, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 12, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !DILocation(line: 15, column: 3, scope: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 15, column: 16, scope: !83)
// CHECK:STDOUT: !91 = !DILocation(line: 11, column: 1, scope: !83)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fa5d2a8a45149476:core.SubobjectDestroy.Core", scope: null, file: !81, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !21)
// CHECK:STDOUT: !94 = !DILocation(line: 15, column: 3, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2fbae80a61a11ec7:core.SubobjectDestroy.Core", scope: null, file: !81, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !21)
// CHECK:STDOUT: !98 = !DILocation(line: 15, column: 3, scope: !96)
// CHECK:STDOUT: !99 = !{!97}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.44ca5843c95a3b8e", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !106)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !21)
// CHECK:STDOUT: !104 = !DILocation(line: 28, column: 5, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 1, scope: !100)
// CHECK:STDOUT: !106 = !{!101}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.edec27683d878fd6", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !113)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !21)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 1, scope: !107)
// CHECK:STDOUT: !113 = !{!108}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bbd3ab9dc2e81384", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !120)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !21)
// CHECK:STDOUT: !119 = !DILocation(line: 26, column: 1, scope: !114)
// CHECK:STDOUT: !120 = !{!115}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_init_with_nonempty_nonconst.carbon'
// CHECK:STDOUT: source_filename = "implicit_init_with_nonempty_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc17_40.5 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca { i32 }, align 4, !dbg !124
// CHECK:STDOUT:   %_.var = alloca <{ ptr, { i32 } }>, align 8, !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !124
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !126
// CHECK:STDOUT:   %.loc17_40.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 0, !dbg !129
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc17_40.2.a, align 8, !dbg !129
// CHECK:STDOUT:   %.loc17_40.4.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 1, !dbg !129
// CHECK:STDOUT:   %.loc17_39.3.v = getelementptr inbounds nuw { i32 }, ptr %.loc17_40.4.b, i32 0, i32 0, !dbg !130
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc17_40.4.b, ptr align 4 @Other.val.loc17_40.5, i64 4, i1 false), !dbg !129
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.44ca5843c95a3b8e(ptr %_.var), !dbg !126
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.edec27683d878fd6(ptr %o.var), !dbg !124
// CHECK:STDOUT:   ret i32 0, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.92afc3a45a31c789:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc17_3.1.v), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6a519b95f6f059eb:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !139
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc17_3.1.v), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fa5d2a8a45149476:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 1, !dbg !143
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.edec27683d878fd6(ptr %.loc17_3.1.b), !dbg !143
// CHECK:STDOUT:   %.loc17_3.4.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 0, !dbg !143
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbd3ab9dc2e81384(ptr %.loc17_3.4.a), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2fbae80a61a11ec7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 1, !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.edec27683d878fd6(ptr %.loc17_3.1.b), !dbg !147
// CHECK:STDOUT:   %.loc17_3.4.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 0, !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbd3ab9dc2e81384(ptr %.loc17_3.4.a), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.44ca5843c95a3b8e(ptr %t) #0 !dbg !149 {
// CHECK:STDOUT:   call void @"_COp.2fbae80a61a11ec7:core.SubobjectDestroy.Core"(ptr %t), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.edec27683d878fd6(ptr %t) #0 !dbg !156 {
// CHECK:STDOUT:   call void @"_COp.6a519b95f6f059eb:core.SubobjectDestroy.Core"(ptr %t), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !163 {
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bbd3ab9dc2e81384(ptr %t) #0 !dbg !170 {
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.edec27683d878fd6, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.bbd3ab9dc2e81384, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!122}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !53 = !{null, !20}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !121 = !DIFile(filename: "implicit_init_with_nonempty_nonconst.carbon", directory: "")
// CHECK:STDOUT: !122 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !121, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !121, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !124 = !DILocation(line: 14, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !DILocation(line: 17, column: 3, scope: !123)
// CHECK:STDOUT: !129 = !DILocation(line: 17, column: 16, scope: !123)
// CHECK:STDOUT: !130 = !DILocation(line: 17, column: 31, scope: !123)
// CHECK:STDOUT: !132 = !DILocation(line: 13, column: 1, scope: !123)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Op", linkageName: "_COp.92afc3a45a31c789:core.SubobjectDestroy.Core", scope: null, file: !121, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !21)
// CHECK:STDOUT: !135 = !DILocation(line: 17, column: 3, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6a519b95f6f059eb:core.SubobjectDestroy.Core", scope: null, file: !121, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !21)
// CHECK:STDOUT: !139 = !DILocation(line: 17, column: 3, scope: !137)
// CHECK:STDOUT: !140 = !{!138}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fa5d2a8a45149476:core.SubobjectDestroy.Core", scope: null, file: !121, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !21)
// CHECK:STDOUT: !143 = !DILocation(line: 17, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2fbae80a61a11ec7:core.SubobjectDestroy.Core", scope: null, file: !121, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !21)
// CHECK:STDOUT: !147 = !DILocation(line: 17, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.44ca5843c95a3b8e", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !155)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !21)
// CHECK:STDOUT: !153 = !DILocation(line: 28, column: 5, scope: !149)
// CHECK:STDOUT: !154 = !DILocation(line: 26, column: 1, scope: !149)
// CHECK:STDOUT: !155 = !{!150}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.edec27683d878fd6", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !162)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !21)
// CHECK:STDOUT: !160 = !DILocation(line: 28, column: 5, scope: !156)
// CHECK:STDOUT: !161 = !DILocation(line: 26, column: 1, scope: !156)
// CHECK:STDOUT: !162 = !{!157}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !54, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !169)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !20)
// CHECK:STDOUT: !168 = !DILocation(line: 26, column: 1, scope: !163)
// CHECK:STDOUT: !169 = !{!164}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bbd3ab9dc2e81384", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !176)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !21)
// CHECK:STDOUT: !175 = !DILocation(line: 26, column: 1, scope: !170)
// CHECK:STDOUT: !176 = !{!171}
// CHECK:STDOUT:
