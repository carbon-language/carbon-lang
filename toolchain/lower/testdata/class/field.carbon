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
// CHECK:STDOUT: define i32 @_CF.Main(ptr %c) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_13.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %c, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   %.loc10_13.2 = load ptr, ptr %.loc10_13.1.b, align 8, !dbg !35
// CHECK:STDOUT:   %.loc10_16.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %.loc10_13.2, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %.loc10_16.2 = load i32, ptr %.loc10_16.1.a, align 4, !dbg !37
// CHECK:STDOUT:   ret i32 %.loc10_16.2, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !43
// CHECK:STDOUT:   %.loc15_4.a = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   store i32 1, ptr %.loc15_4.a, align 4, !dbg !45
// CHECK:STDOUT:   %.loc16.b = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   store ptr %c.var, ptr %.loc16.b, align 8, !dbg !46
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr %c.var), !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %c.var), !dbg !43
// CHECK:STDOUT:   ret i32 %F.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !69
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
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !21)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 12, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 10, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 10, column: 3, scope: !30)
// CHECK:STDOUT: !39 = !{!31}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 15, column: 3, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 16, column: 3, scope: !40)
// CHECK:STDOUT: !50 = !DILocation(line: 17, column: 10, scope: !40)
// CHECK:STDOUT: !51 = !DILocation(line: 17, column: 3, scope: !40)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 14, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 14, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !21)
// CHECK:STDOUT: !62 = !DILocation(line: 14, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !70)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !68 = !DILocation(line: 26, column: 5, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 24, column: 3, scope: !64)
// CHECK:STDOUT: !70 = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'inplace_init_with_nonconst.carbon'
// CHECK:STDOUT: source_filename = "inplace_init_with_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc15_33.5 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca {}, align 1, !dbg !74
// CHECK:STDOUT:   %_.var = alloca { ptr, {} }, align 8, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !76
// CHECK:STDOUT:   %.loc15_33.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 0, !dbg !79
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc15_33.2.a, align 8, !dbg !79
// CHECK:STDOUT:   %.loc15_33.4.b = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 1, !dbg !79
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc15_33.4.b, ptr align 1 @Other.val.loc15_33.5, i64 0, i1 false), !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c91373ef3ed1ea00(ptr %_.var), !dbg !76
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dccf856d95ac605b(ptr %o.var), !dbg !74
// CHECK:STDOUT:   ret i32 0, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c91373ef3ed1ea00(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.dccf856d95ac605b(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!72}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !71 = !DIFile(filename: "inplace_init_with_nonconst.carbon", directory: "")
// CHECK:STDOUT: !72 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !71, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !71, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 12, column: 3, scope: !73)
// CHECK:STDOUT: !76 = !DILocation(line: 15, column: 3, scope: !73)
// CHECK:STDOUT: !79 = !DILocation(line: 15, column: 16, scope: !73)
// CHECK:STDOUT: !81 = !DILocation(line: 11, column: 1, scope: !73)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !71, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !21)
// CHECK:STDOUT: !84 = !DILocation(line: 15, column: 3, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !71, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !21)
// CHECK:STDOUT: !88 = !DILocation(line: 15, column: 3, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !71, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !93)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !21)
// CHECK:STDOUT: !92 = !DILocation(line: 15, column: 3, scope: !90)
// CHECK:STDOUT: !93 = !{!91}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c91373ef3ed1ea00", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !100)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !21)
// CHECK:STDOUT: !98 = !DILocation(line: 26, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 24, column: 3, scope: !94)
// CHECK:STDOUT: !100 = !{!95}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.dccf856d95ac605b", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !107)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !21)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 24, column: 3, scope: !101)
// CHECK:STDOUT: !107 = !{!102}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_init_with_nonempty_nonconst.carbon'
// CHECK:STDOUT: source_filename = "implicit_init_with_nonempty_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc17_40.5 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca { i32 }, align 4, !dbg !111
// CHECK:STDOUT:   %_.var = alloca <{ ptr, { i32 } }>, align 8, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !113
// CHECK:STDOUT:   %.loc17_40.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc17_40.2.a, align 8, !dbg !116
// CHECK:STDOUT:   %.loc17_40.4.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 1, !dbg !116
// CHECK:STDOUT:   %.loc17_39.3.v = getelementptr inbounds nuw { i32 }, ptr %.loc17_40.4.b, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc17_40.4.b, ptr align 4 @Other.val.loc17_40.5, i64 4, i1 false), !dbg !116
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c91373ef3ed1ea00(ptr %_.var), !dbg !113
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dccf856d95ac605b(ptr %o.var), !dbg !111
// CHECK:STDOUT:   ret i32 0, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core"(ptr %self) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c91373ef3ed1ea00(ptr %self) #0 !dbg !140 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.dccf856d95ac605b(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!109}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !108 = !DIFile(filename: "implicit_init_with_nonempty_nonconst.carbon", directory: "")
// CHECK:STDOUT: !109 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !108, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !108, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !109)
// CHECK:STDOUT: !111 = !DILocation(line: 14, column: 3, scope: !110)
// CHECK:STDOUT: !113 = !DILocation(line: 17, column: 3, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 17, column: 16, scope: !110)
// CHECK:STDOUT: !117 = !DILocation(line: 17, column: 31, scope: !110)
// CHECK:STDOUT: !119 = !DILocation(line: 13, column: 1, scope: !110)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !108, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !123)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !20)
// CHECK:STDOUT: !122 = !DILocation(line: 17, column: 3, scope: !120)
// CHECK:STDOUT: !123 = !{!121}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core", scope: null, file: !108, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !127)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !21)
// CHECK:STDOUT: !126 = !DILocation(line: 17, column: 3, scope: !124)
// CHECK:STDOUT: !127 = !{!125}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !108, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !21)
// CHECK:STDOUT: !130 = !DILocation(line: 17, column: 3, scope: !128)
// CHECK:STDOUT: !131 = !{!129}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !108, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !135)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !21)
// CHECK:STDOUT: !134 = !DILocation(line: 17, column: 3, scope: !132)
// CHECK:STDOUT: !135 = !{!133}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !108, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !139)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !21)
// CHECK:STDOUT: !138 = !DILocation(line: 17, column: 3, scope: !136)
// CHECK:STDOUT: !139 = !{!137}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c91373ef3ed1ea00", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !146)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !21)
// CHECK:STDOUT: !144 = !DILocation(line: 26, column: 5, scope: !140)
// CHECK:STDOUT: !145 = !DILocation(line: 24, column: 3, scope: !140)
// CHECK:STDOUT: !146 = !{!141}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.dccf856d95ac605b", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !153)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !21)
// CHECK:STDOUT: !151 = !DILocation(line: 26, column: 5, scope: !147)
// CHECK:STDOUT: !152 = !DILocation(line: 24, column: 3, scope: !147)
// CHECK:STDOUT: !153 = !{!148}
// CHECK:STDOUT:
