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
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !43
// CHECK:STDOUT:   ret i32 %F.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !62
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc14_3.1.b), !dbg !62
// CHECK:STDOUT:   %.loc14_3.2.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !62
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc14_3.2.a), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc14_3.1.b), !dbg !70
// CHECK:STDOUT:   %.loc14_3.2.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc14_3.2.a), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
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
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 14, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 14, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !21)
// CHECK:STDOUT: !62 = !DILocation(line: 14, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 14, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 14, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !21)
// CHECK:STDOUT: !74 = !DILocation(line: 14, column: 3, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'inplace_init_with_nonconst.carbon'
// CHECK:STDOUT: source_filename = "inplace_init_with_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc15_33.5 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca {}, align 1, !dbg !79
// CHECK:STDOUT:   %_.var = alloca { ptr, {} }, align 8, !dbg !81
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !81
// CHECK:STDOUT:   %.loc15_33.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 0, !dbg !84
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc15_33.2.a, align 8, !dbg !84
// CHECK:STDOUT:   %.loc15_33.4.b = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 1, !dbg !84
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc15_33.4.b, ptr align 1 @Other.val.loc15_33.5, i64 0, i1 false), !dbg !84
// CHECK:STDOUT:   call void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %_.var), !dbg !81
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %o.var), !dbg !79
// CHECK:STDOUT:   ret i32 0, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.b = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc15_3.1.b), !dbg !101
// CHECK:STDOUT:   %.loc15_3.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc15_3.2.a), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.b = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !109
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc15_3.1.b), !dbg !109
// CHECK:STDOUT:   %.loc15_3.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc15_3.2.a), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!77}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !76 = !DIFile(filename: "inplace_init_with_nonconst.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !76, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !76, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !77)
// CHECK:STDOUT: !79 = !DILocation(line: 12, column: 3, scope: !78)
// CHECK:STDOUT: !81 = !DILocation(line: 15, column: 3, scope: !78)
// CHECK:STDOUT: !84 = !DILocation(line: 15, column: 16, scope: !78)
// CHECK:STDOUT: !86 = !DILocation(line: 11, column: 1, scope: !78)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !21)
// CHECK:STDOUT: !89 = !DILocation(line: 15, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !21)
// CHECK:STDOUT: !93 = !DILocation(line: 15, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !21)
// CHECK:STDOUT: !97 = !DILocation(line: 15, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !21)
// CHECK:STDOUT: !101 = !DILocation(line: 15, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !21)
// CHECK:STDOUT: !105 = !DILocation(line: 15, column: 3, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !21)
// CHECK:STDOUT: !109 = !DILocation(line: 15, column: 3, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !76, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !21)
// CHECK:STDOUT: !113 = !DILocation(line: 15, column: 3, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_init_with_nonempty_nonconst.carbon'
// CHECK:STDOUT: source_filename = "implicit_init_with_nonempty_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc17_40.5 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca { i32 }, align 4, !dbg !118
// CHECK:STDOUT:   %_.var = alloca <{ ptr, { i32 } }>, align 8, !dbg !120
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !120
// CHECK:STDOUT:   %.loc17_40.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 0, !dbg !123
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc17_40.2.a, align 8, !dbg !123
// CHECK:STDOUT:   %.loc17_40.4.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 1, !dbg !123
// CHECK:STDOUT:   %.loc17_39.3.v = getelementptr inbounds nuw { i32 }, ptr %.loc17_40.4.b, i32 0, i32 0, !dbg !124
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc17_40.4.b, ptr align 4 @Other.val.loc17_40.5, i64 4, i1 false), !dbg !123
// CHECK:STDOUT:   call void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %_.var), !dbg !120
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %o.var), !dbg !118
// CHECK:STDOUT:   ret i32 0, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %self) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core"(ptr %self) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !137
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc17.v), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.92afc3a45a31c789:core.Destroy.Core"(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core"(ptr %self), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !145
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc17.v), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 1, !dbg !153
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc17_3.1.b), !dbg !153
// CHECK:STDOUT:   %.loc17_3.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 0, !dbg !153
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc17_3.2.a), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 1, !dbg !161
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc17_3.1.b), !dbg !161
// CHECK:STDOUT:   %.loc17_3.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 0, !dbg !161
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc17_3.2.a), !dbg !161
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!116}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !115 = !DIFile(filename: "implicit_init_with_nonempty_nonconst.carbon", directory: "")
// CHECK:STDOUT: !116 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !115, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !115, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !116)
// CHECK:STDOUT: !118 = !DILocation(line: 14, column: 3, scope: !117)
// CHECK:STDOUT: !120 = !DILocation(line: 17, column: 3, scope: !117)
// CHECK:STDOUT: !123 = !DILocation(line: 17, column: 16, scope: !117)
// CHECK:STDOUT: !124 = !DILocation(line: 17, column: 31, scope: !117)
// CHECK:STDOUT: !126 = !DILocation(line: 13, column: 1, scope: !117)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !130)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !21)
// CHECK:STDOUT: !129 = !DILocation(line: 17, column: 3, scope: !127)
// CHECK:STDOUT: !130 = !{!128}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !115, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !20)
// CHECK:STDOUT: !133 = !DILocation(line: 17, column: 3, scope: !131)
// CHECK:STDOUT: !134 = !{!132}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !21)
// CHECK:STDOUT: !137 = !DILocation(line: 17, column: 3, scope: !135)
// CHECK:STDOUT: !138 = !{!136}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.92afc3a45a31c789:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !21)
// CHECK:STDOUT: !141 = !DILocation(line: 17, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !21)
// CHECK:STDOUT: !145 = !DILocation(line: 17, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !{!144}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !150)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !21)
// CHECK:STDOUT: !149 = !DILocation(line: 17, column: 3, scope: !147)
// CHECK:STDOUT: !150 = !{!148}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !21)
// CHECK:STDOUT: !153 = !DILocation(line: 17, column: 3, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !21)
// CHECK:STDOUT: !157 = !DILocation(line: 17, column: 3, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !21)
// CHECK:STDOUT: !161 = !DILocation(line: 17, column: 3, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !115, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !21)
// CHECK:STDOUT: !165 = !DILocation(line: 17, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT:
