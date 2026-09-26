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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc14_3.1.a), !dbg !70
// CHECK:STDOUT:   %.loc14_3.2.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc14_3.2.b), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !78
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc14_3.1.a), !dbg !78
// CHECK:STDOUT:   %.loc14_3.2.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !78
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc14_3.2.b), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !82
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
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 14, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !20)
// CHECK:STDOUT: !58 = !DILocation(line: 14, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 14, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 14, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 14, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !21)
// CHECK:STDOUT: !74 = !DILocation(line: 14, column: 3, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !21)
// CHECK:STDOUT: !78 = !DILocation(line: 14, column: 3, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !21)
// CHECK:STDOUT: !82 = !DILocation(line: 14, column: 3, scope: !80)
// CHECK:STDOUT: !83 = !{!81}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'inplace_init_with_nonconst.carbon'
// CHECK:STDOUT: source_filename = "inplace_init_with_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc15_33.5 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca {}, align 1, !dbg !87
// CHECK:STDOUT:   %_.var = alloca { ptr, {} }, align 8, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !89
// CHECK:STDOUT:   %.loc15_33.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 0, !dbg !92
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc15_33.2.a, align 8, !dbg !92
// CHECK:STDOUT:   %.loc15_33.4.b = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 1, !dbg !92
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc15_33.4.b, ptr align 1 @Other.val.loc15_33.5, i64 0, i1 false), !dbg !92
// CHECK:STDOUT:   call void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %_.var), !dbg !89
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %o.var), !dbg !87
// CHECK:STDOUT:   ret i32 0, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.a = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !113
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc15_3.1.a), !dbg !113
// CHECK:STDOUT:   %.loc15_3.2.b = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !113
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc15_3.2.b), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.a = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !121
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc15_3.1.a), !dbg !121
// CHECK:STDOUT:   %.loc15_3.2.b = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !121
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc15_3.2.b), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !125
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!85}
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
// CHECK:STDOUT: !84 = !DIFile(filename: "inplace_init_with_nonconst.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !84, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !84, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !85)
// CHECK:STDOUT: !87 = !DILocation(line: 12, column: 3, scope: !86)
// CHECK:STDOUT: !89 = !DILocation(line: 15, column: 3, scope: !86)
// CHECK:STDOUT: !92 = !DILocation(line: 15, column: 16, scope: !86)
// CHECK:STDOUT: !94 = !DILocation(line: 11, column: 1, scope: !86)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !21)
// CHECK:STDOUT: !97 = !DILocation(line: 15, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !21)
// CHECK:STDOUT: !101 = !DILocation(line: 15, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !21)
// CHECK:STDOUT: !105 = !DILocation(line: 15, column: 3, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !21)
// CHECK:STDOUT: !109 = !DILocation(line: 15, column: 3, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !21)
// CHECK:STDOUT: !113 = !DILocation(line: 15, column: 3, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !118)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !21)
// CHECK:STDOUT: !117 = !DILocation(line: 15, column: 3, scope: !115)
// CHECK:STDOUT: !118 = !{!116}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !21)
// CHECK:STDOUT: !121 = !DILocation(line: 15, column: 3, scope: !119)
// CHECK:STDOUT: !122 = !{!120}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !84, line: 15, type: !29, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !126)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !21)
// CHECK:STDOUT: !125 = !DILocation(line: 15, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !{!124}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_init_with_nonempty_nonconst.carbon'
// CHECK:STDOUT: source_filename = "implicit_init_with_nonempty_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc17_40.5 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca { i32 }, align 4, !dbg !130
// CHECK:STDOUT:   %_.var = alloca <{ ptr, { i32 } }>, align 8, !dbg !132
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !130
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !132
// CHECK:STDOUT:   %.loc17_40.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc17_40.2.a, align 8, !dbg !135
// CHECK:STDOUT:   %.loc17_40.4.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 1, !dbg !135
// CHECK:STDOUT:   %.loc17_39.3.v = getelementptr inbounds nuw { i32 }, ptr %.loc17_40.4.b, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc17_40.4.b, ptr align 4 @Other.val.loc17_40.5, i64 4, i1 false), !dbg !135
// CHECK:STDOUT:   call void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %_.var), !dbg !132
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %o.var), !dbg !130
// CHECK:STDOUT:   ret i32 0, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core"(ptr %self) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !157
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc17.v), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.92afc3a45a31c789:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core"(ptr %self), !dbg !161
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc17.v), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core"(ptr %self), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 0, !dbg !173
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc17_3.1.a), !dbg !173
// CHECK:STDOUT:   %.loc17_3.2.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 1, !dbg !173
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc17_3.2.b), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core"(ptr %self) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core"(ptr %self), !dbg !177
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 0, !dbg !181
// CHECK:STDOUT:   call void @"_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core"(ptr %.loc17_3.1.a), !dbg !181
// CHECK:STDOUT:   %.loc17_3.2.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %self, i32 0, i32 1, !dbg !181
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core"(ptr %.loc17_3.2.b), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core"(ptr %self), !dbg !185
// CHECK:STDOUT:   ret void, !dbg !185
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!128}
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
// CHECK:STDOUT: !127 = !DIFile(filename: "implicit_init_with_nonempty_nonconst.carbon", directory: "")
// CHECK:STDOUT: !128 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !127, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !127, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !130 = !DILocation(line: 14, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !DILocation(line: 17, column: 3, scope: !129)
// CHECK:STDOUT: !135 = !DILocation(line: 17, column: 16, scope: !129)
// CHECK:STDOUT: !136 = !DILocation(line: 17, column: 31, scope: !129)
// CHECK:STDOUT: !138 = !DILocation(line: 13, column: 1, scope: !129)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d7c3fbe4c84d0842:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !21)
// CHECK:STDOUT: !141 = !DILocation(line: 17, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !127, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !20)
// CHECK:STDOUT: !145 = !DILocation(line: 17, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !{!144}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !127, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !150)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !20)
// CHECK:STDOUT: !149 = !DILocation(line: 17, column: 3, scope: !147)
// CHECK:STDOUT: !150 = !{!148}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !127, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !20)
// CHECK:STDOUT: !153 = !DILocation(line: 17, column: 3, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.92afc3a45a31c789:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !21)
// CHECK:STDOUT: !157 = !DILocation(line: 17, column: 3, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.92afc3a45a31c789:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !21)
// CHECK:STDOUT: !161 = !DILocation(line: 17, column: 3, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !21)
// CHECK:STDOUT: !165 = !DILocation(line: 17, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a519b95f6f059eb:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !21)
// CHECK:STDOUT: !169 = !DILocation(line: 17, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !21)
// CHECK:STDOUT: !173 = !DILocation(line: 17, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fa5d2a8a45149476:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !21)
// CHECK:STDOUT: !177 = !DILocation(line: 17, column: 3, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !182)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !21)
// CHECK:STDOUT: !181 = !DILocation(line: 17, column: 3, scope: !179)
// CHECK:STDOUT: !182 = !{!180}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2fbae80a61a11ec7:core.Destroy.Core", scope: null, file: !127, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !186)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !21)
// CHECK:STDOUT: !185 = !DILocation(line: 17, column: 3, scope: !183)
// CHECK:STDOUT: !186 = !{!184}
// CHECK:STDOUT:
