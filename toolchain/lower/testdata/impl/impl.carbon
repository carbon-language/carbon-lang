// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/impl.carbon

// --- impl.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(self) -> i32;
}

class A {
  var n: i32;
}

impl A as I {
  fn F(self: A) -> i32 {
    return self.n;
  }
}

fn Call(a: A) -> i32 {
  return a.(I.F)();
}

// --- generic_impls.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
class C {}

interface I(T: type) {
  fn F(self) -> T;
}

impl C as I(A) {
  fn F(unused self: C) -> A { return {}; }
}

impl C as I(B) {
  fn F(unused self: C) -> B { return {}; }
}

fn F1(x: C, y: C) {
  var unused a: A = x.(I(A).F)();
  var unused b: B = y.(I(B).F)();
}

fn F2[T: I(A), U: I(B)](x: T, y: U) {
  var unused a: A = x.F();
  var unused b: B = y.F();
}

// --- generic_class_impls.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

class C(T: type) {}

interface I {
  fn F();
}

impl C(A) as I {
  fn F() {}
}

impl C(B) as I {
  fn F() {}
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'impl.carbon'
// CHECK:STDOUT: source_filename = "impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc14_16.2 = load i32, ptr %.loc14_16.1.n, align 4, !dbg !29
// CHECK:STDOUT:   ret i32 %.loc14_16.2, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !37
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "impl.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !18, line: 13, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !31)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 12, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 5, scope: !24)
// CHECK:STDOUT: !31 = !{!25}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !18, line: 18, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 10, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 3, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_impls.carbon'
// CHECK:STDOUT: source_filename = "generic_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc13_40 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc17_40 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.bd4fb75a414fe858.Main"(ptr sret({}) %return, ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc13_40, i64 0, i1 false), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.a6a60dfee2d7d273.Main"(ptr sret({}) %return, ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc17_40, i64 0, i1 false), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF1.Main(ptr %x, ptr %y) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !67
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !67
// CHECK:STDOUT:   call void @"_CF.C.Main:I.bd4fb75a414fe858.Main"(ptr %a.var, ptr %x), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !70
// CHECK:STDOUT:   call void @"_CF.C.Main:I.a6a60dfee2d7d273.Main"(ptr %b.var, ptr %y), !dbg !71
// CHECK:STDOUT:   call void @"_COp.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %b.var), !dbg !70
// CHECK:STDOUT:   call void @"_COp.5a23cc09585c55d2:core.Destroy.Core"(ptr %a.var), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5a23cc09585c55d2:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !DIFile(filename: "generic_impls.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !{!21, !21}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{null, !21, !21}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !{null, !21}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.bd4fb75a414fe858.Main", scope: null, file: !40, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !21)
// CHECK:STDOUT: !53 = !DILocation(line: 13, column: 31, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.a6a60dfee2d7d273.Main", scope: null, file: !40, line: 17, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !21)
// CHECK:STDOUT: !60 = !DILocation(line: 17, column: 31, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "F1", linkageName: "_CF1.Main", scope: null, file: !40, line: 20, type: !45, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !74)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !21)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !62, type: !21)
// CHECK:STDOUT: !67 = !DILocation(line: 21, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 21, column: 21, scope: !62)
// CHECK:STDOUT: !70 = !DILocation(line: 22, column: 3, scope: !62)
// CHECK:STDOUT: !71 = !DILocation(line: 22, column: 21, scope: !62)
// CHECK:STDOUT: !73 = !DILocation(line: 20, column: 1, scope: !62)
// CHECK:STDOUT: !74 = !{!63, !64}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !40, line: 22, type: !47, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !21)
// CHECK:STDOUT: !77 = !DILocation(line: 22, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5a23cc09585c55d2:core.Destroy.Core", scope: null, file: !40, line: 21, type: !47, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !21)
// CHECK:STDOUT: !81 = !DILocation(line: 21, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_class_impls.carbon'
// CHECK:STDOUT: source_filename = "generic_class_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.bd4fb75a414fe858.Main:I.Main"() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.a6a60dfee2d7d273.Main:I.Main"() #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !83 = !DIFile(filename: "generic_class_impls.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = !{null}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.bd4fb75a414fe858.Main:I.Main", scope: null, file: !83, line: 14, type: !86, spFlags: DISPFlagDefinition, unit: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 14, column: 3, scope: !87)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.a6a60dfee2d7d273.Main:I.Main", scope: null, file: !83, line: 18, type: !86, spFlags: DISPFlagDefinition, unit: !84)
// CHECK:STDOUT: !90 = !DILocation(line: 18, column: 3, scope: !89)
// CHECK:STDOUT:
