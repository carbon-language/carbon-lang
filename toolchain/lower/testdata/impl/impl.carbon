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
  fn F[self: Self]() -> i32;
}

class A {
  var n: i32;
}

impl A as I {
  fn F[self: A]() -> i32 {
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

interface I(T:! type) {
  fn F[self: Self]() -> T;
}

impl C as I(A) {
  fn F[unused self: C]() -> A { return {}; }
}

impl C as I(B) {
  fn F[unused self: C]() -> B { return {}; }
}

fn F1(x: C, y: C) {
  var unused a: A = x.(I(A).F)();
  var unused b: B = y.(I(B).F)();
}

fn F2[T:! I(A), U:! I(B)](x: T, y: U) {
  var unused a: A = x.F();
  var unused b: B = y.F();
}

// --- generic_class_impls.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

class C(T:! type) {}

interface I {
  fn F();
}

impl C(A) as I {
  fn F() {}
}

impl C(B) as I {
  fn F() {}
}

// CHECK:STDOUT: ; ModuleID = 'impl.carbon'
// CHECK:STDOUT: source_filename = "impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc14_16.2 = load i32, ptr %.loc14_16.1.n, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 %.loc14_16.2, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !16
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 12, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 5, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !8)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 10, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 3, scope: !13)
// CHECK:STDOUT: ; ModuleID = 'generic_impls.carbon'
// CHECK:STDOUT: source_filename = "generic_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc13_42 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc17_42 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.96947bccc1c5e61b.Main"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc13_42, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.707dbe5ae2414d29.Main"(ptr sret({}) %return, ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc17_42, i64 0, i1 false), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF1.Main(ptr %x, ptr %y) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 8, !dbg !21
// CHECK:STDOUT:   %b.var = alloca {}, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !21
// CHECK:STDOUT:   call void @"_CF.C.Main:I.96947bccc1c5e61b.Main"(ptr %a.var, ptr %x), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !22
// CHECK:STDOUT:   call void @"_CF.C.Main:I.707dbe5ae2414d29.Main"(ptr %b.var, ptr %y), !dbg !24
// CHECK:STDOUT:   call void @"_COp.301c2cde953858ec:core.Destroy.Core"(ptr %b.var), !dbg !22
// CHECK:STDOUT:   call void @"_COp.1afa5e4ee26f3f8f:core.Destroy.Core"(ptr %a.var), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.301c2cde953858ec:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1afa5e4ee26f3f8f:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_impls.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.96947bccc1c5e61b.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 33, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.707dbe5ae2414d29.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 17, column: 33, scope: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "F1", linkageName: "_CF1.Main", scope: null, file: !3, line: 20, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !7, !7}
// CHECK:STDOUT: !18 = !{!19, !20}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !15, type: !7)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 2, scope: !15, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 21, column: 3, scope: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 22, column: 3, scope: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 21, scope: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 21, scope: !15)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 1, scope: !15)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.301c2cde953858ec:core.Destroy.Core", scope: null, file: !3, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !7}
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 3, scope: !26)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1afa5e4ee26f3f8f:core.Destroy.Core", scope: null, file: !3, line: 21, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !35 = !DILocation(line: 21, column: 3, scope: !32)
// CHECK:STDOUT: ; ModuleID = 'generic_class_impls.carbon'
// CHECK:STDOUT: source_filename = "generic_class_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.96947bccc1c5e61b.Main:I.Main"() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.707dbe5ae2414d29.Main:I.Main"() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_class_impls.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.96947bccc1c5e61b.Main:I.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.707dbe5ae2414d29.Main:I.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 3, scope: !8)
