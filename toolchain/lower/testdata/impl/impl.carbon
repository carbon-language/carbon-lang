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
// CHECK:STDOUT: define void @"_CF.C.Main:I.bd4fb75a414fe858.Main"(ptr sret({}) %return, ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc13_40, i64 0, i1 false), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.a6a60dfee2d7d273.Main"(ptr sret({}) %return, ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc17_40, i64 0, i1 false), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF1.Main(ptr %x, ptr %y) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !65
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !65
// CHECK:STDOUT:   call void @"_CF.C.Main:I.bd4fb75a414fe858.Main"(ptr %a.var, ptr %x), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !68
// CHECK:STDOUT:   call void @"_CF.C.Main:I.a6a60dfee2d7d273.Main"(ptr %b.var, ptr %y), !dbg !69
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %b.var), !dbg !68
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %a.var), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %t) #0 !dbg !75 {
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %t) #0 !dbg !82 {
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !DIFile(filename: "generic_impls.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !{!21, !21}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{null, !21, !21}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.bd4fb75a414fe858.Main", scope: null, file: !40, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !52)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 13, column: 31, scope: !46)
// CHECK:STDOUT: !52 = !{!47}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.a6a60dfee2d7d273.Main", scope: null, file: !40, line: 17, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 17, column: 31, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "F1", linkageName: "_CF1.Main", scope: null, file: !40, line: 20, type: !45, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !74)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !21)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 2, scope: !60, type: !21)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 3, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 21, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 3, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 22, column: 21, scope: !60)
// CHECK:STDOUT: !71 = !{null, !21}
// CHECK:STDOUT: !72 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 20, column: 1, scope: !60)
// CHECK:STDOUT: !74 = !{!61, !62}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b963e75af1c16acd", scope: null, file: !10, line: 30, type: !72, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !81)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !21)
// CHECK:STDOUT: !80 = !DILocation(line: 30, column: 1, scope: !75)
// CHECK:STDOUT: !81 = !{!76}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e517637b3b58cb2c", scope: null, file: !10, line: 30, type: !72, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !88)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !21)
// CHECK:STDOUT: !87 = !DILocation(line: 30, column: 1, scope: !82)
// CHECK:STDOUT: !88 = !{!83}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_class_impls.carbon'
// CHECK:STDOUT: source_filename = "generic_class_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.bd4fb75a414fe858.Main:I.Main"() #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.a6a60dfee2d7d273.Main:I.Main"() #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!90}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !89 = !DIFile(filename: "generic_class_impls.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !89, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !91 = !{null}
// CHECK:STDOUT: !92 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.bd4fb75a414fe858.Main:I.Main", scope: null, file: !89, line: 14, type: !92, spFlags: DISPFlagDefinition, unit: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 14, column: 3, scope: !93)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.a6a60dfee2d7d273.Main:I.Main", scope: null, file: !89, line: 18, type: !92, spFlags: DISPFlagDefinition, unit: !90)
// CHECK:STDOUT: !96 = !DILocation(line: 18, column: 3, scope: !95)
// CHECK:STDOUT:
