// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/thunk.carbon

// --- thunk.carbon

library "[[@TEST_NAME]]";

class A { var a: i32; }
class B { var b: i32; }
class C { var c: i32; }

impl A as Core.ImplicitAs(B) {
  fn Convert(self: A) -> B { return {.b = self.a}; }
}
impl B as Core.ImplicitAs(C) {
  fn Convert(self: B) -> C { return {.c = self.b}; }
}

interface I {
  fn F(a: A) -> C;
}

impl () as I {
  fn F(b: B) -> B { return {.b = b.b}; }
}

fn Test(a: A) -> C {
  return ().(I.F)(a);
}

// --- generic_thunk.carbon

library "[[@TEST_NAME]]";

class A {}
class B {
  impl as Core.ImplicitAs(A) {
    fn Convert(unused self) -> A { return {}; }
  }
}

interface I(T: type) {
  fn F(self, x: B) -> A;
}

class C(U: type) {}

impl forall [V: type] C(V) as I(V) {
  fn F(unused self, _: A) -> B { return {}; }
}

fn Call(c: C(()), b: B) -> A {
  return c.(I(()).F)(b);
}

fn CallGeneric[T: I(())](c: T, b: B) -> A {
  return c.F(b);
}

fn CallCallGeneric(c: C(()), b: B) -> A {
  return CallGeneric(c, b);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc9_47.2 = load i32, ptr %.loc9_47.1.a, align 4, !dbg !33
// CHECK:STDOUT:   %.loc9_49.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   store i32 %.loc9_47.2, ptr %.loc9_49.2.b, align 4, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_47.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %.loc12_47.2 = load i32, ptr %.loc12_47.1.b, align 4, !dbg !42
// CHECK:STDOUT:   %.loc12_49.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   store i32 %.loc12_47.2, ptr %.loc12_49.2.c, align 4, !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !51
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 4, !dbg !58
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !57
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !57
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.4.temp), !dbg !58
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.4.temp), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_20.1.temp = alloca { i32 }, align 4, !dbg !78
// CHECK:STDOUT:   %.loc24_20.2.temp = alloca { i32 }, align 4, !dbg !78
// CHECK:STDOUT:   %.loc24_19.1.temp = alloca { i32 }, align 4, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.1.temp), !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.2.temp), !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_19.1.temp), !dbg !77
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc24_19.1.temp, ptr %a), !dbg !77
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc24_20.2.temp, ptr %.loc24_19.1.temp), !dbg !78
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc24_20.2.temp), !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc24_20.2.temp), !dbg !78
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc24_19.1.temp), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6a6353dfb36934de, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !36)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 9, column: 43, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 37, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 30, scope: !28)
// CHECK:STDOUT: !36 = !{!29}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !21)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 43, scope: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 12, column: 37, scope: !37)
// CHECK:STDOUT: !44 = !DILocation(line: 12, column: 30, scope: !37)
// CHECK:STDOUT: !45 = !{!38}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 34, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 20, column: 28, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 20, column: 21, scope: !46)
// CHECK:STDOUT: !54 = !{!47}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !21)
// CHECK:STDOUT: !57 = !DILocation(line: 0, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 20, column: 3, scope: !55)
// CHECK:STDOUT: !59 = !{!56}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 20, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 20, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !18, line: 23, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !80)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !21)
// CHECK:STDOUT: !77 = !DILocation(line: 24, column: 19, scope: !72)
// CHECK:STDOUT: !78 = !DILocation(line: 24, column: 10, scope: !72)
// CHECK:STDOUT: !79 = !DILocation(line: 24, column: 3, scope: !72)
// CHECK:STDOUT: !80 = !{!73}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !21)
// CHECK:STDOUT: !85 = !DILocation(line: 26, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 24, column: 3, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_thunk.carbon'
// CHECK:STDOUT: source_filename = "generic_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc7_45 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc18_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc7_45, i64 0, i1 false), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5a23cc09585c55d2:core.Destroy.Core"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.1.temp = alloca {}, align 1, !dbg !114
// CHECK:STDOUT:   %.loc22_23.3.temp = alloca {}, align 1, !dbg !114
// CHECK:STDOUT:   %.loc22_22.1.temp = alloca {}, align 1, !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.1.temp), !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.3.temp), !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_22.1.temp), !dbg !115
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.loc22_22.1.temp, ptr %b), !dbg !115
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc22_23.3.temp, ptr %c, ptr %.loc22_22.1.temp), !dbg !114
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc22_23.3.temp), !dbg !116
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc22_23.3.temp), !dbg !114
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15(ptr %.loc22_22.1.temp), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallGeneric.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallGeneric.Main.8a681e94b982e379(ptr %return, ptr %c, ptr %b), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %_) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc18_43, i64 0, i1 false), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15(ptr %self) #0 !dbg !146 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5a23cc09585c55d2:core.Destroy.Core"(ptr %self), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallGeneric.Main.8a681e94b982e379(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk:I.eca2102529696f40.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %return, ptr %c, ptr %b), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF:thunk:I.eca2102529696f40.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) #2 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.2.temp = alloca {}, align 1, !dbg !168
// CHECK:STDOUT:   %.6.temp = alloca {}, align 1, !dbg !167
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.2.temp), !dbg !168
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !167
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.6.temp, ptr %x), !dbg !167
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc18_32.2.temp, ptr %self, ptr %.6.temp), !dbg !168
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc18_32.2.temp), !dbg !168
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc18_32.2.temp), !dbg !168
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15(ptr %.6.temp), !dbg !167
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!89}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !88 = !DIFile(filename: "generic_thunk.carbon", directory: "")
// CHECK:STDOUT: !89 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !88, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !90 = !{!21, !21, !21}
// CHECK:STDOUT: !91 = !DISubroutineType(types: !90)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !88, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !98)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !21)
// CHECK:STDOUT: !97 = !DILocation(line: 7, column: 36, scope: !92)
// CHECK:STDOUT: !98 = !{!93}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !88, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !21)
// CHECK:STDOUT: !101 = !DILocation(line: 18, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5a23cc09585c55d2:core.Destroy.Core", scope: null, file: !88, type: !27, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !21)
// CHECK:STDOUT: !105 = !DILocation(line: 0, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !88, line: 21, type: !91, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !117)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !21)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 2, scope: !107, type: !21)
// CHECK:STDOUT: !114 = !DILocation(line: 22, column: 10, scope: !107)
// CHECK:STDOUT: !115 = !DILocation(line: 22, column: 22, scope: !107)
// CHECK:STDOUT: !116 = !DILocation(line: 22, column: 3, scope: !107)
// CHECK:STDOUT: !117 = !{!108, !109}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "CallCallGeneric", linkageName: "_CCallCallGeneric.Main", scope: null, file: !88, line: 29, type: !91, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !129)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !21)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 2, scope: !118, type: !21)
// CHECK:STDOUT: !127 = !DILocation(line: 30, column: 10, scope: !118)
// CHECK:STDOUT: !128 = !DILocation(line: 30, column: 3, scope: !118)
// CHECK:STDOUT: !129 = !{!119, !120}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !88, line: 18, type: !91, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !138)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !21)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 2, scope: !130, type: !21)
// CHECK:STDOUT: !137 = !DILocation(line: 18, column: 34, scope: !130)
// CHECK:STDOUT: !138 = !{!131, !132}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !145)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !21)
// CHECK:STDOUT: !143 = !DILocation(line: 26, column: 5, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 24, column: 3, scope: !139)
// CHECK:STDOUT: !145 = !{!140}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !152)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !21)
// CHECK:STDOUT: !150 = !DILocation(line: 26, column: 5, scope: !146)
// CHECK:STDOUT: !151 = !DILocation(line: 24, column: 3, scope: !146)
// CHECK:STDOUT: !152 = !{!147}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main.8a681e94b982e379", scope: null, file: !88, line: 25, type: !91, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !163)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !21)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 2, scope: !153, type: !21)
// CHECK:STDOUT: !160 = !DILocation(line: 26, column: 10, scope: !153)
// CHECK:STDOUT: !162 = !DILocation(line: 26, column: 3, scope: !153)
// CHECK:STDOUT: !163 = !{!154, !155}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.eca2102529696f40.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !88, line: 18, type: !91, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !169)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !21)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 2, scope: !164, type: !21)
// CHECK:STDOUT: !167 = !DILocation(line: 0, scope: !164)
// CHECK:STDOUT: !168 = !DILocation(line: 18, column: 3, scope: !164)
// CHECK:STDOUT: !169 = !{!165, !166}
// CHECK:STDOUT:
