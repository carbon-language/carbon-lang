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
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.4.temp), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc20.b), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !74
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc20.b), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_20.1.temp = alloca { i32 }, align 4, !dbg !86
// CHECK:STDOUT:   %.loc24_20.2.temp = alloca { i32 }, align 4, !dbg !86
// CHECK:STDOUT:   %.loc24_19.1.temp = alloca { i32 }, align 4, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.1.temp), !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.2.temp), !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_19.1.temp), !dbg !85
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc24_19.1.temp, ptr %a), !dbg !85
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc24_20.2.temp, ptr %.loc24_19.1.temp), !dbg !86
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc24_20.2.temp), !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc24_20.2.temp), !dbg !86
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc24_19.1.temp), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 20, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 20, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !21)
// CHECK:STDOUT: !74 = !DILocation(line: 20, column: 3, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !21)
// CHECK:STDOUT: !78 = !DILocation(line: 20, column: 3, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !18, line: 23, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !88)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !21)
// CHECK:STDOUT: !85 = !DILocation(line: 24, column: 19, scope: !80)
// CHECK:STDOUT: !86 = !DILocation(line: 24, column: 10, scope: !80)
// CHECK:STDOUT: !87 = !DILocation(line: 24, column: 3, scope: !80)
// CHECK:STDOUT: !88 = !{!81}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_thunk.carbon'
// CHECK:STDOUT: source_filename = "generic_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc7_45 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc18_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc7_45, i64 0, i1 false), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core"(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.1.temp = alloca {}, align 1, !dbg !119
// CHECK:STDOUT:   %.loc22_23.3.temp = alloca {}, align 1, !dbg !119
// CHECK:STDOUT:   %.loc22_22.1.temp = alloca {}, align 1, !dbg !120
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.1.temp), !dbg !119
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.3.temp), !dbg !119
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_22.1.temp), !dbg !120
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.loc22_22.1.temp, ptr %b), !dbg !120
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc22_23.3.temp, ptr %c, ptr %.loc22_22.1.temp), !dbg !119
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc22_23.3.temp), !dbg !121
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc22_23.3.temp), !dbg !119
// CHECK:STDOUT:   call void @"_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core"(ptr %.loc22_22.1.temp), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallGeneric.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallGeneric.Main.8a681e94b982e379(ptr %return, ptr %c, ptr %b), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %_) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc18_43, i64 0, i1 false), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallGeneric.Main.8a681e94b982e379(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk:I.eca2102529696f40.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %return, ptr %c, ptr %b), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF:thunk:I.eca2102529696f40.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) #2 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.2.temp = alloca {}, align 1, !dbg !159
// CHECK:STDOUT:   %.6.temp = alloca {}, align 1, !dbg !158
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.2.temp), !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !158
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.6.temp, ptr %x), !dbg !158
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc18_32.2.temp, ptr %self, ptr %.6.temp), !dbg !159
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc18_32.2.temp), !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc18_32.2.temp), !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core"(ptr %.6.temp), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!90}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !89 = !DIFile(filename: "generic_thunk.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !89, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !91 = !{!21, !21, !21}
// CHECK:STDOUT: !92 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !89, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !21)
// CHECK:STDOUT: !98 = !DILocation(line: 7, column: 36, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !89, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !21)
// CHECK:STDOUT: !102 = !DILocation(line: 18, column: 3, scope: !100)
// CHECK:STDOUT: !103 = !{!101}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !89, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !21)
// CHECK:STDOUT: !106 = !DILocation(line: 18, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !{!105}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core", scope: null, file: !89, type: !27, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !111)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !21)
// CHECK:STDOUT: !110 = !DILocation(line: 0, scope: !108)
// CHECK:STDOUT: !111 = !{!109}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !89, line: 21, type: !92, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !122)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !21)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 2, scope: !112, type: !21)
// CHECK:STDOUT: !119 = !DILocation(line: 22, column: 10, scope: !112)
// CHECK:STDOUT: !120 = !DILocation(line: 22, column: 22, scope: !112)
// CHECK:STDOUT: !121 = !DILocation(line: 22, column: 3, scope: !112)
// CHECK:STDOUT: !122 = !{!113, !114}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "CallCallGeneric", linkageName: "_CCallCallGeneric.Main", scope: null, file: !89, line: 29, type: !92, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !134)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !21)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 2, scope: !123, type: !21)
// CHECK:STDOUT: !132 = !DILocation(line: 30, column: 10, scope: !123)
// CHECK:STDOUT: !133 = !DILocation(line: 30, column: 3, scope: !123)
// CHECK:STDOUT: !134 = !{!124, !125}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !89, line: 18, type: !92, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !143)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !21)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 2, scope: !135, type: !21)
// CHECK:STDOUT: !142 = !DILocation(line: 18, column: 34, scope: !135)
// CHECK:STDOUT: !143 = !{!136, !137}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main.8a681e94b982e379", scope: null, file: !89, line: 25, type: !92, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !154)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !21)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 2, scope: !144, type: !21)
// CHECK:STDOUT: !151 = !DILocation(line: 26, column: 10, scope: !144)
// CHECK:STDOUT: !153 = !DILocation(line: 26, column: 3, scope: !144)
// CHECK:STDOUT: !154 = !{!145, !146}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.eca2102529696f40.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !89, line: 18, type: !92, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !160)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !21)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 2, scope: !155, type: !21)
// CHECK:STDOUT: !158 = !DILocation(line: 0, scope: !155)
// CHECK:STDOUT: !159 = !DILocation(line: 18, column: 3, scope: !155)
// CHECK:STDOUT: !160 = !{!156, !157}
// CHECK:STDOUT:
