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
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc9_47.2 = load i32, ptr %.loc9_47.1.a, align 4, !dbg !29
// CHECK:STDOUT:   %.loc9_49.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   store i32 %.loc9_47.2, ptr %.loc9_49.2.b, align 4, !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_47.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   %.loc12_47.2 = load i32, ptr %.loc12_47.1.b, align 4, !dbg !38
// CHECK:STDOUT:   %.loc12_49.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   store i32 %.loc12_47.2, ptr %.loc12_49.2.c, align 4, !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !47
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 4, !dbg !54
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !53
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !53
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.4.temp), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc20_19.1.temp), !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.4.temp), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_20.1.temp = alloca { i32 }, align 4, !dbg !64
// CHECK:STDOUT:   %.loc24_20.2.temp = alloca { i32 }, align 4, !dbg !64
// CHECK:STDOUT:   %.loc24_19.1.temp = alloca { i32 }, align 4, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.1.temp), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.2.temp), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_19.1.temp), !dbg !63
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc24_19.1.temp, ptr %a), !dbg !63
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc24_20.2.temp, ptr %.loc24_19.1.temp), !dbg !64
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc24_20.2.temp), !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc24_20.2.temp), !dbg !64
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc24_19.1.temp), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %t) #0 !dbg !67 {
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.b963e75af1c16acd, { 3, 2, 1, 0 }
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !32)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 9, column: 43, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 9, column: 37, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 30, scope: !24)
// CHECK:STDOUT: !32 = !{!25}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !21)
// CHECK:STDOUT: !38 = !DILocation(line: 12, column: 43, scope: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 37, scope: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 30, scope: !33)
// CHECK:STDOUT: !41 = !{!34}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 20, column: 34, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 28, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 21, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !21)
// CHECK:STDOUT: !53 = !DILocation(line: 0, scope: !51)
// CHECK:STDOUT: !54 = !DILocation(line: 20, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !{null, !21}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !18, line: 23, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !66)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !63 = !DILocation(line: 24, column: 19, scope: !58)
// CHECK:STDOUT: !64 = !DILocation(line: 24, column: 10, scope: !58)
// CHECK:STDOUT: !65 = !DILocation(line: 24, column: 3, scope: !58)
// CHECK:STDOUT: !66 = !{!59}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b963e75af1c16acd", scope: null, file: !10, line: 30, type: !56, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 30, column: 1, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_thunk.carbon'
// CHECK:STDOUT: source_filename = "generic_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc7_45 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc18_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc7_45, i64 0, i1 false), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.1.temp = alloca {}, align 1, !dbg !92
// CHECK:STDOUT:   %.loc22_23.3.temp = alloca {}, align 1, !dbg !92
// CHECK:STDOUT:   %.loc22_22.1.temp = alloca {}, align 1, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.1.temp), !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.3.temp), !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_22.1.temp), !dbg !93
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.loc22_22.1.temp, ptr %b), !dbg !93
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc22_23.3.temp, ptr %c, ptr %.loc22_22.1.temp), !dbg !92
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc22_23.3.temp), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc22_23.3.temp), !dbg !92
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %.loc22_22.1.temp), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallGeneric.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallGeneric.Main.e6730b2ec192c142(ptr %return, ptr %c, ptr %b), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %_) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc18_43, i64 0, i1 false), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %t) #0 !dbg !117 {
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %t) #0 !dbg !124 {
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallGeneric.Main.e6730b2ec192c142(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk:I.e2e8ace3e0ca3527.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %return, ptr %c, ptr %b), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF:thunk:I.e2e8ace3e0ca3527.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) #2 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.2.temp = alloca {}, align 1, !dbg !146
// CHECK:STDOUT:   %.6.temp = alloca {}, align 1, !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.2.temp), !dbg !146
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !145
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.6.temp, ptr %x), !dbg !145
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc18_32.2.temp, ptr %self, ptr %.6.temp), !dbg !146
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc18_32.2.temp), !dbg !146
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc18_32.2.temp), !dbg !146
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %.6.temp), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!75}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !55 = !{null, !21}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !74 = !DIFile(filename: "generic_thunk.carbon", directory: "")
// CHECK:STDOUT: !75 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !74, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !76 = !{!21, !21, !21}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !74, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !84)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !21)
// CHECK:STDOUT: !83 = !DILocation(line: 7, column: 36, scope: !78)
// CHECK:STDOUT: !84 = !{!79}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !74, line: 21, type: !77, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !95)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !21)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !85, type: !21)
// CHECK:STDOUT: !92 = !DILocation(line: 22, column: 10, scope: !85)
// CHECK:STDOUT: !93 = !DILocation(line: 22, column: 22, scope: !85)
// CHECK:STDOUT: !94 = !DILocation(line: 22, column: 3, scope: !85)
// CHECK:STDOUT: !95 = !{!86, !87}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "CallCallGeneric", linkageName: "_CCallCallGeneric.Main", scope: null, file: !74, line: 29, type: !77, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !107)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !21)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !96, type: !21)
// CHECK:STDOUT: !105 = !DILocation(line: 30, column: 10, scope: !96)
// CHECK:STDOUT: !106 = !DILocation(line: 30, column: 3, scope: !96)
// CHECK:STDOUT: !107 = !{!97, !98}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !74, line: 18, type: !77, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !116)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !21)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !108, type: !21)
// CHECK:STDOUT: !115 = !DILocation(line: 18, column: 34, scope: !108)
// CHECK:STDOUT: !116 = !{!109, !110}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b963e75af1c16acd", scope: null, file: !10, line: 30, type: !56, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !123)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !21)
// CHECK:STDOUT: !122 = !DILocation(line: 30, column: 1, scope: !117)
// CHECK:STDOUT: !123 = !{!118}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e517637b3b58cb2c", scope: null, file: !10, line: 30, type: !56, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !130)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !21)
// CHECK:STDOUT: !129 = !DILocation(line: 30, column: 1, scope: !124)
// CHECK:STDOUT: !130 = !{!125}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main.e6730b2ec192c142", scope: null, file: !74, line: 25, type: !77, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !141)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !21)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 2, scope: !131, type: !21)
// CHECK:STDOUT: !138 = !DILocation(line: 26, column: 10, scope: !131)
// CHECK:STDOUT: !140 = !DILocation(line: 26, column: 3, scope: !131)
// CHECK:STDOUT: !141 = !{!132, !133}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.e2e8ace3e0ca3527.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !74, line: 18, type: !77, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !147)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !21)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 2, scope: !142, type: !21)
// CHECK:STDOUT: !145 = !DILocation(line: 0, scope: !142)
// CHECK:STDOUT: !146 = !DILocation(line: 18, column: 3, scope: !142)
// CHECK:STDOUT: !147 = !{!143, !144}
// CHECK:STDOUT:
