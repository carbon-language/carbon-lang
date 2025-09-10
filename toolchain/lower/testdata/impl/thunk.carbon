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
  fn Convert[self: A]() -> B { return {.b = self.a}; }
}
impl B as Core.ImplicitAs(C) {
  fn Convert[self: B]() -> C { return {.c = self.b}; }
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
    fn Convert[self: Self]() -> A { return {}; }
  }
}

interface I(T:! type) {
  fn F[self: Self](x: B) -> A;
}

class C(U:! type) {}

impl forall [V:! type] C(V) as I(V) {
  fn F[self: Self](x: A) -> B { return {}; }
}

fn Call(c: C(()), b: B) -> A {
  return c.(I(()).F)(b);
}

fn CallGeneric[T:! I(())](c: T, b: B) -> A {
  return c.F(b);
}

fn CallCallGeneric(c: C(()), b: B) -> A {
  return CallGeneric(c, b);
}

// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.Core"(ptr sret({ i32 }) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_49.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc9_49.2 = load i32, ptr %.loc9_49.1.a, align 4, !dbg !7
// CHECK:STDOUT:   %.loc9_51.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc9_49.2, ptr %.loc9_51.2.b, align 4, !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({ i32 }) %return, ptr %self) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_49.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc12_49.2 = load i32, ptr %.loc12_49.1.b, align 4, !dbg !11
// CHECK:STDOUT:   %.loc12_51.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   store i32 %.loc12_49.2, ptr %.loc12_51.2.c, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !15
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 8, !dbg !19
// CHECK:STDOUT:   %.loc16_9.1.temp = alloca { i32 }, align 8, !dbg !20
// CHECK:STDOUT:   %.loc20_19.2.temp = alloca { i32 }, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_9.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.loc16_9.1.temp, ptr %a), !dbg !20
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.loc16_9.1.temp), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.2.temp), !dbg !19
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc20_19.2.temp, ptr %.loc20_19.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_20.1.temp = alloca { i32 }, align 8, !dbg !22
// CHECK:STDOUT:   %.loc24_20.2.temp = alloca { i32 }, align 8, !dbg !22
// CHECK:STDOUT:   %.loc24_19.1.temp = alloca { i32 }, align 8, !dbg !23
// CHECK:STDOUT:   %.loc24_21.1.temp = alloca { i32 }, align 8, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.1.temp), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.2.temp), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_19.1.temp), !dbg !23
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.loc24_19.1.temp, ptr %a), !dbg !23
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc24_20.2.temp, ptr %.loc24_19.1.temp), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_21.1.temp), !dbg !24
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc24_21.1.temp, ptr %.loc24_20.2.temp), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.Core", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 45, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 39, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 32, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.Core", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 45, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 39, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 32, scope: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 20, column: 34, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 28, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 20, column: 21, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 20, column: 3, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 16, column: 8, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 10, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 24, column: 19, scope: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 24, column: 3, scope: !21)
// CHECK:STDOUT: ; ModuleID = 'generic_thunk.carbon'
// CHECK:STDOUT: source_filename = "generic_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc7_46 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc18_42 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({}) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc7_46, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %c, ptr %b) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.1.temp = alloca {}, align 8, !dbg !9
// CHECK:STDOUT:   %.loc22_23.3.temp = alloca {}, align 8, !dbg !9
// CHECK:STDOUT:   %.loc22_23.7.temp = alloca {}, align 8, !dbg !9
// CHECK:STDOUT:   %.loc22_22.1.temp = alloca {}, align 8, !dbg !10
// CHECK:STDOUT:   %.loc22_24.1.temp = alloca {}, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.1.temp), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.3.temp), !dbg !9
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc22_23.3.temp, ptr %b), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.7.temp), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_22.1.temp), !dbg !10
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc22_22.1.temp, ptr %b), !dbg !10
// CHECK:STDOUT:   call void @"_CF.C.Main:I.Main.e43630e9a6c38c3f"(ptr %.loc22_23.7.temp, ptr %c, ptr %.loc22_22.1.temp), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_24.1.temp), !dbg !11
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc22_24.1.temp, ptr %.loc22_23.7.temp), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCallCallGeneric.Main(ptr sret({}) %return, ptr %c, ptr %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallGeneric.Main.0bb47992fefb9e3b(ptr %return, ptr %c, ptr %b), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.Main:I.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc18_42, i64 0, i1 false), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CCallGeneric.Main.0bb47992fefb9e3b(ptr sret({}) %return, ptr %c, ptr %b) !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk.C.Main:I.Main.e43630e9a6c38c3f"(ptr %return, ptr %c, ptr %b), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CF:thunk.C.Main:I.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_31.2.temp = alloca {}, align 8, !dbg !21
// CHECK:STDOUT:   %.loc18_31.6.temp = alloca {}, align 8, !dbg !21
// CHECK:STDOUT:   %.loc12_21.1.temp = alloca {}, align 8, !dbg !22
// CHECK:STDOUT:   %.loc18_31.7.temp = alloca {}, align 8, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_31.2.temp), !dbg !21
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc18_31.2.temp, ptr %x), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_31.6.temp), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_21.1.temp), !dbg !22
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc12_21.1.temp, ptr %x), !dbg !22
// CHECK:STDOUT:   call void @"_CF.C.Main:I.Main.e43630e9a6c38c3f"(ptr %.loc18_31.6.temp, ptr %self, ptr %.loc12_21.1.temp), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_31.7.temp), !dbg !21
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc18_31.7.temp, ptr %.loc18_31.6.temp), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 8, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.Core", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 37, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 22, column: 10, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 22, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 3, scope: !8)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "CallCallGeneric", linkageName: "_CCallCallGeneric.Main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 30, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 30, column: 3, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.Main.e43630e9a6c38c3f", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 33, scope: !15)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main.0bb47992fefb9e3b", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !18 = !DILocation(line: 26, column: 10, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 26, column: 3, scope: !17)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.C.Main:I.Main.e43630e9a6c38c3f", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 3, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 12, column: 20, scope: !20)
