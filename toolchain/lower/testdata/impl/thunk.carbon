// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/thunk.carbon

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

// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.Core"(ptr sret({ i32 }) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_49.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc16_49.2 = load i32, ptr %.loc16_49.1.a, align 4, !dbg !7
// CHECK:STDOUT:   %.loc16_51.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_49.2, ptr %.loc16_51.2.b, align 4, !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({ i32 }) %return, ptr %self) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_49.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc19_49.2 = load i32, ptr %.loc19_49.1.b, align 4, !dbg !11
// CHECK:STDOUT:   %.loc19_51.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   store i32 %.loc19_49.2, ptr %.loc19_51.2.c, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc27_35.2 = load i32, ptr %.loc27_35.1.b, align 4, !dbg !15
// CHECK:STDOUT:   %.loc27_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store i32 %.loc27_35.2, ptr %.loc27_37.2.b, align 4, !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_19.1.temp = alloca { i32 }, align 8, !dbg !19
// CHECK:STDOUT:   %.loc23_9.1.temp = alloca { i32 }, align 8, !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.loc23_9.1.temp, ptr %a), !dbg !20
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc27_19.1.temp, ptr %.loc23_9.1.temp), !dbg !19
// CHECK:STDOUT:   %.loc27_19.2.temp = alloca { i32 }, align 8, !dbg !19
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc27_19.2.temp, ptr %.loc27_19.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr %return, ptr %a), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.Core", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 45, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 39, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 32, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.Core", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 45, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 19, column: 39, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 32, scope: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 27, column: 34, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 27, column: 28, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 27, column: 21, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 27, column: 3, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 8, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 31, column: 10, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 31, column: 3, scope: !21)
