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
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc9_47.2 = load i32, ptr %.loc9_47.1.a, align 4, !dbg !31
// CHECK:STDOUT:   %.loc9_49.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   store i32 %.loc9_47.2, ptr %.loc9_49.2.b, align 4, !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_47.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc12_47.2 = load i32, ptr %.loc12_47.1.b, align 4, !dbg !40
// CHECK:STDOUT:   %.loc12_49.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   store i32 %.loc12_47.2, ptr %.loc12_49.2.c, align 4, !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !49
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 4, !dbg !56
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !55
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !55
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.4.temp), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.4.temp), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4cdc9d2e531bf474:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc20_19.1.b), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc20_19.1.b), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_20.1.temp = alloca { i32 }, align 4, !dbg !74
// CHECK:STDOUT:   %.loc24_20.2.temp = alloca { i32 }, align 4, !dbg !74
// CHECK:STDOUT:   %.loc24_19.1.temp = alloca { i32 }, align 4, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.1.temp), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.2.temp), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_19.1.temp), !dbg !73
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc24_19.1.temp, ptr %a), !dbg !73
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc24_20.2.temp, ptr %.loc24_19.1.temp), !dbg !74
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc24_20.2.temp), !dbg !75
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc24_20.2.temp), !dbg !74
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc24_19.1.temp), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #0 !dbg !77 {
// CHECK:STDOUT:   call void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %t), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !84 {
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c8fa36f2e394941a, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 1, 0 }
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
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !34)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 43, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 37, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 9, column: 30, scope: !26)
// CHECK:STDOUT: !34 = !{!27}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !43)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !21)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 43, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 37, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 30, scope: !35)
// CHECK:STDOUT: !43 = !{!36}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !21)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 34, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 20, column: 28, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 21, scope: !44)
// CHECK:STDOUT: !52 = !{!45}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !55 = !DILocation(line: 0, scope: !53)
// CHECK:STDOUT: !56 = !DILocation(line: 20, column: 3, scope: !53)
// CHECK:STDOUT: !57 = !{!54}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4cdc9d2e531bf474:core.SubobjectDestroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !60 = !DILocation(line: 20, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !{null, !20}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !{!59}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !18, line: 23, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !76)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !73 = !DILocation(line: 24, column: 19, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 24, column: 10, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 24, column: 3, scope: !68)
// CHECK:STDOUT: !76 = !{!69}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !21)
// CHECK:STDOUT: !81 = !DILocation(line: 28, column: 5, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !62, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 26, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_thunk.carbon'
// CHECK:STDOUT: source_filename = "generic_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc7_45 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc18_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc7_45, i64 0, i1 false), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.1.temp = alloca {}, align 1, !dbg !109
// CHECK:STDOUT:   %.loc22_23.3.temp = alloca {}, align 1, !dbg !109
// CHECK:STDOUT:   %.loc22_22.1.temp = alloca {}, align 1, !dbg !110
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.1.temp), !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.3.temp), !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_22.1.temp), !dbg !110
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.loc22_22.1.temp, ptr %b), !dbg !110
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc22_23.3.temp, ptr %c, ptr %.loc22_22.1.temp), !dbg !109
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc22_23.3.temp), !dbg !111
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc22_23.3.temp), !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %.loc22_22.1.temp), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallGeneric.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallGeneric.Main.5ebe45018617384b(ptr %return, ptr %c, ptr %b), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %_) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc18_43, i64 0, i1 false), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #0 !dbg !134 {
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %t) #0 !dbg !141 {
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallGeneric.Main.5ebe45018617384b(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk:I.31c1026e1454ba5e.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %return, ptr %c, ptr %b), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF:thunk:I.31c1026e1454ba5e.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) #2 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.2.temp = alloca {}, align 1, !dbg !163
// CHECK:STDOUT:   %.6.temp = alloca {}, align 1, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.2.temp), !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !162
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.6.temp, ptr %x), !dbg !162
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc18_32.2.temp, ptr %self, ptr %.6.temp), !dbg !163
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %return, ptr %.loc18_32.2.temp), !dbg !163
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc18_32.2.temp), !dbg !163
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %.6.temp), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !91 = !DIFile(filename: "generic_thunk.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = !{!21, !21, !21}
// CHECK:STDOUT: !94 = !DISubroutineType(types: !93)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !91, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !21)
// CHECK:STDOUT: !100 = !DILocation(line: 7, column: 36, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !91, line: 21, type: !94, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !112)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !21)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !102, type: !21)
// CHECK:STDOUT: !109 = !DILocation(line: 22, column: 10, scope: !102)
// CHECK:STDOUT: !110 = !DILocation(line: 22, column: 22, scope: !102)
// CHECK:STDOUT: !111 = !DILocation(line: 22, column: 3, scope: !102)
// CHECK:STDOUT: !112 = !{!103, !104}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "CallCallGeneric", linkageName: "_CCallCallGeneric.Main", scope: null, file: !91, line: 29, type: !94, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !124)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !21)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !113, type: !21)
// CHECK:STDOUT: !122 = !DILocation(line: 30, column: 10, scope: !113)
// CHECK:STDOUT: !123 = !DILocation(line: 30, column: 3, scope: !113)
// CHECK:STDOUT: !124 = !{!114, !115}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !91, line: 18, type: !94, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !133)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !21)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 2, scope: !125, type: !21)
// CHECK:STDOUT: !132 = !DILocation(line: 18, column: 34, scope: !125)
// CHECK:STDOUT: !133 = !{!126, !127}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !21)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 1, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1ff1de8b251403c3", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !147)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !21)
// CHECK:STDOUT: !146 = !DILocation(line: 26, column: 1, scope: !141)
// CHECK:STDOUT: !147 = !{!142}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main.5ebe45018617384b", scope: null, file: !91, line: 25, type: !94, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !158)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !21)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 2, scope: !148, type: !21)
// CHECK:STDOUT: !155 = !DILocation(line: 26, column: 10, scope: !148)
// CHECK:STDOUT: !157 = !DILocation(line: 26, column: 3, scope: !148)
// CHECK:STDOUT: !158 = !{!149, !150}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.31c1026e1454ba5e.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !91, line: 18, type: !94, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !164)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !21)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 2, scope: !159, type: !21)
// CHECK:STDOUT: !162 = !DILocation(line: 0, scope: !159)
// CHECK:STDOUT: !163 = !DILocation(line: 18, column: 3, scope: !159)
// CHECK:STDOUT: !164 = !{!160, !161}
// CHECK:STDOUT:
