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
    fn Convert[unused self: Self]() -> A { return {}; }
  }
}

interface I(T:! type) {
  fn F[self: Self](x: B) -> A;
}

class C(U:! type) {}

impl forall [V:! type] C(V) as I(V) {
  fn F[unused self: Self](_: A) -> B { return {}; }
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.8b79c6c1932e58f6.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_49.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc9_49.2 = load i32, ptr %.loc9_49.1.a, align 4, !dbg !10
// CHECK:STDOUT:   %.loc9_51.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store i32 %.loc9_49.2, ptr %.loc9_51.2.b, align 4, !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.cc918b9f1e9f377a.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_49.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   %.loc12_49.2 = load i32, ptr %.loc12_49.1.b, align 4, !dbg !16
// CHECK:STDOUT:   %.loc12_51.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   store i32 %.loc12_49.2, ptr %.loc12_51.2.c, align 4, !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !22
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !22
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk:I.01f5c37cad34e90b.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 8, !dbg !28
// CHECK:STDOUT:   %.loc16_9.1.temp = alloca { i32 }, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_9.1.temp), !dbg !29
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.8b79c6c1932e58f6.Core"(ptr %.loc16_9.1.temp, ptr %a), !dbg !29
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.loc16_9.1.temp), !dbg !28
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.cc918b9f1e9f377a.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !28
// CHECK:STDOUT:   call void @"_COp.c989b489e17b3e13:core.Destroy.Core"(ptr %.loc20_19.1.temp), !dbg !28
// CHECK:STDOUT:   call void @"_COp.c989b489e17b3e13:core.Destroy.Core"(ptr %.loc16_9.1.temp), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a9ac47a486d88918:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c989b489e17b3e13:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_20.1.temp = alloca { i32 }, align 8, !dbg !50
// CHECK:STDOUT:   %.loc24_20.2.temp = alloca { i32 }, align 8, !dbg !50
// CHECK:STDOUT:   %.loc24_19.1.temp = alloca { i32 }, align 8, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.1.temp), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_20.2.temp), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_19.1.temp), !dbg !51
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.8b79c6c1932e58f6.Core"(ptr %.loc24_19.1.temp, ptr %a), !dbg !51
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc24_20.2.temp, ptr %.loc24_19.1.temp), !dbg !50
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.cc918b9f1e9f377a.Core"(ptr %return, ptr %.loc24_20.2.temp), !dbg !52
// CHECK:STDOUT:   call void @"_COp.c989b489e17b3e13:core.Destroy.Core"(ptr %.loc24_20.2.temp), !dbg !50
// CHECK:STDOUT:   call void @"_COp.c989b489e17b3e13:core.Destroy.Core"(ptr %.loc24_19.1.temp), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.c989b489e17b3e13:core.Destroy.Core", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.8b79c6c1932e58f6.Core", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 45, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 39, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 32, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.cc918b9f1e9f377a.Core", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 45, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 39, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 12, column: 32, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 34, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 20, column: 28, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 20, column: 21, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.01f5c37cad34e90b.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !28 = !DILocation(line: 20, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 8, scope: !25)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{null, !33}
// CHECK:STDOUT: !33 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !30, type: !33)
// CHECK:STDOUT: !36 = !DILocation(line: 20, column: 3, scope: !30)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a9ac47a486d88918:core.Destroy.Core", scope: null, file: !3, line: 20, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{null, !7}
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !37, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 3, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c989b489e17b3e13:core.Destroy.Core", scope: null, file: !3, line: 20, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !46 = !DILocation(line: 20, column: 3, scope: !43)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 10, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 24, column: 19, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 24, column: 3, scope: !47)
// CHECK:STDOUT: ; ModuleID = 'generic_thunk.carbon'
// CHECK:STDOUT: source_filename = "generic_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc7_53 = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc18_49 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.46f7e99dd8f31ccb.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc7_53, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ce54547ef4d499a:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c14becfc5db4251e:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.1.temp = alloca {}, align 8, !dbg !27
// CHECK:STDOUT:   %.loc22_23.3.temp = alloca {}, align 8, !dbg !27
// CHECK:STDOUT:   %.loc22_22.1.temp = alloca {}, align 8, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.1.temp), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.3.temp), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_22.1.temp), !dbg !28
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.46f7e99dd8f31ccb.Core"(ptr %.loc22_22.1.temp, ptr %b), !dbg !28
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc22_23.3.temp, ptr %c, ptr %.loc22_22.1.temp), !dbg !27
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.46f7e99dd8f31ccb.Core"(ptr %return, ptr %.loc22_23.3.temp), !dbg !29
// CHECK:STDOUT:   call void @"_COp.7ce54547ef4d499a:core.Destroy.Core"(ptr %.loc22_23.3.temp), !dbg !27
// CHECK:STDOUT:   call void @"_COp.c14becfc5db4251e:core.Destroy.Core"(ptr %.loc22_22.1.temp), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallGeneric.Main(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallGeneric.Main.18bae34346379f4e(ptr %return, ptr %c, ptr %b), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %_) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc18_49, i64 0, i1 false), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallGeneric.Main.18bae34346379f4e(ptr sret({}) %return, ptr %c, ptr %b) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF:thunk:I.a61d280f5ea4bfd8.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %return, ptr %c, ptr %b), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF:thunk:I.a61d280f5ea4bfd8.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr sret({}) %return, ptr %self, ptr %x) #2 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_38.2.temp = alloca {}, align 8, !dbg !51
// CHECK:STDOUT:   %.loc12_21.1.temp = alloca {}, align 8, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_38.2.temp), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_21.1.temp), !dbg !52
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.46f7e99dd8f31ccb.Core"(ptr %.loc12_21.1.temp, ptr %x), !dbg !52
// CHECK:STDOUT:   call void @"_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f"(ptr %.loc18_38.2.temp, ptr %self, ptr %.loc12_21.1.temp), !dbg !51
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.46f7e99dd8f31ccb.Core"(ptr %return, ptr %.loc18_38.2.temp), !dbg !51
// CHECK:STDOUT:   call void @"_COp.7ce54547ef4d499a:core.Destroy.Core"(ptr %.loc18_38.2.temp), !dbg !51
// CHECK:STDOUT:   call void @"_COp.c14becfc5db4251e:core.Destroy.Core"(ptr %.loc12_21.1.temp), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.46f7e99dd8f31ccb.Core", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 44, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ce54547ef4d499a:core.Destroy.Core", scope: null, file: !3, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !7}
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c14becfc5db4251e:core.Destroy.Core", scope: null, file: !3, line: 12, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 12, column: 20, scope: !17)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !3, line: 21, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{!7, !7, !7}
// CHECK:STDOUT: !24 = !{!25, !26}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !21, type: !7)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !21, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 22, column: 10, scope: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 22, scope: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 22, column: 3, scope: !21)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "CallCallGeneric", linkageName: "_CCallCallGeneric.Main", scope: null, file: !3, line: 29, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !7)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !30, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 30, column: 10, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 30, column: 3, scope: !30)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !3, line: 18, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !7)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !36, type: !7)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 40, scope: !36)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main.18bae34346379f4e", scope: null, file: !3, line: 25, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43, !44}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !7)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !41, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 10, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 3, scope: !41)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.a61d280f5ea4bfd8.Main:C.eb057aa32837c84e.Main:I.eb057aa32837c84e.Main.e43630e9a6c38c3f", scope: null, file: !3, line: 18, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49, !50}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !47, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 18, column: 3, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 12, column: 20, scope: !47)
