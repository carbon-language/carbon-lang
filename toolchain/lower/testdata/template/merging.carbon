// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/merging.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/merging.carbon

// --- merge_constant.carbon
library "[[@TEST_NAME]]";

base class A { fn F(); }
base class B { fn F(); }
class C { extend base: B; }

fn CallF(template T: type) { T.F(); }

fn CallCallF() {
  CallF(A);
  // These are merged because their splices evaluate to the same constant value.
  CallF(B);
  CallF(C);
}

// --- no_merge_different_conversions.carbon
library "[[@TEST_NAME]]";

class A { fn F(self); }
class B { fn F(self); }
class C {
  alias F = B.F;
  impl as Core.ImplicitAs(B) {
    fn Convert(unused self) -> B { return {}; }
  }
}
class D {
  alias F = B.F;
  impl as Core.ImplicitAs(B) {
    fn Convert(unused self) -> B { return {}; }
  }
}

fn InitAndCallF(template T: type) { ({} as T).F(); }

fn CallCallF() {
  InitAndCallF(A);
  InitAndCallF(B);
  InitAndCallF(C);
  InitAndCallF(D);
}

fn CallBF[template T: type](x: T) {
  x.(B.F)();
}

fn CallCallBF() {
  CallBF({} as B);
  CallBF({} as C);
  CallBF({} as D);
}

// --- no_merge_convert_one_arg.carbon
library "[[@TEST_NAME]]";

class A {}
class B {
  impl as Core.ImplicitAs(A) {
    fn Convert(unused self) -> A { return {}; }
  }
}

fn F(x: A, y: A);

fn CallF[template T: type, template U: type](x: T, y: U) {
  F(x, y);
}

fn CallCallF(a: A, b: B) {
  CallF(a, a);
  CallF(a, b);
  CallF(b, a);
  CallF(b, b);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'merge_constant.carbon'
// CHECK:STDOUT: source_filename = "merge_constant.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.A.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.B.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.bd4fb75a414fe858(), !dbg !7
// CHECK:STDOUT:   call void @_CCallF.Main.a6a60dfee2d7d273(), !dbg !8
// CHECK:STDOUT:   call void @_CCallF.Main.a6a60dfee2d7d273(), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.bd4fb75a414fe858() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.a6a60dfee2d7d273() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CCallF.Main.a6a60dfee2d7d273, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "merge_constant.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.bd4fb75a414fe858", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 30, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 1, scope: !11)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.a6a60dfee2d7d273", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 30, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 1, scope: !14)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_merge_different_conversions.carbon'
// CHECK:STDOUT: source_filename = "no_merge_different_conversions.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @B.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc8_45 = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc33_13.3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val.loc34_13.3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @A.val.A.val = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.A.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.B.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc8_45, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc8_45, i64 0, i1 false), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.bd4fb75a414fe858(), !dbg !18
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.a6a60dfee2d7d273(), !dbg !19
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.9522ff908e97e267(), !dbg !20
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.931317eeae6ffe17(), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallBF.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_11.2.temp = alloca {}, align 1, !dbg !24
// CHECK:STDOUT:   %.loc33_11.2.temp = alloca {}, align 1, !dbg !25
// CHECK:STDOUT:   %.loc34_11.2.temp = alloca {}, align 1, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_11.2.temp), !dbg !24
// CHECK:STDOUT:   call void @_CCallBF.Main.a6a60dfee2d7d273(ptr @B.val.loc8_45), !dbg !27
// CHECK:STDOUT:   call void @"_COp.535580a8a7a96b3b:core.Destroy.Core"(ptr @B.val), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.2.temp), !dbg !25
// CHECK:STDOUT:   call void @_CCallBF.Main.9522ff908e97e267(ptr @C.val.loc33_13.3), !dbg !28
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr @C.val), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_11.2.temp), !dbg !26
// CHECK:STDOUT:   call void @_CCallBF.Main.931317eeae6ffe17(ptr @D.val.loc34_13.3), !dbg !29
// CHECK:STDOUT:   call void @"_COp.86bfabb7802f32e5:core.Destroy.Core"(ptr @D.val), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.535580a8a7a96b3b:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.86bfabb7802f32e5:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.bd4fb75a414fe858() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr @A.val.A.val), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.a6a60dfee2d7d273() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr @B.val.loc8_45), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.9522ff908e97e267() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.4.temp = alloca {}, align 1, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.4.temp), !dbg !52
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.4.temp, ptr @C.val.loc33_13.3), !dbg !52
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.4.temp), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.931317eeae6ffe17() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.3.temp = alloca {}, align 1, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.3.temp), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.3.temp, ptr @D.val.loc34_13.3), !dbg !56
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.3.temp), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.a6a60dfee2d7d273(ptr %x) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %x), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.9522ff908e97e267(ptr %x) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.2.temp = alloca {}, align 1, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.2.temp), !dbg !67
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.2.temp, ptr %x), !dbg !67
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.2.temp), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.931317eeae6ffe17(ptr %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.1.temp = alloca {}, align 1, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.1.temp), !dbg !72
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.1.temp, ptr %x), !dbg !72
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.1.temp), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 6, 5, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "no_merge_different_conversions.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !1, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 8, column: 36, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 14, column: 36, scope: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !1, line: 20, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DILocation(line: 21, column: 3, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 3, scope: !15)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 3, scope: !15)
// CHECK:STDOUT: !21 = !DILocation(line: 24, column: 3, scope: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 1, scope: !15)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "CallCallBF", linkageName: "_CCallCallBF.Main", scope: null, file: !1, line: 31, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !24 = !DILocation(line: 32, column: 10, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 33, column: 10, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 34, column: 10, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 32, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 33, column: 3, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 34, column: 3, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 31, column: 1, scope: !23)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.535580a8a7a96b3b:core.Destroy.Core", scope: null, file: !1, line: 32, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{null, !7}
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !31, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 32, column: 10, scope: !31)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !1, line: 33, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !7)
// CHECK:STDOUT: !40 = !DILocation(line: 33, column: 10, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.86bfabb7802f32e5:core.Destroy.Core", scope: null, file: !1, line: 34, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 34, column: 10, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.bd4fb75a414fe858", scope: null, file: !1, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 37, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 1, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.a6a60dfee2d7d273", scope: null, file: !1, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 37, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 18, column: 1, scope: !48)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.9522ff908e97e267", scope: null, file: !1, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !52 = !DILocation(line: 0, scope: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 18, column: 37, scope: !51)
// CHECK:STDOUT: !54 = !DILocation(line: 18, column: 1, scope: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.931317eeae6ffe17", scope: null, file: !1, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !56 = !DILocation(line: 0, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 18, column: 37, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 18, column: 1, scope: !55)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.a6a60dfee2d7d273", scope: null, file: !1, line: 27, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !60)
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !59, type: !7)
// CHECK:STDOUT: !62 = !DILocation(line: 28, column: 3, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 27, column: 1, scope: !59)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.9522ff908e97e267", scope: null, file: !1, line: 27, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !7)
// CHECK:STDOUT: !67 = !DILocation(line: 28, column: 3, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 27, column: 1, scope: !64)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.931317eeae6ffe17", scope: null, file: !1, line: 27, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !7)
// CHECK:STDOUT: !72 = !DILocation(line: 28, column: 3, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 27, column: 1, scope: !69)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_merge_convert_one_arg.carbon'
// CHECK:STDOUT: source_filename = "no_merge_convert_one_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc6_45 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc6_45, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main(ptr %a, ptr %b) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.2877a42045e01a3b(ptr %a, ptr %a), !dbg !17
// CHECK:STDOUT:   call void @_CCallF.Main.d01354a0ad5f3687(ptr %a, ptr %b), !dbg !18
// CHECK:STDOUT:   call void @_CCallF.Main.d01354a0ad5f3687(ptr %b, ptr %a), !dbg !19
// CHECK:STDOUT:   call void @_CCallF.Main.7096323a0fb88cc3(ptr %b, ptr %b), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.2877a42045e01a3b(ptr %x, ptr %y) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %y), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.d01354a0ad5f3687(ptr %x, ptr %y) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.4.temp = alloca {}, align 1, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.4.temp), !dbg !32
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.4.temp, ptr %y), !dbg !32
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %.9745c1.4.temp), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.7096323a0fb88cc3(ptr %x, ptr %y) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.1.temp = alloca {}, align 1, !dbg !38
// CHECK:STDOUT:   %.9745c1.2.temp = alloca {}, align 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.1.temp), !dbg !38
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.1.temp, ptr %x), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.2.temp), !dbg !38
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.2.temp, ptr %y), !dbg !38
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.1.temp, ptr %.9745c1.2.temp), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CCallF.Main.d01354a0ad5f3687, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "no_merge_convert_one_arg.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !1, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 36, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !1, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !7, !7}
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !11, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 20, column: 3, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 16, column: 1, scope: !11)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.2877a42045e01a3b", scope: null, file: !1, line: 12, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !7)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !22, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 12, column: 1, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.d01354a0ad5f3687", scope: null, file: !1, line: 12, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30, !31}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 2, scope: !28, type: !7)
// CHECK:STDOUT: !32 = !DILocation(line: 13, column: 3, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 1, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.7096323a0fb88cc3", scope: null, file: !1, line: 12, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36, !37}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 2, scope: !34, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 13, column: 3, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 1, scope: !34)
// CHECK:STDOUT:
