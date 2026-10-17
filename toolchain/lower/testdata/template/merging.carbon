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
  // TODO: It'd be nice to merge these, because they both resolve to calls that
  // call the same constant callee. But they're splicing different instructions,
  // so that will take some work.
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
// CHECK:STDOUT: define void @_CCallCallF.Main() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.bd4fb75a414fe858(), !dbg !9
// CHECK:STDOUT:   call void @_CCallF.Main.a6a60dfee2d7d273(), !dbg !10
// CHECK:STDOUT:   call void @_CCallF.Main.9522ff908e97e267(), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.bd4fb75a414fe858() #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.a6a60dfee2d7d273() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.9522ff908e97e267() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "merge_constant.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !4, line: 9, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !9 = !DILocation(line: 10, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 3, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 3, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 1, scope: !8)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.bd4fb75a414fe858", scope: null, file: !4, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 30, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 1, scope: !13)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.a6a60dfee2d7d273", scope: null, file: !4, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 30, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 1, scope: !16)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.9522ff908e97e267", scope: null, file: !4, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 30, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 7, column: 1, scope: !19)
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
// CHECK:STDOUT: define void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc8_45, i64 0, i1 false), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc8_45, i64 0, i1 false), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.bd4fb75a414fe858(), !dbg !44
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.a6a60dfee2d7d273(), !dbg !45
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.9522ff908e97e267(), !dbg !46
// CHECK:STDOUT:   call void @_CInitAndCallF.Main.931317eeae6ffe17(), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallBF.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_11.2.temp = alloca {}, align 1, !dbg !50
// CHECK:STDOUT:   %.loc33_11.2.temp = alloca {}, align 1, !dbg !52
// CHECK:STDOUT:   %.loc34_11.2.temp = alloca {}, align 1, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_11.2.temp), !dbg !50
// CHECK:STDOUT:   call void @_CCallBF.Main.a6a60dfee2d7d273(ptr @B.val.loc8_45), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr @B.val), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.2.temp), !dbg !52
// CHECK:STDOUT:   call void @_CCallBF.Main.9522ff908e97e267(ptr @C.val.loc33_13.3), !dbg !53
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr @C.val), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_11.2.temp), !dbg !54
// CHECK:STDOUT:   call void @_CCallBF.Main.931317eeae6ffe17(ptr @D.val.loc34_13.3), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4ef1144724f11c12(ptr @D.val), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.bd4fb75a414fe858() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr @A.val.A.val), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.a6a60dfee2d7d273() #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr @B.val.loc8_45), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.9522ff908e97e267() #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.4.temp = alloca {}, align 1, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.4.temp), !dbg !65
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.4.temp, ptr @C.val.loc33_13.3), !dbg !65
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.4.temp), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.931317eeae6ffe17() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.3.temp = alloca {}, align 1, !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.3.temp), !dbg !69
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.3.temp, ptr @D.val.loc34_13.3), !dbg !69
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.3.temp), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.a6a60dfee2d7d273(ptr %x) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %x), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %t) #0 !dbg !78 {
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.9522ff908e97e267(ptr %x) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.2.temp = alloca {}, align 1, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.2.temp), !dbg !88
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.2.temp, ptr %x), !dbg !88
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.2.temp), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr %t) #0 !dbg !91 {
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.931317eeae6ffe17(ptr %x) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.1.temp = alloca {}, align 1, !dbg !101
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.1.temp), !dbg !101
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.1.temp, ptr %x), !dbg !101
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.1.temp), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4ef1144724f11c12(ptr %t) #0 !dbg !104 {
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 6, 5, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !22 = !DIFile(filename: "no_merge_different_conversions.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !{!24, !24}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !22, line: 8, type: !28, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !35)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 36, scope: !29)
// CHECK:STDOUT: !35 = !{!30}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !22, line: 14, type: !28, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !24)
// CHECK:STDOUT: !41 = !DILocation(line: 14, column: 36, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !22, line: 20, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !44 = !DILocation(line: 21, column: 3, scope: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 22, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 23, column: 3, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 24, column: 3, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 1, scope: !43)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "CallCallBF", linkageName: "_CCallCallBF.Main", scope: null, file: !22, line: 31, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 32, column: 10, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 32, column: 3, scope: !49)
// CHECK:STDOUT: !52 = !DILocation(line: 33, column: 10, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 33, column: 3, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 10, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 34, column: 3, scope: !49)
// CHECK:STDOUT: !56 = !DILocation(line: 31, column: 1, scope: !49)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.bd4fb75a414fe858", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !58 = !DILocation(line: 18, column: 37, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 18, column: 1, scope: !57)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.a6a60dfee2d7d273", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 37, scope: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 18, column: 1, scope: !60)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.9522ff908e97e267", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 37, scope: !63)
// CHECK:STDOUT: !65 = !DILocation(line: 0, scope: !63)
// CHECK:STDOUT: !66 = !DILocation(line: 18, column: 1, scope: !63)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.931317eeae6ffe17", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 37, scope: !67)
// CHECK:STDOUT: !69 = !DILocation(line: 0, scope: !67)
// CHECK:STDOUT: !70 = !DILocation(line: 18, column: 1, scope: !67)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.a6a60dfee2d7d273", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !76)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !24)
// CHECK:STDOUT: !74 = !DILocation(line: 28, column: 3, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 27, column: 1, scope: !71)
// CHECK:STDOUT: !76 = !{!72}
// CHECK:STDOUT: !77 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b963e75af1c16acd", scope: null, file: !77, line: 33, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !84)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !24)
// CHECK:STDOUT: !83 = !DILocation(line: 33, column: 1, scope: !78)
// CHECK:STDOUT: !84 = !{!79}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.9522ff908e97e267", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !90)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !24)
// CHECK:STDOUT: !88 = !DILocation(line: 28, column: 3, scope: !85)
// CHECK:STDOUT: !89 = !DILocation(line: 27, column: 1, scope: !85)
// CHECK:STDOUT: !90 = !{!86}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.38e0a41f1f1b0274", scope: null, file: !77, line: 33, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !97)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !24)
// CHECK:STDOUT: !96 = !DILocation(line: 33, column: 1, scope: !91)
// CHECK:STDOUT: !97 = !{!92}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.931317eeae6ffe17", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !103)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !24)
// CHECK:STDOUT: !101 = !DILocation(line: 28, column: 3, scope: !98)
// CHECK:STDOUT: !102 = !DILocation(line: 27, column: 1, scope: !98)
// CHECK:STDOUT: !103 = !{!99}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4ef1144724f11c12", scope: null, file: !77, line: 33, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !24)
// CHECK:STDOUT: !109 = !DILocation(line: 33, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_merge_convert_one_arg.carbon'
// CHECK:STDOUT: source_filename = "no_merge_convert_one_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc6_45 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc6_45, i64 0, i1 false), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main(ptr %a, ptr %b) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.2877a42045e01a3b(ptr %a, ptr %a), !dbg !129
// CHECK:STDOUT:   call void @_CCallF.Main.d01354a0ad5f3687(ptr %a, ptr %b), !dbg !132
// CHECK:STDOUT:   call void @_CCallF.Main.37b67b83fe9ed3dd(ptr %b, ptr %a), !dbg !135
// CHECK:STDOUT:   call void @_CCallF.Main.7096323a0fb88cc3(ptr %b, ptr %b), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.2877a42045e01a3b(ptr %x, ptr %y) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %y), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.d01354a0ad5f3687(ptr %x, ptr %y) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.4.temp = alloca {}, align 1, !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.4.temp), !dbg !157
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.4.temp, ptr %y), !dbg !157
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %.9745c1.4.temp), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.37b67b83fe9ed3dd(ptr %x, ptr %y) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.3.temp = alloca {}, align 1, !dbg !166
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.3.temp), !dbg !166
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.3.temp, ptr %x), !dbg !166
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.3.temp, ptr %y), !dbg !168
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.7096323a0fb88cc3(ptr %x, ptr %y) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.1.temp = alloca {}, align 1, !dbg !176
// CHECK:STDOUT:   %.9745c1.2.temp = alloca {}, align 1, !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.1.temp), !dbg !176
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.1.temp, ptr %x), !dbg !176
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.2.temp), !dbg !177
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.2.temp, ptr %y), !dbg !177
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.1.temp, ptr %.9745c1.2.temp), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!112}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{!24, !24}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !111 = !DIFile(filename: "no_merge_convert_one_arg.carbon", directory: "")
// CHECK:STDOUT: !112 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !111, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !113 = !{null, !24, !24}
// CHECK:STDOUT: !114 = !DISubroutineType(types: !113)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !111, line: 6, type: !28, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !121)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !24)
// CHECK:STDOUT: !120 = !DILocation(line: 6, column: 36, scope: !115)
// CHECK:STDOUT: !121 = !{!116}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !111, line: 16, type: !114, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !140)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !24)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !122, type: !24)
// CHECK:STDOUT: !129 = !DILocation(line: 17, column: 3, scope: !122)
// CHECK:STDOUT: !132 = !DILocation(line: 18, column: 3, scope: !122)
// CHECK:STDOUT: !135 = !DILocation(line: 19, column: 3, scope: !122)
// CHECK:STDOUT: !138 = !DILocation(line: 20, column: 3, scope: !122)
// CHECK:STDOUT: !139 = !DILocation(line: 16, column: 1, scope: !122)
// CHECK:STDOUT: !140 = !{!123, !124}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.2877a42045e01a3b", scope: null, file: !111, line: 12, type: !114, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !150)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !24)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 2, scope: !141, type: !24)
// CHECK:STDOUT: !148 = !DILocation(line: 13, column: 3, scope: !141)
// CHECK:STDOUT: !149 = !DILocation(line: 12, column: 1, scope: !141)
// CHECK:STDOUT: !150 = !{!142, !143}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.d01354a0ad5f3687", scope: null, file: !111, line: 12, type: !114, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !160)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !24)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 2, scope: !151, type: !24)
// CHECK:STDOUT: !157 = !DILocation(line: 13, column: 8, scope: !151)
// CHECK:STDOUT: !158 = !DILocation(line: 13, column: 3, scope: !151)
// CHECK:STDOUT: !159 = !DILocation(line: 12, column: 1, scope: !151)
// CHECK:STDOUT: !160 = !{!152, !153}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.37b67b83fe9ed3dd", scope: null, file: !111, line: 12, type: !114, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !170)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !24)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 2, scope: !161, type: !24)
// CHECK:STDOUT: !166 = !DILocation(line: 13, column: 5, scope: !161)
// CHECK:STDOUT: !168 = !DILocation(line: 13, column: 3, scope: !161)
// CHECK:STDOUT: !169 = !DILocation(line: 12, column: 1, scope: !161)
// CHECK:STDOUT: !170 = !{!162, !163}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.7096323a0fb88cc3", scope: null, file: !111, line: 12, type: !114, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !180)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !24)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 2, scope: !171, type: !24)
// CHECK:STDOUT: !176 = !DILocation(line: 13, column: 5, scope: !171)
// CHECK:STDOUT: !177 = !DILocation(line: 13, column: 8, scope: !171)
// CHECK:STDOUT: !178 = !DILocation(line: 13, column: 3, scope: !171)
// CHECK:STDOUT: !179 = !DILocation(line: 12, column: 1, scope: !171)
// CHECK:STDOUT: !180 = !{!172, !173}
// CHECK:STDOUT:
