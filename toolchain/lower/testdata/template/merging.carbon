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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr @B.val), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.2.temp), !dbg !52
// CHECK:STDOUT:   call void @_CCallBF.Main.9522ff908e97e267(ptr @C.val.loc33_13.3), !dbg !53
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_11.2.temp), !dbg !54
// CHECK:STDOUT:   call void @_CCallBF.Main.931317eeae6ffe17(ptr @D.val.loc34_13.3), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f8f025949cf12143(ptr @D.val), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a50fd899f969e609:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.bd4fb75a414fe858() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr @A.val.A.val), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.a6a60dfee2d7d273() #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr @B.val.loc8_45), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.9522ff908e97e267() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.4.temp = alloca {}, align 1, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.4.temp), !dbg !77
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.4.temp, ptr @C.val.loc33_13.3), !dbg !77
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.4.temp), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.931317eeae6ffe17() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.3.temp = alloca {}, align 1, !dbg !81
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.3.temp), !dbg !81
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.3.temp, ptr @D.val.loc34_13.3), !dbg !81
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.3.temp), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.a6a60dfee2d7d273(ptr %x) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %x), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.9522ff908e97e267(ptr %x) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.2.temp = alloca {}, align 1, !dbg !100
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.2.temp), !dbg !100
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.2.temp, ptr %x), !dbg !100
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.2.temp), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.931317eeae6ffe17(ptr %x) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.1.temp = alloca {}, align 1, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.1.temp), !dbg !113
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.1.temp, ptr %x), !dbg !113
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.1.temp), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f8f025949cf12143(ptr %self) #0 !dbg !116 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a50fd899f969e609:core.Destroy.Core"(ptr %self), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
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
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !22, line: 32, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !24)
// CHECK:STDOUT: !59 = !DILocation(line: 32, column: 10, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !22, line: 33, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !24)
// CHECK:STDOUT: !63 = !DILocation(line: 33, column: 10, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a50fd899f969e609:core.Destroy.Core", scope: null, file: !22, line: 34, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !24)
// CHECK:STDOUT: !67 = !DILocation(line: 34, column: 10, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.bd4fb75a414fe858", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !70 = !DILocation(line: 18, column: 37, scope: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 18, column: 1, scope: !69)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.a6a60dfee2d7d273", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !73 = !DILocation(line: 18, column: 37, scope: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 18, column: 1, scope: !72)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.9522ff908e97e267", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !76 = !DILocation(line: 18, column: 37, scope: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 0, scope: !75)
// CHECK:STDOUT: !78 = !DILocation(line: 18, column: 1, scope: !75)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.931317eeae6ffe17", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !80 = !DILocation(line: 18, column: 37, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 0, scope: !79)
// CHECK:STDOUT: !82 = !DILocation(line: 18, column: 1, scope: !79)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.a6a60dfee2d7d273", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !88)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !24)
// CHECK:STDOUT: !86 = !DILocation(line: 28, column: 3, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 27, column: 1, scope: !83)
// CHECK:STDOUT: !88 = !{!84}
// CHECK:STDOUT: !89 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !89, line: 26, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !96)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !24)
// CHECK:STDOUT: !94 = !DILocation(line: 28, column: 5, scope: !90)
// CHECK:STDOUT: !95 = !DILocation(line: 26, column: 3, scope: !90)
// CHECK:STDOUT: !96 = !{!91}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.9522ff908e97e267", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !102)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !24)
// CHECK:STDOUT: !100 = !DILocation(line: 28, column: 3, scope: !97)
// CHECK:STDOUT: !101 = !DILocation(line: 27, column: 1, scope: !97)
// CHECK:STDOUT: !102 = !{!98}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !89, line: 26, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !24)
// CHECK:STDOUT: !107 = !DILocation(line: 28, column: 5, scope: !103)
// CHECK:STDOUT: !108 = !DILocation(line: 26, column: 3, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.931317eeae6ffe17", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !115)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !24)
// CHECK:STDOUT: !113 = !DILocation(line: 28, column: 3, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 27, column: 1, scope: !110)
// CHECK:STDOUT: !115 = !{!111}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f8f025949cf12143", scope: null, file: !89, line: 26, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !24)
// CHECK:STDOUT: !120 = !DILocation(line: 28, column: 5, scope: !116)
// CHECK:STDOUT: !121 = !DILocation(line: 26, column: 3, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_merge_convert_one_arg.carbon'
// CHECK:STDOUT: source_filename = "no_merge_convert_one_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc6_45 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc6_45, i64 0, i1 false), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main(ptr %a, ptr %b) #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.2877a42045e01a3b(ptr %a, ptr %a), !dbg !141
// CHECK:STDOUT:   call void @_CCallF.Main.d01354a0ad5f3687(ptr %a, ptr %b), !dbg !144
// CHECK:STDOUT:   call void @_CCallF.Main.37b67b83fe9ed3dd(ptr %b, ptr %a), !dbg !147
// CHECK:STDOUT:   call void @_CCallF.Main.7096323a0fb88cc3(ptr %b, ptr %b), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.2877a42045e01a3b(ptr %x, ptr %y) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %y), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.d01354a0ad5f3687(ptr %x, ptr %y) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.4.temp = alloca {}, align 1, !dbg !170
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.4.temp), !dbg !170
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.4.temp, ptr %y), !dbg !170
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %.9745c1.4.temp), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.37b67b83fe9ed3dd(ptr %x, ptr %y) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.3.temp = alloca {}, align 1, !dbg !180
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.3.temp), !dbg !180
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.3.temp, ptr %x), !dbg !180
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.3.temp, ptr %y), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.7096323a0fb88cc3(ptr %x, ptr %y) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.1.temp = alloca {}, align 1, !dbg !190
// CHECK:STDOUT:   %.9745c1.2.temp = alloca {}, align 1, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.1.temp), !dbg !190
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.1.temp, ptr %x), !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.2.temp), !dbg !190
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.2.temp, ptr %y), !dbg !190
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.1.temp, ptr %.9745c1.2.temp), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !191
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!124}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{!24, !24}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !123 = !DIFile(filename: "no_merge_convert_one_arg.carbon", directory: "")
// CHECK:STDOUT: !124 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !123, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !125 = !{null, !24, !24}
// CHECK:STDOUT: !126 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !123, line: 6, type: !28, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !24)
// CHECK:STDOUT: !132 = !DILocation(line: 6, column: 36, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !152)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !24)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 2, scope: !134, type: !24)
// CHECK:STDOUT: !141 = !DILocation(line: 17, column: 3, scope: !134)
// CHECK:STDOUT: !144 = !DILocation(line: 18, column: 3, scope: !134)
// CHECK:STDOUT: !147 = !DILocation(line: 19, column: 3, scope: !134)
// CHECK:STDOUT: !150 = !DILocation(line: 20, column: 3, scope: !134)
// CHECK:STDOUT: !151 = !DILocation(line: 16, column: 1, scope: !134)
// CHECK:STDOUT: !152 = !{!135, !136}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.2877a42045e01a3b", scope: null, file: !123, line: 12, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !162)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !24)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 2, scope: !153, type: !24)
// CHECK:STDOUT: !160 = !DILocation(line: 13, column: 3, scope: !153)
// CHECK:STDOUT: !161 = !DILocation(line: 12, column: 1, scope: !153)
// CHECK:STDOUT: !162 = !{!154, !155}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.d01354a0ad5f3687", scope: null, file: !123, line: 12, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !172)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !24)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 2, scope: !163, type: !24)
// CHECK:STDOUT: !170 = !DILocation(line: 13, column: 3, scope: !163)
// CHECK:STDOUT: !171 = !DILocation(line: 12, column: 1, scope: !163)
// CHECK:STDOUT: !172 = !{!164, !165}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.37b67b83fe9ed3dd", scope: null, file: !123, line: 12, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !182)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !24)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 2, scope: !173, type: !24)
// CHECK:STDOUT: !180 = !DILocation(line: 13, column: 3, scope: !173)
// CHECK:STDOUT: !181 = !DILocation(line: 12, column: 1, scope: !173)
// CHECK:STDOUT: !182 = !{!174, !175}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.7096323a0fb88cc3", scope: null, file: !123, line: 12, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !192)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !24)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !24)
// CHECK:STDOUT: !190 = !DILocation(line: 13, column: 3, scope: !183)
// CHECK:STDOUT: !191 = !DILocation(line: 12, column: 1, scope: !183)
// CHECK:STDOUT: !192 = !{!184, !185}
// CHECK:STDOUT:
