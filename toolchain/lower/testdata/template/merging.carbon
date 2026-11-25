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
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.2.temp), !dbg !52
// CHECK:STDOUT:   call void @_CCallBF.Main.9522ff908e97e267(ptr @C.val.loc33_13.3), !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_11.2.temp), !dbg !54
// CHECK:STDOUT:   call void @_CCallBF.Main.931317eeae6ffe17(ptr @D.val.loc34_13.3), !dbg !55
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
// CHECK:STDOUT:   %.2a64a1.4.temp = alloca {}, align 1, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.4.temp), !dbg !64
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.4.temp, ptr @C.val.loc33_13.3), !dbg !64
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.4.temp), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CInitAndCallF.Main.931317eeae6ffe17() #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.3.temp = alloca {}, align 1, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.3.temp), !dbg !67
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.3.temp, ptr @D.val.loc34_13.3), !dbg !67
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.3.temp), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.a6a60dfee2d7d273(ptr %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %x), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.9522ff908e97e267(ptr %x) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.2.temp = alloca {}, align 1, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.2.temp), !dbg !78
// CHECK:STDOUT:   call void @"_CConvert.C.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.2.temp, ptr %x), !dbg !78
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.2.temp), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallBF.Main.931317eeae6ffe17(ptr %x) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2a64a1.1.temp = alloca {}, align 1, !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2a64a1.1.temp), !dbg !84
// CHECK:STDOUT:   call void @"_CConvert.D.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.2a64a1.1.temp, ptr %x), !dbg !84
// CHECK:STDOUT:   call void @_CF.B.Main(ptr %.2a64a1.1.temp), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
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
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 1, scope: !63)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "InitAndCallF", linkageName: "_CInitAndCallF.Main.931317eeae6ffe17", scope: null, file: !22, line: 18, type: !7, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !67 = !DILocation(line: 18, column: 37, scope: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 1, scope: !66)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.a6a60dfee2d7d273", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !74)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !24)
// CHECK:STDOUT: !72 = !DILocation(line: 28, column: 3, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 27, column: 1, scope: !69)
// CHECK:STDOUT: !74 = !{!70}
// CHECK:STDOUT: !75 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6579a5013ba3dcd2", scope: null, file: !75, line: 35, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !82)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !24)
// CHECK:STDOUT: !81 = !DILocation(line: 35, column: 1, scope: !76)
// CHECK:STDOUT: !82 = !{!77}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.9522ff908e97e267", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !88)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !24)
// CHECK:STDOUT: !86 = !DILocation(line: 28, column: 3, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 27, column: 1, scope: !83)
// CHECK:STDOUT: !88 = !{!84}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !75, line: 35, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !95)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !24)
// CHECK:STDOUT: !94 = !DILocation(line: 35, column: 1, scope: !89)
// CHECK:STDOUT: !95 = !{!90}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "CallBF", linkageName: "_CCallBF.Main.931317eeae6ffe17", scope: null, file: !22, line: 27, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !101)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !24)
// CHECK:STDOUT: !99 = !DILocation(line: 28, column: 3, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 27, column: 1, scope: !96)
// CHECK:STDOUT: !101 = !{!97}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0ee92c77788c7639", scope: null, file: !75, line: 35, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !24)
// CHECK:STDOUT: !107 = !DILocation(line: 35, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_merge_convert_one_arg.carbon'
// CHECK:STDOUT: source_filename = "no_merge_convert_one_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc6_45 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @A.val.loc6_45, i64 0, i1 false), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main(ptr %a, ptr %b) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.2877a42045e01a3b(ptr %a, ptr %a), !dbg !105
// CHECK:STDOUT:   call void @_CCallF.Main.d01354a0ad5f3687(ptr %a, ptr %b), !dbg !108
// CHECK:STDOUT:   call void @_CCallF.Main.37b67b83fe9ed3dd(ptr %b, ptr %a), !dbg !111
// CHECK:STDOUT:   call void @_CCallF.Main.7096323a0fb88cc3(ptr %b, ptr %b), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.2877a42045e01a3b(ptr %x, ptr %y) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %y), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.d01354a0ad5f3687(ptr %x, ptr %y) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.4.temp = alloca {}, align 1, !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.4.temp), !dbg !133
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.4.temp, ptr %y), !dbg !133
// CHECK:STDOUT:   call void @_CF.Main(ptr %x, ptr %.9745c1.4.temp), !dbg !134
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.37b67b83fe9ed3dd(ptr %x, ptr %y) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.3.temp = alloca {}, align 1, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.3.temp), !dbg !142
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.3.temp, ptr %x), !dbg !142
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.3.temp, ptr %y), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.7096323a0fb88cc3(ptr %x, ptr %y) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.9745c1.1.temp = alloca {}, align 1, !dbg !152
// CHECK:STDOUT:   %.9745c1.2.temp = alloca {}, align 1, !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.1.temp), !dbg !152
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.1.temp, ptr %x), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.9745c1.2.temp), !dbg !153
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core"(ptr %.9745c1.2.temp, ptr %y), !dbg !153
// CHECK:STDOUT:   call void @_CF.Main(ptr %.9745c1.1.temp, ptr %.9745c1.2.temp), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !155
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!88}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{!24, !24}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !87 = !DIFile(filename: "no_merge_convert_one_arg.carbon", directory: "")
// CHECK:STDOUT: !88 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !87, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !89 = !{null, !24, !24}
// CHECK:STDOUT: !90 = !DISubroutineType(types: !89)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.bd4fb75a414fe858.Core", scope: null, file: !87, line: 6, type: !28, spFlags: DISPFlagDefinition, unit: !88, retainedNodes: !97)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !24)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 36, scope: !91)
// CHECK:STDOUT: !97 = !{!92}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !87, line: 16, type: !90, spFlags: DISPFlagDefinition, unit: !88, retainedNodes: !116)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !24)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !24)
// CHECK:STDOUT: !105 = !DILocation(line: 17, column: 3, scope: !98)
// CHECK:STDOUT: !108 = !DILocation(line: 18, column: 3, scope: !98)
// CHECK:STDOUT: !111 = !DILocation(line: 19, column: 3, scope: !98)
// CHECK:STDOUT: !114 = !DILocation(line: 20, column: 3, scope: !98)
// CHECK:STDOUT: !115 = !DILocation(line: 16, column: 1, scope: !98)
// CHECK:STDOUT: !116 = !{!99, !100}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.2877a42045e01a3b", scope: null, file: !87, line: 12, type: !90, spFlags: DISPFlagDefinition, unit: !88, retainedNodes: !126)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !24)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 2, scope: !117, type: !24)
// CHECK:STDOUT: !124 = !DILocation(line: 13, column: 3, scope: !117)
// CHECK:STDOUT: !125 = !DILocation(line: 12, column: 1, scope: !117)
// CHECK:STDOUT: !126 = !{!118, !119}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.d01354a0ad5f3687", scope: null, file: !87, line: 12, type: !90, spFlags: DISPFlagDefinition, unit: !88, retainedNodes: !136)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !24)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 2, scope: !127, type: !24)
// CHECK:STDOUT: !133 = !DILocation(line: 13, column: 8, scope: !127)
// CHECK:STDOUT: !134 = !DILocation(line: 13, column: 3, scope: !127)
// CHECK:STDOUT: !135 = !DILocation(line: 12, column: 1, scope: !127)
// CHECK:STDOUT: !136 = !{!128, !129}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.37b67b83fe9ed3dd", scope: null, file: !87, line: 12, type: !90, spFlags: DISPFlagDefinition, unit: !88, retainedNodes: !146)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !24)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 2, scope: !137, type: !24)
// CHECK:STDOUT: !142 = !DILocation(line: 13, column: 5, scope: !137)
// CHECK:STDOUT: !144 = !DILocation(line: 13, column: 3, scope: !137)
// CHECK:STDOUT: !145 = !DILocation(line: 12, column: 1, scope: !137)
// CHECK:STDOUT: !146 = !{!138, !139}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.7096323a0fb88cc3", scope: null, file: !87, line: 12, type: !90, spFlags: DISPFlagDefinition, unit: !88, retainedNodes: !156)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !24)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !147, type: !24)
// CHECK:STDOUT: !152 = !DILocation(line: 13, column: 5, scope: !147)
// CHECK:STDOUT: !153 = !DILocation(line: 13, column: 8, scope: !147)
// CHECK:STDOUT: !154 = !DILocation(line: 13, column: 3, scope: !147)
// CHECK:STDOUT: !155 = !DILocation(line: 12, column: 1, scope: !147)
// CHECK:STDOUT: !156 = !{!148, !149}
// CHECK:STDOUT:
