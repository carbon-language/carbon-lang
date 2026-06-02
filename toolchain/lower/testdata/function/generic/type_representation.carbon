// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/type_representation.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/type_representation.carbon

interface Copy {
  fn Op[self: Self]() -> Self;
}

fn F[T:! Copy & Core.Destroy](a: T) -> T {
  var _: T = a.(Copy.Op)();
  return a.(Copy.Op)();
}

impl i32 as Copy {
  fn Op[self: i32]() -> i32 {
    return self;
  }
}

fn F_i32(a: i32) -> i32 {
  return F(a);
}

class X {
  var a: i32;
  var b: i32;

  impl as Copy {
    fn Op[self: X]() -> X {
      return {.a = self.a, .b = self.b};
    }
  }
}

fn F_X(a: X) -> X {
  return F(a);
}

impl () as Copy {
  fn Op[unused self: ()]() -> () {
    return ();
  }
}

fn F_empty_tuple(a: ()) -> () {
  return F(a);
}

impl (i32, i32) as Copy {
  fn Op[self: (i32, i32)]() -> (i32, i32) {
    return self;
  }
}

fn F_two_tuple(a: (i32, i32)) -> (i32, i32) {
  return F(a);
}

impl ((i32, i32), X) as Copy {
  fn Op[self: ((i32, i32), X)]() -> ((i32, i32), X) {
    return (self.0.(Copy.Op)(), self.1.(Copy.Op)());
  }
}

fn F_nested_tuple(a: ((i32, i32), X)) -> ((i32, i32), X) {
  return F(a);
}

// CHECK:STDOUT: ; ModuleID = 'type_representation.carbon'
// CHECK:STDOUT: source_filename = "type_representation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF_i32.Main(i32 %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.165a5d00c312ef9e(i32 %a), !dbg !14
// CHECK:STDOUT:   ret i32 %F.call, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.X.Main:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_24.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !28
// CHECK:STDOUT:   %.loc38_24.2 = load i32, ptr %.loc38_24.1.a, align 4, !dbg !28
// CHECK:STDOUT:   %.loc38_37.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !29
// CHECK:STDOUT:   %.loc38_37.2 = load i32, ptr %.loc38_37.1.b, align 4, !dbg !29
// CHECK:STDOUT:   %.loc38_39.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   store i32 %.loc38_24.2, ptr %.loc38_39.2.a, align 4, !dbg !30
// CHECK:STDOUT:   %.loc38_39.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   store i32 %.loc38_37.2, ptr %.loc38_39.4.b, align 4, !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_X.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.e4214cdf04e44ea0(ptr %return, ptr %a), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.57e275a610bca340:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6956ff2f835d4aac:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.61ea2aba74ab3bf1:Copy.Main"() #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_empty_tuple.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.6942691454f579cd(), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.87cc08e781602b22:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem0.loc59_12.1.tuple.elem, align 4, !dbg !57
// CHECK:STDOUT:   %tuple.elem0.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   store i32 %tuple.elem0.loc59_12.1.tuple.elem.load, ptr %tuple.elem0.loc59_12.2.tuple.elem, align 4, !dbg !57
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !57
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem1.loc59_12.1.tuple.elem, align 4, !dbg !57
// CHECK:STDOUT:   %tuple.elem1.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !57
// CHECK:STDOUT:   store i32 %tuple.elem1.loc59_12.1.tuple.elem.load, ptr %tuple.elem1.loc59_12.2.tuple.elem, align 4, !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_two_tuple.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.52eb52f0ffbd1db6(ptr %return, ptr %a), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.26f166843e861eec:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.b1957bb3c5dab115:Copy.Main"(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 0, !dbg !71
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem.load = load ptr, ptr %tuple.elem0.loc69_17.tuple.elem, align 8, !dbg !71
// CHECK:STDOUT:   %tuple.elem0.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   call void @"_COp.87cc08e781602b22:Copy.Main"(ptr %tuple.elem0.loc69_51.tuple.elem, ptr %tuple.elem0.loc69_17.tuple.elem.load), !dbg !71
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 1, !dbg !73
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69_37.tuple.elem, align 8, !dbg !73
// CHECK:STDOUT:   %tuple.elem1.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !72
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %tuple.elem1.loc69_51.tuple.elem, ptr %tuple.elem1.loc69_37.tuple.elem.load), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_nested_tuple.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.cbd425227ee9ed4b(ptr %return, ptr %a), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1d554fde8ca0d7fe:core.Destroy.Core"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.165a5d00c312ef9e(i32 %a) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !87
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc18 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !88
// CHECK:STDOUT:   store i32 %Copy.WithSelf.Op.call.loc18, ptr %_.var, align 4, !dbg !87
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc19 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !89
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %_.var), !dbg !87
// CHECK:STDOUT:   ret i32 %Copy.WithSelf.Op.call.loc19, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.e4214cdf04e44ea0(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !94
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %_.var, ptr %a), !dbg !95
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %return, ptr %a), !dbg !96
// CHECK:STDOUT:   call void @"_COp.6956ff2f835d4aac:core.Destroy.Core"(ptr %_.var), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.6942691454f579cd() #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !99
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !100
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.52eb52f0ffbd1db6(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !106
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !106
// CHECK:STDOUT:   call void @"_COp.87cc08e781602b22:Copy.Main"(ptr %_.var, ptr %a), !dbg !107
// CHECK:STDOUT:   call void @"_COp.87cc08e781602b22:Copy.Main"(ptr %return, ptr %a), !dbg !108
// CHECK:STDOUT:   call void @"_COp.26f166843e861eec:core.Destroy.Core"(ptr %_.var), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.cbd425227ee9ed4b(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { i32, i32 }, { i32, i32 } }, align 4, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !113
// CHECK:STDOUT:   call void @"_COp.b1957bb3c5dab115:Copy.Main"(ptr %_.var, ptr %a), !dbg !114
// CHECK:STDOUT:   call void @"_COp.b1957bb3c5dab115:Copy.Main"(ptr %return, ptr %a), !dbg !115
// CHECK:STDOUT:   call void @"_COp.1d554fde8ca0d7fe:core.Destroy.Core"(ptr %_.var), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "type_representation.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.be1e879c1ad406d8.Core:Copy.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 24, column: 5, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F_i32", linkageName: "_CF_i32.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 29, column: 10, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 29, column: 3, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 29, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null, !7}
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !16, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 29, column: 10, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.X.Main:Copy.Main", scope: null, file: !3, line: 37, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{!25, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 38, column: 20, scope: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 38, column: 33, scope: !22)
// CHECK:STDOUT: !30 = !DILocation(line: 38, column: 14, scope: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 38, column: 7, scope: !22)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F_X", linkageName: "_CF_X.Main", scope: null, file: !3, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !25)
// CHECK:STDOUT: !35 = !DILocation(line: 44, column: 10, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 44, column: 3, scope: !32)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.57e275a610bca340:core.Destroy.Core", scope: null, file: !3, line: 44, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{null, !25}
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !37, type: !25)
// CHECK:STDOUT: !42 = !DILocation(line: 44, column: 10, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6956ff2f835d4aac:core.Destroy.Core", scope: null, file: !3, line: 44, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !25)
// CHECK:STDOUT: !46 = !DILocation(line: 44, column: 10, scope: !43)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.61ea2aba74ab3bf1:Copy.Main", scope: null, file: !3, line: 48, type: !48, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{null}
// CHECK:STDOUT: !50 = !DILocation(line: 49, column: 5, scope: !47)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "F_empty_tuple", linkageName: "_CF_empty_tuple.Main", scope: null, file: !3, line: 53, type: !48, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !52 = !DILocation(line: 54, column: 10, scope: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 54, column: 3, scope: !51)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.87cc08e781602b22:Copy.Main", scope: null, file: !3, line: 58, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !25)
// CHECK:STDOUT: !57 = !DILocation(line: 59, column: 12, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 59, column: 5, scope: !54)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "F_two_tuple", linkageName: "_CF_two_tuple.Main", scope: null, file: !3, line: 63, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !60)
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !59, type: !25)
// CHECK:STDOUT: !62 = !DILocation(line: 64, column: 10, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 64, column: 3, scope: !59)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.26f166843e861eec:core.Destroy.Core", scope: null, file: !3, line: 64, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !25)
// CHECK:STDOUT: !67 = !DILocation(line: 64, column: 10, scope: !64)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b1957bb3c5dab115:Copy.Main", scope: null, file: !3, line: 68, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !69)
// CHECK:STDOUT: !69 = !{!70}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !68, type: !25)
// CHECK:STDOUT: !71 = !DILocation(line: 69, column: 13, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 69, column: 12, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 69, column: 33, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 69, column: 5, scope: !68)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "F_nested_tuple", linkageName: "_CF_nested_tuple.Main", scope: null, file: !3, line: 73, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !76)
// CHECK:STDOUT: !76 = !{!77}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !75, type: !25)
// CHECK:STDOUT: !78 = !DILocation(line: 74, column: 10, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 74, column: 3, scope: !75)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1d554fde8ca0d7fe:core.Destroy.Core", scope: null, file: !3, line: 74, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !81)
// CHECK:STDOUT: !81 = !{!82}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !80, type: !25)
// CHECK:STDOUT: !83 = !DILocation(line: 74, column: 10, scope: !80)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.165a5d00c312ef9e", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !85)
// CHECK:STDOUT: !85 = !{!86}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !84, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 18, column: 3, scope: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 18, column: 14, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 19, column: 10, scope: !84)
// CHECK:STDOUT: !90 = !DILocation(line: 19, column: 3, scope: !84)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e4214cdf04e44ea0", scope: null, file: !3, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !25)
// CHECK:STDOUT: !94 = !DILocation(line: 18, column: 3, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 18, column: 14, scope: !91)
// CHECK:STDOUT: !96 = !DILocation(line: 19, column: 10, scope: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 19, column: 3, scope: !91)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.6942691454f579cd", scope: null, file: !3, line: 17, type: !48, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 3, scope: !98)
// CHECK:STDOUT: !100 = !DILocation(line: 18, column: 14, scope: !98)
// CHECK:STDOUT: !101 = !DILocation(line: 19, column: 10, scope: !98)
// CHECK:STDOUT: !102 = !DILocation(line: 19, column: 3, scope: !98)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.52eb52f0ffbd1db6", scope: null, file: !3, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !104)
// CHECK:STDOUT: !104 = !{!105}
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !103, type: !25)
// CHECK:STDOUT: !106 = !DILocation(line: 18, column: 3, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 18, column: 14, scope: !103)
// CHECK:STDOUT: !108 = !DILocation(line: 19, column: 10, scope: !103)
// CHECK:STDOUT: !109 = !DILocation(line: 19, column: 3, scope: !103)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.cbd425227ee9ed4b", scope: null, file: !3, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !111)
// CHECK:STDOUT: !111 = !{!112}
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !110, type: !25)
// CHECK:STDOUT: !113 = !DILocation(line: 18, column: 3, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 18, column: 14, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 19, column: 10, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 19, column: 3, scope: !110)
