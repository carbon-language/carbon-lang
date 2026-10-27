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
  fn Op(self) -> Self;
}

fn F[T: Copy & Core.Destroy](a: T) -> T {
  var _: T = a.(Copy.Op)();
  return a.(Copy.Op)();
}

impl i32 as Copy {
  fn Op(self: i32) -> i32 {
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
    fn Op(self: X) -> X {
      return {.a = self.a, .b = self.b};
    }
  }
}

fn F_X(a: X) -> X {
  return F(a);
}

impl () as Copy {
  fn Op(unused self: ()) -> () {
    return ();
  }
}

fn F_empty_tuple(a: ()) -> () {
  return F(a);
}

impl (i32, i32) as Copy {
  fn Op(self: (i32, i32)) -> (i32, i32) {
    return self;
  }
}

fn F_two_tuple(a: (i32, i32)) -> (i32, i32) {
  return F(a);
}

impl ((i32, i32), X) as Copy {
  fn Op(self: ((i32, i32), X)) -> ((i32, i32), X) {
    return (self.0.(Copy.Op)(), self.1.(Copy.Op)());
  }
}

fn F_nested_tuple(a: ((i32, i32), X)) -> ((i32, i32), X) {
  return F(a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'type_representation.carbon'
// CHECK:STDOUT: source_filename = "type_representation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF_i32.Main(i32 %a) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.82606fae8d66d37c(i32 %a), !dbg !42
// CHECK:STDOUT:   ret i32 %F.call, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.X.Main:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_24.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   %.loc38_24.2 = load i32, ptr %.loc38_24.1.a, align 4, !dbg !50
// CHECK:STDOUT:   %.loc38_37.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   %.loc38_37.2 = load i32, ptr %.loc38_37.1.b, align 4, !dbg !51
// CHECK:STDOUT:   %.loc38_39.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   store i32 %.loc38_24.2, ptr %.loc38_39.2.a, align 4, !dbg !52
// CHECK:STDOUT:   %.loc38_39.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !52
// CHECK:STDOUT:   store i32 %.loc38_37.2, ptr %.loc38_39.4.b, align 4, !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_X.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.821738257746b808(ptr %return, ptr %a), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.61ea2aba74ab3bf1:Copy.Main"() #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_empty_tuple.Main() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.45674e9c4b6d1871(), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.35f6012581c5aabb:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !81
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem0.loc59_12.1.tuple.elem, align 4, !dbg !81
// CHECK:STDOUT:   %tuple.elem0.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !81
// CHECK:STDOUT:   store i32 %tuple.elem0.loc59_12.1.tuple.elem.load, ptr %tuple.elem0.loc59_12.2.tuple.elem, align 4, !dbg !81
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !81
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem1.loc59_12.1.tuple.elem, align 4, !dbg !81
// CHECK:STDOUT:   %tuple.elem1.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !81
// CHECK:STDOUT:   store i32 %tuple.elem1.loc59_12.1.tuple.elem.load, ptr %tuple.elem1.loc59_12.2.tuple.elem, align 4, !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_two_tuple.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.10c8e74cd85d15bb(ptr %return, ptr %a), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.3fdbec21075a186f:Copy.Main"(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 0, !dbg !98
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem.load = load ptr, ptr %tuple.elem0.loc69_17.tuple.elem, align 8, !dbg !98
// CHECK:STDOUT:   %tuple.elem0.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !99
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:Copy.Main"(ptr %tuple.elem0.loc69_51.tuple.elem, ptr %tuple.elem0.loc69_17.tuple.elem.load), !dbg !98
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 1, !dbg !100
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69_37.tuple.elem, align 8, !dbg !100
// CHECK:STDOUT:   %tuple.elem1.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !99
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %tuple.elem1.loc69_51.tuple.elem, ptr %tuple.elem1.loc69_37.tuple.elem.load), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_nested_tuple.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.2ed3f5f1d0ae313e(ptr %return, ptr %a), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.82606fae8d66d37c(i32 %a) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !117
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !117
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc18 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !118
// CHECK:STDOUT:   store i32 %Copy.WithSelf.Op.call.loc18, ptr %_.var, align 4, !dbg !117
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc19 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !120
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %_.var), !dbg !117
// CHECK:STDOUT:   ret i32 %Copy.WithSelf.Op.call.loc19, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.821738257746b808(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !130
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !130
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %_.var, ptr %a), !dbg !131
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %return, ptr %a), !dbg !133
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2abb6e2621dcf00d(ptr %_.var), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.45674e9c4b6d1871() #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !142
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !143
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !145
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %_.var), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.10c8e74cd85d15bb(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !152
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:Copy.Main"(ptr %_.var, ptr %a), !dbg !153
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:Copy.Main"(ptr %return, ptr %a), !dbg !155
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %_.var), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.2ed3f5f1d0ae313e(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { i32, i32 }, { i32, i32 } }, align 4, !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !163
// CHECK:STDOUT:   call void @"_COp.3fdbec21075a186f:Copy.Main"(ptr %_.var, ptr %a), !dbg !164
// CHECK:STDOUT:   call void @"_COp.3fdbec21075a186f:Copy.Main"(ptr %return, ptr %a), !dbg !166
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1856a248bc68ad79(ptr %_.var), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !169 {
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2abb6e2621dcf00d(ptr %t) #0 !dbg !176 {
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %t) #0 !dbg !183 {
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %t) #0 !dbg !190 {
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1856a248bc68ad79(ptr %t) #0 !dbg !197 {
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "type_representation.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.be1e879c1ad406d8.Core:Copy.Main", scope: null, file: !18, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 24, column: 5, scope: !28)
// CHECK:STDOUT: !35 = !{!29}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "F_i32", linkageName: "_CF_i32.Main", scope: null, file: !18, line: 28, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !42 = !DILocation(line: 29, column: 10, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 29, column: 3, scope: !36)
// CHECK:STDOUT: !44 = !{!37}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.X.Main:Copy.Main", scope: null, file: !18, line: 37, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 38, column: 20, scope: !45)
// CHECK:STDOUT: !51 = !DILocation(line: 38, column: 33, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 38, column: 14, scope: !45)
// CHECK:STDOUT: !53 = !DILocation(line: 38, column: 7, scope: !45)
// CHECK:STDOUT: !54 = !{!46}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "F_X", linkageName: "_CF_X.Main", scope: null, file: !18, line: 43, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 44, column: 10, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 44, column: 3, scope: !55)
// CHECK:STDOUT: !63 = !{!56}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.61ea2aba74ab3bf1:Copy.Main", scope: null, file: !18, line: 48, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !68 = !DILocation(line: 49, column: 5, scope: !64)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "F_empty_tuple", linkageName: "_CF_empty_tuple.Main", scope: null, file: !18, line: 53, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !74 = !DILocation(line: 54, column: 10, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 54, column: 3, scope: !69)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Op", linkageName: "_COp.35f6012581c5aabb:Copy.Main", scope: null, file: !18, line: 58, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !83)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !23)
// CHECK:STDOUT: !81 = !DILocation(line: 59, column: 12, scope: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 59, column: 5, scope: !76)
// CHECK:STDOUT: !83 = !{!77}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "F_two_tuple", linkageName: "_CF_two_tuple.Main", scope: null, file: !18, line: 63, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !92)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !23)
// CHECK:STDOUT: !90 = !DILocation(line: 64, column: 10, scope: !84)
// CHECK:STDOUT: !91 = !DILocation(line: 64, column: 3, scope: !84)
// CHECK:STDOUT: !92 = !{!85}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3fdbec21075a186f:Copy.Main", scope: null, file: !18, line: 68, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !102)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !23)
// CHECK:STDOUT: !98 = !DILocation(line: 69, column: 13, scope: !93)
// CHECK:STDOUT: !99 = !DILocation(line: 69, column: 12, scope: !93)
// CHECK:STDOUT: !100 = !DILocation(line: 69, column: 33, scope: !93)
// CHECK:STDOUT: !101 = !DILocation(line: 69, column: 5, scope: !93)
// CHECK:STDOUT: !102 = !{!94}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "F_nested_tuple", linkageName: "_CF_nested_tuple.Main", scope: null, file: !18, line: 73, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !111)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !23)
// CHECK:STDOUT: !109 = !DILocation(line: 74, column: 10, scope: !103)
// CHECK:STDOUT: !110 = !DILocation(line: 74, column: 3, scope: !103)
// CHECK:STDOUT: !111 = !{!104}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.82606fae8d66d37c", scope: null, file: !18, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !124)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !20)
// CHECK:STDOUT: !117 = !DILocation(line: 18, column: 3, scope: !112)
// CHECK:STDOUT: !118 = !DILocation(line: 18, column: 14, scope: !112)
// CHECK:STDOUT: !120 = !DILocation(line: 19, column: 10, scope: !112)
// CHECK:STDOUT: !121 = !{null, !20}
// CHECK:STDOUT: !122 = !DISubroutineType(types: !121)
// CHECK:STDOUT: !123 = !DILocation(line: 19, column: 3, scope: !112)
// CHECK:STDOUT: !124 = !{!113}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.821738257746b808", scope: null, file: !18, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !137)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !23)
// CHECK:STDOUT: !130 = !DILocation(line: 18, column: 3, scope: !125)
// CHECK:STDOUT: !131 = !DILocation(line: 18, column: 14, scope: !125)
// CHECK:STDOUT: !133 = !DILocation(line: 19, column: 10, scope: !125)
// CHECK:STDOUT: !134 = !{null, !23}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 19, column: 3, scope: !125)
// CHECK:STDOUT: !137 = !{!126}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.45674e9c4b6d1871", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !142 = !DILocation(line: 18, column: 3, scope: !138)
// CHECK:STDOUT: !143 = !DILocation(line: 18, column: 14, scope: !138)
// CHECK:STDOUT: !145 = !DILocation(line: 19, column: 10, scope: !138)
// CHECK:STDOUT: !146 = !DILocation(line: 19, column: 3, scope: !138)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.10c8e74cd85d15bb", scope: null, file: !18, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !157)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !23)
// CHECK:STDOUT: !152 = !DILocation(line: 18, column: 3, scope: !147)
// CHECK:STDOUT: !153 = !DILocation(line: 18, column: 14, scope: !147)
// CHECK:STDOUT: !155 = !DILocation(line: 19, column: 10, scope: !147)
// CHECK:STDOUT: !156 = !DILocation(line: 19, column: 3, scope: !147)
// CHECK:STDOUT: !157 = !{!148}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2ed3f5f1d0ae313e", scope: null, file: !18, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !168)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !23)
// CHECK:STDOUT: !163 = !DILocation(line: 18, column: 3, scope: !158)
// CHECK:STDOUT: !164 = !DILocation(line: 18, column: 14, scope: !158)
// CHECK:STDOUT: !166 = !DILocation(line: 19, column: 10, scope: !158)
// CHECK:STDOUT: !167 = !DILocation(line: 19, column: 3, scope: !158)
// CHECK:STDOUT: !168 = !{!159}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !122, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !175)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !20)
// CHECK:STDOUT: !174 = !DILocation(line: 32, column: 1, scope: !169)
// CHECK:STDOUT: !175 = !{!170}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2abb6e2621dcf00d", scope: null, file: !10, line: 32, type: !135, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !182)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !23)
// CHECK:STDOUT: !181 = !DILocation(line: 32, column: 1, scope: !176)
// CHECK:STDOUT: !182 = !{!177}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9045abb8230cbcdf", scope: null, file: !10, line: 32, type: !135, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !189)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !23)
// CHECK:STDOUT: !188 = !DILocation(line: 32, column: 1, scope: !183)
// CHECK:STDOUT: !189 = !{!184}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !10, line: 32, type: !135, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !196)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !23)
// CHECK:STDOUT: !195 = !DILocation(line: 32, column: 1, scope: !190)
// CHECK:STDOUT: !196 = !{!191}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1856a248bc68ad79", scope: null, file: !10, line: 32, type: !135, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !203)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !23)
// CHECK:STDOUT: !202 = !DILocation(line: 32, column: 1, scope: !197)
// CHECK:STDOUT: !203 = !{!198}
// CHECK:STDOUT:
