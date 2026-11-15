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
// CHECK:STDOUT: define i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF_i32.Main(i32 %a) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.82606fae8d66d37c(i32 %a), !dbg !44
// CHECK:STDOUT:   ret i32 %F.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.X.Main:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_24.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   %.loc38_24.2 = load i32, ptr %.loc38_24.1.a, align 4, !dbg !52
// CHECK:STDOUT:   %.loc38_37.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   %.loc38_37.2 = load i32, ptr %.loc38_37.1.b, align 4, !dbg !53
// CHECK:STDOUT:   %.loc38_39.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !54
// CHECK:STDOUT:   store i32 %.loc38_24.2, ptr %.loc38_39.2.a, align 4, !dbg !54
// CHECK:STDOUT:   %.loc38_39.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !54
// CHECK:STDOUT:   store i32 %.loc38_37.2, ptr %.loc38_39.4.b, align 4, !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_X.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.821738257746b808(ptr %return, ptr %a), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.61ea2aba74ab3bf1:Copy.Main"() #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_empty_tuple.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.45674e9c4b6d1871(), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.35f6012581c5aabb:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !83
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem0.loc59_12.1.tuple.elem, align 4, !dbg !83
// CHECK:STDOUT:   %tuple.elem0.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !83
// CHECK:STDOUT:   store i32 %tuple.elem0.loc59_12.1.tuple.elem.load, ptr %tuple.elem0.loc59_12.2.tuple.elem, align 4, !dbg !83
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !83
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem1.loc59_12.1.tuple.elem, align 4, !dbg !83
// CHECK:STDOUT:   %tuple.elem1.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !83
// CHECK:STDOUT:   store i32 %tuple.elem1.loc59_12.1.tuple.elem.load, ptr %tuple.elem1.loc59_12.2.tuple.elem, align 4, !dbg !83
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_two_tuple.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.10c8e74cd85d15bb(ptr %return, ptr %a), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.3fdbec21075a186f:Copy.Main"(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 0, !dbg !100
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem.load = load ptr, ptr %tuple.elem0.loc69_17.tuple.elem, align 8, !dbg !100
// CHECK:STDOUT:   %tuple.elem0.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !101
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:Copy.Main"(ptr %tuple.elem0.loc69_51.tuple.elem, ptr %tuple.elem0.loc69_17.tuple.elem.load), !dbg !100
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 1, !dbg !102
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69_37.tuple.elem, align 8, !dbg !102
// CHECK:STDOUT:   %tuple.elem1.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !101
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %tuple.elem1.loc69_51.tuple.elem, ptr %tuple.elem1.loc69_37.tuple.elem.load), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_nested_tuple.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.2ed3f5f1d0ae313e(ptr %return, ptr %a), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.82606fae8d66d37c(i32 %a) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !119
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !119
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc18 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !120
// CHECK:STDOUT:   store i32 %Copy.WithSelf.Op.call.loc18, ptr %_.var, align 4, !dbg !119
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc19 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !122
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
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.45674e9c4b6d1871() #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !140
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !140
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !141
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.10c8e74cd85d15bb(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !150
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:Copy.Main"(ptr %_.var, ptr %a), !dbg !151
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:Copy.Main"(ptr %return, ptr %a), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.2ed3f5f1d0ae313e(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { i32, i32 }, { i32, i32 } }, align 4, !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !161
// CHECK:STDOUT:   call void @"_COp.3fdbec21075a186f:Copy.Main"(ptr %_.var, ptr %a), !dbg !162
// CHECK:STDOUT:   call void @"_COp.3fdbec21075a186f:Copy.Main"(ptr %return, ptr %a), !dbg !164
// CHECK:STDOUT:   ret void, !dbg !165
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "type_representation.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!25, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.be1e879c1ad406d8.Core:Copy.Main", scope: null, file: !20, line: 23, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !37)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 5, scope: !30)
// CHECK:STDOUT: !37 = !{!31}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F_i32", linkageName: "_CF_i32.Main", scope: null, file: !20, line: 28, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !46)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !22)
// CHECK:STDOUT: !44 = !DILocation(line: 29, column: 10, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 3, scope: !38)
// CHECK:STDOUT: !46 = !{!39}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.X.Main:Copy.Main", scope: null, file: !20, line: 37, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !56)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !25)
// CHECK:STDOUT: !52 = !DILocation(line: 38, column: 20, scope: !47)
// CHECK:STDOUT: !53 = !DILocation(line: 38, column: 33, scope: !47)
// CHECK:STDOUT: !54 = !DILocation(line: 38, column: 14, scope: !47)
// CHECK:STDOUT: !55 = !DILocation(line: 38, column: 7, scope: !47)
// CHECK:STDOUT: !56 = !{!48}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "F_X", linkageName: "_CF_X.Main", scope: null, file: !20, line: 43, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !65)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !25)
// CHECK:STDOUT: !63 = !DILocation(line: 44, column: 10, scope: !57)
// CHECK:STDOUT: !64 = !DILocation(line: 44, column: 3, scope: !57)
// CHECK:STDOUT: !65 = !{!58}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.61ea2aba74ab3bf1:Copy.Main", scope: null, file: !20, line: 48, type: !29, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 49, column: 5, scope: !66)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "F_empty_tuple", linkageName: "_CF_empty_tuple.Main", scope: null, file: !20, line: 53, type: !29, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !76 = !DILocation(line: 54, column: 10, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 54, column: 3, scope: !71)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Op", linkageName: "_COp.35f6012581c5aabb:Copy.Main", scope: null, file: !20, line: 58, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !85)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !25)
// CHECK:STDOUT: !83 = !DILocation(line: 59, column: 12, scope: !78)
// CHECK:STDOUT: !84 = !DILocation(line: 59, column: 5, scope: !78)
// CHECK:STDOUT: !85 = !{!79}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "F_two_tuple", linkageName: "_CF_two_tuple.Main", scope: null, file: !20, line: 63, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !94)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !25)
// CHECK:STDOUT: !92 = !DILocation(line: 64, column: 10, scope: !86)
// CHECK:STDOUT: !93 = !DILocation(line: 64, column: 3, scope: !86)
// CHECK:STDOUT: !94 = !{!87}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3fdbec21075a186f:Copy.Main", scope: null, file: !20, line: 68, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !104)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !25)
// CHECK:STDOUT: !100 = !DILocation(line: 69, column: 13, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 69, column: 12, scope: !95)
// CHECK:STDOUT: !102 = !DILocation(line: 69, column: 33, scope: !95)
// CHECK:STDOUT: !103 = !DILocation(line: 69, column: 5, scope: !95)
// CHECK:STDOUT: !104 = !{!96}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "F_nested_tuple", linkageName: "_CF_nested_tuple.Main", scope: null, file: !20, line: 73, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !113)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !25)
// CHECK:STDOUT: !111 = !DILocation(line: 74, column: 10, scope: !105)
// CHECK:STDOUT: !112 = !DILocation(line: 74, column: 3, scope: !105)
// CHECK:STDOUT: !113 = !{!106}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.82606fae8d66d37c", scope: null, file: !20, line: 17, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !124)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !22)
// CHECK:STDOUT: !119 = !DILocation(line: 18, column: 3, scope: !114)
// CHECK:STDOUT: !120 = !DILocation(line: 18, column: 14, scope: !114)
// CHECK:STDOUT: !122 = !DILocation(line: 19, column: 10, scope: !114)
// CHECK:STDOUT: !123 = !DILocation(line: 19, column: 3, scope: !114)
// CHECK:STDOUT: !124 = !{!115}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.821738257746b808", scope: null, file: !20, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !135)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !25)
// CHECK:STDOUT: !130 = !DILocation(line: 18, column: 3, scope: !125)
// CHECK:STDOUT: !131 = !DILocation(line: 18, column: 14, scope: !125)
// CHECK:STDOUT: !133 = !DILocation(line: 19, column: 10, scope: !125)
// CHECK:STDOUT: !134 = !DILocation(line: 19, column: 3, scope: !125)
// CHECK:STDOUT: !135 = !{!126}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.45674e9c4b6d1871", scope: null, file: !20, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !140 = !DILocation(line: 18, column: 3, scope: !136)
// CHECK:STDOUT: !141 = !DILocation(line: 18, column: 14, scope: !136)
// CHECK:STDOUT: !143 = !DILocation(line: 19, column: 10, scope: !136)
// CHECK:STDOUT: !144 = !DILocation(line: 19, column: 3, scope: !136)
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.10c8e74cd85d15bb", scope: null, file: !20, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !155)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !25)
// CHECK:STDOUT: !150 = !DILocation(line: 18, column: 3, scope: !145)
// CHECK:STDOUT: !151 = !DILocation(line: 18, column: 14, scope: !145)
// CHECK:STDOUT: !153 = !DILocation(line: 19, column: 10, scope: !145)
// CHECK:STDOUT: !154 = !DILocation(line: 19, column: 3, scope: !145)
// CHECK:STDOUT: !155 = !{!146}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2ed3f5f1d0ae313e", scope: null, file: !20, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !166)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !25)
// CHECK:STDOUT: !161 = !DILocation(line: 18, column: 3, scope: !156)
// CHECK:STDOUT: !162 = !DILocation(line: 18, column: 14, scope: !156)
// CHECK:STDOUT: !164 = !DILocation(line: 19, column: 10, scope: !156)
// CHECK:STDOUT: !165 = !DILocation(line: 19, column: 3, scope: !156)
// CHECK:STDOUT: !166 = !{!157}
// CHECK:STDOUT:
