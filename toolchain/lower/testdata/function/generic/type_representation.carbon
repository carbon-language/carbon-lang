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
// CHECK:STDOUT: define i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF_i32.Main(i32 %a) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.51815a09c4c03ee5(i32 %a), !dbg !46
// CHECK:STDOUT:   ret i32 %F.call, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.X.Main:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_24.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   %.loc38_24.2 = load i32, ptr %.loc38_24.1.a, align 4, !dbg !58
// CHECK:STDOUT:   %.loc38_37.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !59
// CHECK:STDOUT:   %.loc38_37.2 = load i32, ptr %.loc38_37.1.b, align 4, !dbg !59
// CHECK:STDOUT:   %.loc38_39.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   store i32 %.loc38_24.2, ptr %.loc38_39.2.a, align 4, !dbg !60
// CHECK:STDOUT:   %.loc38_39.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !60
// CHECK:STDOUT:   store i32 %.loc38_37.2, ptr %.loc38_39.4.b, align 4, !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_X.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.425841177dc25b60(ptr %return, ptr %a), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d0e8d3e870241846:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.617e09fd62745f5d:core.Destroy.Core"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.61ea2aba74ab3bf1:Copy.Main"() #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_empty_tuple.Main() #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.0ffdc268e95480c6(), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.e554695bf635d9c1:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !97
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem0.loc59_12.1.tuple.elem, align 4, !dbg !97
// CHECK:STDOUT:   %tuple.elem0.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !97
// CHECK:STDOUT:   store i32 %tuple.elem0.loc59_12.1.tuple.elem.load, ptr %tuple.elem0.loc59_12.2.tuple.elem, align 4, !dbg !97
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !97
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem1.loc59_12.1.tuple.elem, align 4, !dbg !97
// CHECK:STDOUT:   %tuple.elem1.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !97
// CHECK:STDOUT:   store i32 %tuple.elem1.loc59_12.1.tuple.elem.load, ptr %tuple.elem1.loc59_12.2.tuple.elem, align 4, !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_two_tuple.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fa2890b7def5e5c7(ptr %return, ptr %a), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ebc50e8bd9eb2f1:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.ae1e3622d7875d32:Copy.Main"(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %self) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 0, !dbg !118
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem.load = load ptr, ptr %tuple.elem0.loc69_17.tuple.elem, align 8, !dbg !118
// CHECK:STDOUT:   %tuple.elem0.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !119
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:Copy.Main"(ptr %tuple.elem0.loc69_51.tuple.elem, ptr %tuple.elem0.loc69_17.tuple.elem.load), !dbg !118
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 1, !dbg !120
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69_37.tuple.elem, align 8, !dbg !120
// CHECK:STDOUT:   %tuple.elem1.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !119
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %tuple.elem1.loc69_51.tuple.elem, ptr %tuple.elem1.loc69_37.tuple.elem.load), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_nested_tuple.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.2353b6e2f4825328(ptr %return, ptr %a), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1c942cf0c4ee9edc:core.Destroy.Core"(ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.51815a09c4c03ee5(i32 %a) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !141
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !141
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc18 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !142
// CHECK:STDOUT:   store i32 %Copy.WithSelf.Op.call.loc18, ptr %_.var, align 4, !dbg !141
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc19 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !144
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %_.var), !dbg !141
// CHECK:STDOUT:   ret i32 %Copy.WithSelf.Op.call.loc19, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.425841177dc25b60(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !152
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %_.var, ptr %a), !dbg !153
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %return, ptr %a), !dbg !155
// CHECK:STDOUT:   call void @"_COp.617e09fd62745f5d:core.Destroy.Core"(ptr %_.var), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.0ffdc268e95480c6() #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !162
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !163
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fa2890b7def5e5c7(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !172
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:Copy.Main"(ptr %_.var, ptr %a), !dbg !173
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:Copy.Main"(ptr %return, ptr %a), !dbg !175
// CHECK:STDOUT:   call void @"_COp.7ebc50e8bd9eb2f1:core.Destroy.Core"(ptr %_.var), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.2353b6e2f4825328(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { i32, i32 }, { i32, i32 } }, align 4, !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !183
// CHECK:STDOUT:   call void @"_COp.ae1e3622d7875d32:Copy.Main"(ptr %_.var, ptr %a), !dbg !184
// CHECK:STDOUT:   call void @"_COp.ae1e3622d7875d32:Copy.Main"(ptr %return, ptr %a), !dbg !186
// CHECK:STDOUT:   call void @"_COp.1c942cf0c4ee9edc:core.Destroy.Core"(ptr %_.var), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !187
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "type_representation.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!25, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !25}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.be1e879c1ad406d8.Core:Copy.Main", scope: null, file: !18, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !38 = !DILocation(line: 24, column: 5, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F_i32", linkageName: "_CF_i32.Main", scope: null, file: !18, line: 28, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !20)
// CHECK:STDOUT: !46 = !DILocation(line: 29, column: 10, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 29, column: 3, scope: !40)
// CHECK:STDOUT: !48 = !{!41}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !18, line: 29, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !20)
// CHECK:STDOUT: !51 = !DILocation(line: 29, column: 10, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.X.Main:Copy.Main", scope: null, file: !18, line: 37, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !62)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !25)
// CHECK:STDOUT: !58 = !DILocation(line: 38, column: 20, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 38, column: 33, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 38, column: 14, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 38, column: 7, scope: !53)
// CHECK:STDOUT: !62 = !{!54}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "F_X", linkageName: "_CF_X.Main", scope: null, file: !18, line: 43, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !25)
// CHECK:STDOUT: !69 = !DILocation(line: 44, column: 10, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 44, column: 3, scope: !63)
// CHECK:STDOUT: !71 = !{!64}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d0e8d3e870241846:core.Destroy.Core", scope: null, file: !18, line: 44, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !25)
// CHECK:STDOUT: !74 = !DILocation(line: 44, column: 10, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Op", linkageName: "_COp.617e09fd62745f5d:core.Destroy.Core", scope: null, file: !18, line: 44, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !25)
// CHECK:STDOUT: !78 = !DILocation(line: 44, column: 10, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Op", linkageName: "_COp.61ea2aba74ab3bf1:Copy.Main", scope: null, file: !18, line: 48, type: !31, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !84 = !DILocation(line: 49, column: 5, scope: !80)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "F_empty_tuple", linkageName: "_CF_empty_tuple.Main", scope: null, file: !18, line: 53, type: !31, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !90 = !DILocation(line: 54, column: 10, scope: !85)
// CHECK:STDOUT: !91 = !DILocation(line: 54, column: 3, scope: !85)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e554695bf635d9c1:Copy.Main", scope: null, file: !18, line: 58, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !99)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !25)
// CHECK:STDOUT: !97 = !DILocation(line: 59, column: 12, scope: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 59, column: 5, scope: !92)
// CHECK:STDOUT: !99 = !{!93}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "F_two_tuple", linkageName: "_CF_two_tuple.Main", scope: null, file: !18, line: 63, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !108)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !25)
// CHECK:STDOUT: !106 = !DILocation(line: 64, column: 10, scope: !100)
// CHECK:STDOUT: !107 = !DILocation(line: 64, column: 3, scope: !100)
// CHECK:STDOUT: !108 = !{!101}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ebc50e8bd9eb2f1:core.Destroy.Core", scope: null, file: !18, line: 64, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !25)
// CHECK:STDOUT: !111 = !DILocation(line: 64, column: 10, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ae1e3622d7875d32:Copy.Main", scope: null, file: !18, line: 68, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !122)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !25)
// CHECK:STDOUT: !118 = !DILocation(line: 69, column: 13, scope: !113)
// CHECK:STDOUT: !119 = !DILocation(line: 69, column: 12, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 69, column: 33, scope: !113)
// CHECK:STDOUT: !121 = !DILocation(line: 69, column: 5, scope: !113)
// CHECK:STDOUT: !122 = !{!114}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "F_nested_tuple", linkageName: "_CF_nested_tuple.Main", scope: null, file: !18, line: 73, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !131)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !25)
// CHECK:STDOUT: !129 = !DILocation(line: 74, column: 10, scope: !123)
// CHECK:STDOUT: !130 = !DILocation(line: 74, column: 3, scope: !123)
// CHECK:STDOUT: !131 = !{!124}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1c942cf0c4ee9edc:core.Destroy.Core", scope: null, file: !18, line: 74, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !135)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !25)
// CHECK:STDOUT: !134 = !DILocation(line: 74, column: 10, scope: !132)
// CHECK:STDOUT: !135 = !{!133}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.51815a09c4c03ee5", scope: null, file: !18, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !146)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !20)
// CHECK:STDOUT: !141 = !DILocation(line: 18, column: 3, scope: !136)
// CHECK:STDOUT: !142 = !DILocation(line: 18, column: 14, scope: !136)
// CHECK:STDOUT: !144 = !DILocation(line: 19, column: 10, scope: !136)
// CHECK:STDOUT: !145 = !DILocation(line: 19, column: 3, scope: !136)
// CHECK:STDOUT: !146 = !{!137}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.425841177dc25b60", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !157)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !25)
// CHECK:STDOUT: !152 = !DILocation(line: 18, column: 3, scope: !147)
// CHECK:STDOUT: !153 = !DILocation(line: 18, column: 14, scope: !147)
// CHECK:STDOUT: !155 = !DILocation(line: 19, column: 10, scope: !147)
// CHECK:STDOUT: !156 = !DILocation(line: 19, column: 3, scope: !147)
// CHECK:STDOUT: !157 = !{!148}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0ffdc268e95480c6", scope: null, file: !18, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !162 = !DILocation(line: 18, column: 3, scope: !158)
// CHECK:STDOUT: !163 = !DILocation(line: 18, column: 14, scope: !158)
// CHECK:STDOUT: !165 = !DILocation(line: 19, column: 10, scope: !158)
// CHECK:STDOUT: !166 = !DILocation(line: 19, column: 3, scope: !158)
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fa2890b7def5e5c7", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !177)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !25)
// CHECK:STDOUT: !172 = !DILocation(line: 18, column: 3, scope: !167)
// CHECK:STDOUT: !173 = !DILocation(line: 18, column: 14, scope: !167)
// CHECK:STDOUT: !175 = !DILocation(line: 19, column: 10, scope: !167)
// CHECK:STDOUT: !176 = !DILocation(line: 19, column: 3, scope: !167)
// CHECK:STDOUT: !177 = !{!168}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2353b6e2f4825328", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !188)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !25)
// CHECK:STDOUT: !183 = !DILocation(line: 18, column: 3, scope: !178)
// CHECK:STDOUT: !184 = !DILocation(line: 18, column: 14, scope: !178)
// CHECK:STDOUT: !186 = !DILocation(line: 19, column: 10, scope: !178)
// CHECK:STDOUT: !187 = !DILocation(line: 19, column: 3, scope: !178)
// CHECK:STDOUT: !188 = !{!179}
// CHECK:STDOUT:
