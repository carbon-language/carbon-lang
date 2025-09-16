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

fn F[T:! Copy](a: T) -> T {
  var v: T = a.Op();
  return a.Op();
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
  fn Op[self: ()]() -> () {
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
// CHECK:STDOUT: define i32 @"_COp.Int.Core:Copy.Main"(i32 %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF_i32.Main(i32 %a) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.c51f5dcf0e081157(i32 %a), !dbg !9
// CHECK:STDOUT:   ret i32 %F.call, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.X.Main:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_24.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc38_24.2 = load i32, ptr %.loc38_24.1.a, align 4, !dbg !12
// CHECK:STDOUT:   %.loc38_37.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   %.loc38_37.2 = load i32, ptr %.loc38_37.1.b, align 4, !dbg !13
// CHECK:STDOUT:   %.loc38_39.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   store i32 %.loc38_24.2, ptr %.loc38_39.2.a, align 4, !dbg !14
// CHECK:STDOUT:   %.loc38_39.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   store i32 %.loc38_37.2, ptr %.loc38_39.4.b, align 4, !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF_X.Main(ptr sret({ i32, i32 }) %return, ptr %a) !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.b1d03582403d0f2e(ptr %return, ptr %a), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.61ea2aba74ab3bf1:Copy.Main"() !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF_empty_tuple.Main() !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.f0b2e3523ff3fa17(), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.d07e2731f1087d49:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem0.loc59_12.1.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   %tuple.elem0.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   store i32 %tuple.elem0.loc59_12.1.tuple.elem.load, ptr %tuple.elem0.loc59_12.2.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !25
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem1.loc59_12.1.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   %tuple.elem1.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !25
// CHECK:STDOUT:   store i32 %tuple.elem1.loc59_12.1.tuple.elem.load, ptr %tuple.elem1.loc59_12.2.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF_two_tuple.Main(ptr sret({ i32, i32 }) %return, ptr %a) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.09ed05d6841b72fa(ptr %return, ptr %a), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.4af564822fd0c29a:Copy.Main"(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %self) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem.load = load ptr, ptr %tuple.elem0.loc69_17.tuple.elem, align 8, !dbg !31
// CHECK:STDOUT:   %tuple.elem0.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   call void @"_COp.d07e2731f1087d49:Copy.Main"(ptr %tuple.elem0.loc69_51.tuple.elem, ptr %tuple.elem0.loc69_17.tuple.elem.load), !dbg !31
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69_37.tuple.elem, align 8, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %tuple.elem1.loc69_51.tuple.elem, ptr %tuple.elem1.loc69_37.tuple.elem.load), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF_nested_tuple.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.6513c914c89dac0d(ptr %return, ptr %a), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.c51f5dcf0e081157(i32 %a) !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca i32, align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !39
// CHECK:STDOUT:   %.loc18_19 = call i32 @"_COp.Int.Core:Copy.Main"(i32 %a), !dbg !40
// CHECK:STDOUT:   store i32 %.loc18_19, ptr %v.var, align 4, !dbg !39
// CHECK:STDOUT:   %.loc19_15 = call i32 @"_COp.Int.Core:Copy.Main"(i32 %a), !dbg !41
// CHECK:STDOUT:   ret i32 %.loc19_15, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b1d03582403d0f2e(ptr sret({ i32, i32 }) %return, ptr %a) !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { i32, i32 }, align 8, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !44
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %v.var, ptr %a), !dbg !45
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %return, ptr %a), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f0b2e3523ff3fa17() !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca {}, align 8, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !49
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !50
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.09ed05d6841b72fa(ptr sret({ i32, i32 }) %return, ptr %a) !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { i32, i32 }, align 8, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !54
// CHECK:STDOUT:   call void @"_COp.d07e2731f1087d49:Copy.Main"(ptr %v.var, ptr %a), !dbg !55
// CHECK:STDOUT:   call void @"_COp.d07e2731f1087d49:Copy.Main"(ptr %return, ptr %a), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.6513c914c89dac0d(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { i32, i32 }, { i32, i32 } }, align 8, !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !59
// CHECK:STDOUT:   call void @"_COp.4af564822fd0c29a:Copy.Main"(ptr %v.var, ptr %a), !dbg !60
// CHECK:STDOUT:   call void @"_COp.4af564822fd0c29a:Copy.Main"(ptr %return, ptr %a), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "type_representation.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Copy.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 24, column: 5, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "F_i32", linkageName: "_CF_i32.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 10, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 29, column: 3, scope: !8)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.X.Main:Copy.Main", scope: null, file: !3, line: 37, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 38, column: 20, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 38, column: 33, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 38, column: 14, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 38, column: 7, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "F_X", linkageName: "_CF_X.Main", scope: null, file: !3, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DILocation(line: 44, column: 10, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 44, column: 3, scope: !16)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.61ea2aba74ab3bf1:Copy.Main", scope: null, file: !3, line: 48, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 49, column: 5, scope: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "F_empty_tuple", linkageName: "_CF_empty_tuple.Main", scope: null, file: !3, line: 53, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 54, column: 10, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 54, column: 3, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d07e2731f1087d49:Copy.Main", scope: null, file: !3, line: 58, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 59, column: 12, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 59, column: 5, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F_two_tuple", linkageName: "_CF_two_tuple.Main", scope: null, file: !3, line: 63, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 64, column: 10, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 64, column: 3, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4af564822fd0c29a:Copy.Main", scope: null, file: !3, line: 68, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 69, column: 13, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 69, column: 12, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 69, column: 33, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 69, column: 5, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "F_nested_tuple", linkageName: "_CF_nested_tuple.Main", scope: null, file: !3, line: 73, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !36 = !DILocation(line: 74, column: 10, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 74, column: 3, scope: !35)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c51f5dcf0e081157", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !39 = !DILocation(line: 18, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 14, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 10, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 3, scope: !38)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b1d03582403d0f2e", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 3, scope: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 18, column: 14, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 10, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 3, scope: !43)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f0b2e3523ff3fa17", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 18, column: 14, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 10, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 19, column: 3, scope: !48)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.09ed05d6841b72fa", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !54 = !DILocation(line: 18, column: 3, scope: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 18, column: 14, scope: !53)
// CHECK:STDOUT: !56 = !DILocation(line: 19, column: 10, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 19, column: 3, scope: !53)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.6513c914c89dac0d", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !59 = !DILocation(line: 18, column: 3, scope: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 18, column: 14, scope: !58)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 10, scope: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 3, scope: !58)
