// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/convert.carbon

// --- convert.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> i32 {
  let n: i32 = x;
  return n;
}

fn Test1(n: i32) -> i32 {
  return F(n);
}

class C {
  var n: i32;
  impl as Core.ImplicitAs(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) -> i32 {
  return F(c);
}

// --- initialize_dependent_target_from_literal.carbon

library "[[@TEST_NAME]]";

fn F(template T: Core.Destroy) {
  var unused w: T = 0;
}

fn G() {
  F(i32);
}

// --- initialize_primitive.carbon

library "[[@TEST_NAME]]";

fn F[template T: Core.Destroy](x: T) {
  var unused w: i32 = x;
}

fn G() {
  F(0 as i32);
}

// --- initialize_from_tuple.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

fn F(template T: Core.Destroy) {
  var unused w: T = (Make(), Make(), Make());
}

class Y {}

impl (X, X, X) as Core.ImplicitAs(Y) {
  fn Convert(unused self: (X, X, X)) -> Y { return {}; }
}

fn G() {
  F((X, X, X));

  F(array(X, 3));

  F(Y);
}

// --- as.carbon

library "[[@TEST_NAME]]";

fn AsI32[template T: type](x: T) -> i32 {
  return x as i32;
}

fn Test1(n: i32) {
  let _: i32 = AsI32(n);
}

class C {
  var n: i32;
  impl as Core.As(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) {
  let _: i32 = AsI32(c);
}

// --- as_type.carbon

library "[[@TEST_NAME]]";

fn AsType[template T: type](x: T, template U: type) {
  let _: U = x as U;
}

fn Test1(n: i32) {
  AsType(n, i32);
}

class C {
  var n: i32;
  impl as Core.As(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) {
  AsType(c, i32);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest1.Main(i32 %n) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.84588f41d61dafba(i32 %n), !dbg !32
// CHECK:STDOUT:   ret i32 %F.call, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:ImplicitAs.84588f41d61dafba.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc16_42.2 = load i32, ptr %.loc16_42.1.n, align 4, !dbg !40
// CHECK:STDOUT:   ret i32 %.loc16_42.2, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest2.Main(ptr %c) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.9522ff908e97e267(ptr %c), !dbg !49
// CHECK:STDOUT:   ret i32 %F.call, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.84588f41d61dafba(i32 %x) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9522ff908e97e267(ptr %x) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.C.Main:ImplicitAs.84588f41d61dafba.Core"(ptr %x), !dbg !68
// CHECK:STDOUT:   ret i32 %C.as.ImplicitAs.impl.Convert.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !18, line: 9, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !34)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !20)
// CHECK:STDOUT: !32 = !DILocation(line: 10, column: 10, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 3, scope: !26)
// CHECK:STDOUT: !34 = !{!27}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.84588f41d61dafba.Core", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 38, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 31, scope: !35)
// CHECK:STDOUT: !42 = !{!36}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !49 = !DILocation(line: 21, column: 10, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 21, column: 3, scope: !43)
// CHECK:STDOUT: !51 = !{!44}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !18, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !62)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 6, column: 3, scope: !52)
// CHECK:STDOUT: !62 = !{!53}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !18, line: 4, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !73)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !23)
// CHECK:STDOUT: !68 = !DILocation(line: 5, column: 16, scope: !63)
// CHECK:STDOUT: !72 = !DILocation(line: 6, column: 3, scope: !63)
// CHECK:STDOUT: !73 = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_dependent_target_from_literal.carbon'
// CHECK:STDOUT: source_filename = "initialize_dependent_target_from_literal.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.739ab18a7e808791(), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.739ab18a7e808791() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !88
// CHECK:STDOUT:   store i32 0, ptr %w.var, align 4, !dbg !88
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!75}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !74 = !DIFile(filename: "initialize_dependent_target_from_literal.carbon", directory: "")
// CHECK:STDOUT: !75 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !74, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !74, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !81 = !DILocation(line: 9, column: 3, scope: !80)
// CHECK:STDOUT: !82 = !DILocation(line: 8, column: 1, scope: !80)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !74, line: 9, type: !79, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 9, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.739ab18a7e808791", scope: null, file: !74, line: 4, type: !77, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !88 = !DILocation(line: 5, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !DILocation(line: 4, column: 1, scope: !87)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.739ab18a7e808791(i32 0), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.739ab18a7e808791(i32 %x) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !103
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !103
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !103
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !91 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !91, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !20)
// CHECK:STDOUT: !95 = !DILocation(line: 5, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !91, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 9, column: 3, scope: !97)
// CHECK:STDOUT: !99 = !DILocation(line: 8, column: 1, scope: !97)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.739ab18a7e808791", scope: null, file: !91, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !107)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !20)
// CHECK:STDOUT: !103 = !DILocation(line: 5, column: 3, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 4, column: 1, scope: !100)
// CHECK:STDOUT: !107 = !{!101}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_tuple.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Y.val.loc14_54 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret({}))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.1671a61a643df346(), !dbg !124
// CHECK:STDOUT:   call void @_CF.Main.63c9fc9ae647f1eb(), !dbg !125
// CHECK:STDOUT:   call void @_CF.Main.019580a566b33d38(), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.1671a61a643df346() #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !145
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !146
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !147
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !146
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !148
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !146
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !149
// CHECK:STDOUT:   call void @"_COp.92e03c783d6af646:core.Destroy.Core"(ptr %w.var), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.63c9fc9ae647f1eb() #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !153
// CHECK:STDOUT:   %.a3a.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !154
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.a3a.array.index), !dbg !155
// CHECK:STDOUT:   %.87b.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !154
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.87b.array.index), !dbg !156
// CHECK:STDOUT:   %.b3d.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !154
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b3d.array.index), !dbg !157
// CHECK:STDOUT:   call void @"_COp.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %w.var), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.019580a566b33d38() #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !161
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !161
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !161
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !161
// CHECK:STDOUT:   %tuple.675 = alloca { ptr, ptr, ptr }, align 8, !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !161
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !161
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !161
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !164
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.675), !dbg !165
// CHECK:STDOUT:   %tuple.6751 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.675, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.6751, align 8, !dbg !165
// CHECK:STDOUT:   %tuple.6752 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.675, i32 0, i32 1, !dbg !165
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.6752, align 8, !dbg !165
// CHECK:STDOUT:   %tuple.6753 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.675, i32 0, i32 2, !dbg !165
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.6753, align 8, !dbg !165
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.675), !dbg !161
// CHECK:STDOUT:   call void @"_COp.12212b84320f2fd6:core.Destroy.Core"(ptr %w.var), !dbg !161
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!109}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !108 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !109 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !108, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = !{!23, !23}
// CHECK:STDOUT: !113 = !DISubroutineType(types: !112)
// CHECK:STDOUT: !114 = !{null, !23}
// CHECK:STDOUT: !115 = !DISubroutineType(types: !114)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !108, line: 14, type: !113, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !23)
// CHECK:STDOUT: !121 = !DILocation(line: 14, column: 45, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !108, line: 17, type: !77, spFlags: DISPFlagDefinition, unit: !109)
// CHECK:STDOUT: !124 = !DILocation(line: 18, column: 3, scope: !123)
// CHECK:STDOUT: !125 = !DILocation(line: 20, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !DILocation(line: 22, column: 3, scope: !123)
// CHECK:STDOUT: !127 = !DILocation(line: 17, column: 1, scope: !123)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !108, line: 18, type: !115, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !23)
// CHECK:STDOUT: !130 = !DILocation(line: 18, column: 3, scope: !128)
// CHECK:STDOUT: !131 = !{!129}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Op", linkageName: "_COp.92e03c783d6af646:core.Destroy.Core", scope: null, file: !108, line: 18, type: !115, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !135)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !23)
// CHECK:STDOUT: !134 = !DILocation(line: 18, column: 3, scope: !132)
// CHECK:STDOUT: !135 = !{!133}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4c02ca82d7b9fa3c:core.Destroy.Core", scope: null, file: !108, line: 20, type: !115, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !139)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !23)
// CHECK:STDOUT: !138 = !DILocation(line: 20, column: 3, scope: !136)
// CHECK:STDOUT: !139 = !{!137}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "Op", linkageName: "_COp.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !108, line: 22, type: !115, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !143)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !23)
// CHECK:STDOUT: !142 = !DILocation(line: 22, column: 3, scope: !140)
// CHECK:STDOUT: !143 = !{!141}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.1671a61a643df346", scope: null, file: !108, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !109)
// CHECK:STDOUT: !145 = !DILocation(line: 8, column: 3, scope: !144)
// CHECK:STDOUT: !146 = !DILocation(line: 8, column: 21, scope: !144)
// CHECK:STDOUT: !147 = !DILocation(line: 8, column: 22, scope: !144)
// CHECK:STDOUT: !148 = !DILocation(line: 8, column: 30, scope: !144)
// CHECK:STDOUT: !149 = !DILocation(line: 8, column: 38, scope: !144)
// CHECK:STDOUT: !151 = !DILocation(line: 7, column: 1, scope: !144)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.63c9fc9ae647f1eb", scope: null, file: !108, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !109)
// CHECK:STDOUT: !153 = !DILocation(line: 8, column: 3, scope: !152)
// CHECK:STDOUT: !154 = !DILocation(line: 8, column: 21, scope: !152)
// CHECK:STDOUT: !155 = !DILocation(line: 8, column: 22, scope: !152)
// CHECK:STDOUT: !156 = !DILocation(line: 8, column: 30, scope: !152)
// CHECK:STDOUT: !157 = !DILocation(line: 8, column: 38, scope: !152)
// CHECK:STDOUT: !159 = !DILocation(line: 7, column: 1, scope: !152)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.019580a566b33d38", scope: null, file: !108, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !109)
// CHECK:STDOUT: !161 = !DILocation(line: 8, column: 3, scope: !160)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 22, scope: !160)
// CHECK:STDOUT: !163 = !DILocation(line: 8, column: 30, scope: !160)
// CHECK:STDOUT: !164 = !DILocation(line: 8, column: 38, scope: !160)
// CHECK:STDOUT: !165 = !DILocation(line: 8, column: 21, scope: !160)
// CHECK:STDOUT: !167 = !DILocation(line: 7, column: 1, scope: !160)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.84588f41d61dafba(i32 %n), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !184
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !184
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !187 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !191
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.84588f41d61dafba(i32 %x) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !204 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !209
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!169}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !114 = !{null, !23}
// CHECK:STDOUT: !115 = !DISubroutineType(types: !114)
// CHECK:STDOUT: !168 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !169 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !168, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !168, line: 8, type: !79, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !178)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !20)
// CHECK:STDOUT: !174 = !DILocation(line: 9, column: 16, scope: !170)
// CHECK:STDOUT: !177 = !DILocation(line: 8, column: 1, scope: !170)
// CHECK:STDOUT: !178 = !{!171}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !168, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !186)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !23)
// CHECK:STDOUT: !184 = !DILocation(line: 15, column: 38, scope: !179)
// CHECK:STDOUT: !185 = !DILocation(line: 15, column: 31, scope: !179)
// CHECK:STDOUT: !186 = !{!180}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !168, line: 19, type: !115, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !195)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !23)
// CHECK:STDOUT: !191 = !DILocation(line: 20, column: 16, scope: !187)
// CHECK:STDOUT: !194 = !DILocation(line: 19, column: 1, scope: !187)
// CHECK:STDOUT: !195 = !{!188}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.84588f41d61dafba", scope: null, file: !168, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !203)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !20)
// CHECK:STDOUT: !202 = !DILocation(line: 5, column: 3, scope: !196)
// CHECK:STDOUT: !203 = !{!197}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !168, line: 4, type: !25, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !211)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !23)
// CHECK:STDOUT: !209 = !DILocation(line: 5, column: 10, scope: !204)
// CHECK:STDOUT: !210 = !DILocation(line: 5, column: 3, scope: !204)
// CHECK:STDOUT: !211 = !{!205}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.8122a03446648385(i32 %n), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !221 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !226
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !226
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !229 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.3280a9d16838b780(ptr %c), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.8122a03446648385(i32 %x) #0 !dbg !236 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !242
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.3280a9d16838b780(ptr %x) #0 !dbg !244 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !247
// CHECK:STDOUT:   ret void, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!213}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !114 = !{null, !23}
// CHECK:STDOUT: !115 = !DISubroutineType(types: !114)
// CHECK:STDOUT: !212 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !213 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !212, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !212, line: 8, type: !79, spFlags: DISPFlagDefinition, unit: !213, retainedNodes: !220)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !20)
// CHECK:STDOUT: !218 = !DILocation(line: 9, column: 3, scope: !214)
// CHECK:STDOUT: !219 = !DILocation(line: 8, column: 1, scope: !214)
// CHECK:STDOUT: !220 = !{!215}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !212, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !213, retainedNodes: !228)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !23)
// CHECK:STDOUT: !226 = !DILocation(line: 15, column: 38, scope: !221)
// CHECK:STDOUT: !227 = !DILocation(line: 15, column: 31, scope: !221)
// CHECK:STDOUT: !228 = !{!222}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !212, line: 19, type: !115, spFlags: DISPFlagDefinition, unit: !213, retainedNodes: !235)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !23)
// CHECK:STDOUT: !233 = !DILocation(line: 20, column: 3, scope: !229)
// CHECK:STDOUT: !234 = !DILocation(line: 19, column: 1, scope: !229)
// CHECK:STDOUT: !235 = !{!230}
// CHECK:STDOUT: !236 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.8122a03446648385", scope: null, file: !212, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !213, retainedNodes: !243)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !236, type: !20)
// CHECK:STDOUT: !242 = !DILocation(line: 4, column: 1, scope: !236)
// CHECK:STDOUT: !243 = !{!237}
// CHECK:STDOUT: !244 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.3280a9d16838b780", scope: null, file: !212, line: 4, type: !115, spFlags: DISPFlagDefinition, unit: !213, retainedNodes: !251)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 1, scope: !244, type: !23)
// CHECK:STDOUT: !247 = !DILocation(line: 5, column: 14, scope: !244)
// CHECK:STDOUT: !250 = !DILocation(line: 4, column: 1, scope: !244)
// CHECK:STDOUT: !251 = !{!245}
// CHECK:STDOUT:
