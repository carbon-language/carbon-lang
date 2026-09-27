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

fn F(template T: type) {
  var unused w: T = 0;
}

fn G() {
  F(i32);
}

// --- initialize_primitive.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) {
  var unused w: i32 = x;
}

fn G() {
  F(0 as i32);
}

// --- initialize_from_tuple.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

fn F(template T: type) {
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

// --- initialize_from_nested_literal.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

//@dump-sem-ir-begin
fn F(template T: type) {
  var unused w: T = ({.a = Make(), .b = Make()}, Make());
}
//@dump-sem-ir-end

class Y {
  impl X as Core.ImplicitAs(Y) {
    fn Convert(self: X) -> Y;
  }
}

class Z {
  var a: X;
  var b: Y;
}

fn G() {
  F(({.a: X, .b: X}, X));

  F(({.a: Y, .b: Y}, Y));

  F((Z, Y));
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
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba() #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !92
// CHECK:STDOUT:   store i32 0, ptr %w.var, align 4, !dbg !92
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !94
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
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !74, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 5, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !74, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 5, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !74, line: 4, type: !77, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !92 = !DILocation(line: 5, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !DILocation(line: 4, column: 1, scope: !91)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 0), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %x) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !111
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !111
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!96}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !95 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !96 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !95, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !95, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !96, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !20)
// CHECK:STDOUT: !99 = !DILocation(line: 5, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !95, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !96, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !20)
// CHECK:STDOUT: !103 = !DILocation(line: 5, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !95, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !96)
// CHECK:STDOUT: !106 = !DILocation(line: 9, column: 3, scope: !105)
// CHECK:STDOUT: !107 = !DILocation(line: 8, column: 1, scope: !105)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !95, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !96, retainedNodes: !115)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !20)
// CHECK:STDOUT: !111 = !DILocation(line: 5, column: 3, scope: !108)
// CHECK:STDOUT: !114 = !DILocation(line: 4, column: 1, scope: !108)
// CHECK:STDOUT: !115 = !{!109}
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
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fb3d597a6b1c4ee1(), !dbg !132
// CHECK:STDOUT:   call void @_CF.Main.cd5dea75f888aed4(), !dbg !133
// CHECK:STDOUT:   call void @_CF.Main.be297cc7300419fe(), !dbg !134
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core"(ptr %self), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self) #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fb3d597a6b1c4ee1() #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !165
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !166
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !167
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !166
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !168
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !166
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !169
// CHECK:STDOUT:   call void @"_CSelfDestruct.92e03c783d6af646:core.Destroy.Core"(ptr %w.var), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.cd5dea75f888aed4() #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !173
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !173
// CHECK:STDOUT:   %.f75.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !174
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.f75.array.index), !dbg !175
// CHECK:STDOUT:   %.efc.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !174
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.efc.array.index), !dbg !176
// CHECK:STDOUT:   %.d3b.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !174
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.d3b.array.index), !dbg !177
// CHECK:STDOUT:   call void @"_CSelfDestruct.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %w.var), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.be297cc7300419fe() #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !181
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !181
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !181
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !181
// CHECK:STDOUT:   %tuple.9aa = alloca { ptr, ptr, ptr }, align 8, !dbg !185
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !181
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !181
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !182
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !181
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !181
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !184
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.9aa), !dbg !185
// CHECK:STDOUT:   %tuple.9aa1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 0, !dbg !185
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.9aa1, align 8, !dbg !185
// CHECK:STDOUT:   %tuple.9aa2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 1, !dbg !185
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.9aa2, align 8, !dbg !185
// CHECK:STDOUT:   %tuple.9aa3 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 2, !dbg !185
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.9aa3, align 8, !dbg !185
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.9aa), !dbg !181
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %w.var), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !187
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!117}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !116 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !117 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !116, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !120 = !{!23, !23}
// CHECK:STDOUT: !121 = !DISubroutineType(types: !120)
// CHECK:STDOUT: !122 = !{null, !23}
// CHECK:STDOUT: !123 = !DISubroutineType(types: !122)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !116, line: 14, type: !121, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !130)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !23)
// CHECK:STDOUT: !129 = !DILocation(line: 14, column: 45, scope: !124)
// CHECK:STDOUT: !130 = !{!125}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !116, line: 17, type: !77, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !132 = !DILocation(line: 18, column: 3, scope: !131)
// CHECK:STDOUT: !133 = !DILocation(line: 20, column: 3, scope: !131)
// CHECK:STDOUT: !134 = !DILocation(line: 22, column: 3, scope: !131)
// CHECK:STDOUT: !135 = !DILocation(line: 17, column: 1, scope: !131)
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !139)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !23)
// CHECK:STDOUT: !138 = !DILocation(line: 8, column: 3, scope: !136)
// CHECK:STDOUT: !139 = !{!137}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !143)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !23)
// CHECK:STDOUT: !142 = !DILocation(line: 8, column: 3, scope: !140)
// CHECK:STDOUT: !143 = !{!141}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !147)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !23)
// CHECK:STDOUT: !146 = !DILocation(line: 8, column: 3, scope: !144)
// CHECK:STDOUT: !147 = !{!145}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.92e03c783d6af646:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !151)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !23)
// CHECK:STDOUT: !150 = !DILocation(line: 8, column: 3, scope: !148)
// CHECK:STDOUT: !151 = !{!149}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4c02ca82d7b9fa3c:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !155)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !23)
// CHECK:STDOUT: !154 = !DILocation(line: 8, column: 3, scope: !152)
// CHECK:STDOUT: !155 = !{!153}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4c02ca82d7b9fa3c:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !159)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !23)
// CHECK:STDOUT: !158 = !DILocation(line: 8, column: 3, scope: !156)
// CHECK:STDOUT: !159 = !{!157}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !116, line: 8, type: !123, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !23)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 3, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fb3d597a6b1c4ee1", scope: null, file: !116, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !165 = !DILocation(line: 8, column: 3, scope: !164)
// CHECK:STDOUT: !166 = !DILocation(line: 8, column: 21, scope: !164)
// CHECK:STDOUT: !167 = !DILocation(line: 8, column: 22, scope: !164)
// CHECK:STDOUT: !168 = !DILocation(line: 8, column: 30, scope: !164)
// CHECK:STDOUT: !169 = !DILocation(line: 8, column: 38, scope: !164)
// CHECK:STDOUT: !171 = !DILocation(line: 7, column: 1, scope: !164)
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.cd5dea75f888aed4", scope: null, file: !116, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !173 = !DILocation(line: 8, column: 3, scope: !172)
// CHECK:STDOUT: !174 = !DILocation(line: 8, column: 21, scope: !172)
// CHECK:STDOUT: !175 = !DILocation(line: 8, column: 22, scope: !172)
// CHECK:STDOUT: !176 = !DILocation(line: 8, column: 30, scope: !172)
// CHECK:STDOUT: !177 = !DILocation(line: 8, column: 38, scope: !172)
// CHECK:STDOUT: !179 = !DILocation(line: 7, column: 1, scope: !172)
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.be297cc7300419fe", scope: null, file: !116, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !181 = !DILocation(line: 8, column: 3, scope: !180)
// CHECK:STDOUT: !182 = !DILocation(line: 8, column: 22, scope: !180)
// CHECK:STDOUT: !183 = !DILocation(line: 8, column: 30, scope: !180)
// CHECK:STDOUT: !184 = !DILocation(line: 8, column: 38, scope: !180)
// CHECK:STDOUT: !185 = !DILocation(line: 8, column: 21, scope: !180)
// CHECK:STDOUT: !187 = !DILocation(line: 7, column: 1, scope: !180)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_nested_literal.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_nested_literal.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret({}))
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.75a9ebb2041c173b(), !dbg !191
// CHECK:STDOUT:   call void @_CF.Main.f5d9b693014ce43e(), !dbg !192
// CHECK:STDOUT:   call void @_CF.Main.84270bb113cb754b(), !dbg !193
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self), !dbg !209
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core"(ptr %self), !dbg !217
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self), !dbg !229
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core"(ptr %self) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2f43e328c637647f:core.Destroy.Core"(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core"(ptr %self), !dbg !237
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core"(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2970be6ba4c2615c:core.Destroy.Core"(ptr %self) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core"(ptr %self), !dbg !245
// CHECK:STDOUT:   ret void, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #0 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self), !dbg !253
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core"(ptr %self) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core"(ptr %self) #0 !dbg !259 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core"(ptr %self), !dbg !261
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.75a9ebb2041c173b() #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !264
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !264
// CHECK:STDOUT:   %tuple.elem0.a28.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !266
// CHECK:STDOUT:   %.d62.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.a28.tuple.elem, i32 0, i32 0, !dbg !265
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.d62.a), !dbg !267
// CHECK:STDOUT:   %.162.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.a28.tuple.elem, i32 0, i32 1, !dbg !265
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.162.b), !dbg !268
// CHECK:STDOUT:   %tuple.elem1.79b.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !266
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.79b.tuple.elem), !dbg !269
// CHECK:STDOUT:   call void @"_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core"(ptr %w.var), !dbg !264
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f5d9b693014ce43e() #0 !dbg !272 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !273
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !273
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !273
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !273
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !273
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !273
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !274
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !273
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !275
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !273
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !277
// CHECK:STDOUT:   %tuple.elem0.25a.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !278
// CHECK:STDOUT:   %.e68.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.25a.tuple.elem, i32 0, i32 0, !dbg !276
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.e68.a, ptr %.7a56fd.1.temp), !dbg !276
// CHECK:STDOUT:   %.9f2.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.25a.tuple.elem, i32 0, i32 1, !dbg !276
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.9f2.b, ptr %.7a56fd.2.temp), !dbg !276
// CHECK:STDOUT:   %tuple.elem1.0e1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !278
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.0e1.tuple.elem, ptr %.7a56fd.3.temp), !dbg !278
// CHECK:STDOUT:   call void @"_CSelfDestruct.2f43e328c637647f:core.Destroy.Core"(ptr %w.var), !dbg !273
// CHECK:STDOUT:   ret void, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84270bb113cb754b() #0 !dbg !281 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !282
// CHECK:STDOUT:   %.7a56fd.4.temp = alloca {}, align 1, !dbg !282
// CHECK:STDOUT:   %.7a56fd.5.temp = alloca {}, align 1, !dbg !282
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !282
// CHECK:STDOUT:   %tuple.elem0.d0d.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !284
// CHECK:STDOUT:   %.b76.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.d0d.tuple.elem, i32 0, i32 0, !dbg !283
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b76.a), !dbg !285
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.4.temp), !dbg !282
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.4.temp), !dbg !286
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.5.temp), !dbg !282
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.5.temp), !dbg !287
// CHECK:STDOUT:   %.25e.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.d0d.tuple.elem, i32 0, i32 1, !dbg !283
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.25e.b, ptr %.7a56fd.4.temp), !dbg !283
// CHECK:STDOUT:   %tuple.elem1.c76.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !284
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.c76.tuple.elem, ptr %.7a56fd.5.temp), !dbg !284
// CHECK:STDOUT:   call void @"_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core"(ptr %w.var), !dbg !282
// CHECK:STDOUT:   ret void, !dbg !289
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!189}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !122 = !{null, !23}
// CHECK:STDOUT: !123 = !DISubroutineType(types: !122)
// CHECK:STDOUT: !188 = !DIFile(filename: "initialize_from_nested_literal.carbon", directory: "")
// CHECK:STDOUT: !189 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !188, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !188, line: 24, type: !77, spFlags: DISPFlagDefinition, unit: !189)
// CHECK:STDOUT: !191 = !DILocation(line: 25, column: 3, scope: !190)
// CHECK:STDOUT: !192 = !DILocation(line: 27, column: 3, scope: !190)
// CHECK:STDOUT: !193 = !DILocation(line: 29, column: 3, scope: !190)
// CHECK:STDOUT: !194 = !DILocation(line: 24, column: 1, scope: !190)
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !198)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !23)
// CHECK:STDOUT: !197 = !DILocation(line: 9, column: 3, scope: !195)
// CHECK:STDOUT: !198 = !{!196}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !202)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !23)
// CHECK:STDOUT: !201 = !DILocation(line: 9, column: 3, scope: !199)
// CHECK:STDOUT: !202 = !{!200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !23)
// CHECK:STDOUT: !205 = !DILocation(line: 9, column: 3, scope: !203)
// CHECK:STDOUT: !206 = !{!204}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !210)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !23)
// CHECK:STDOUT: !209 = !DILocation(line: 9, column: 3, scope: !207)
// CHECK:STDOUT: !210 = !{!208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !23)
// CHECK:STDOUT: !213 = !DILocation(line: 9, column: 3, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !23)
// CHECK:STDOUT: !217 = !DILocation(line: 9, column: 3, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !23)
// CHECK:STDOUT: !221 = !DILocation(line: 9, column: 3, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !226)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !23)
// CHECK:STDOUT: !225 = !DILocation(line: 9, column: 3, scope: !223)
// CHECK:STDOUT: !226 = !{!224}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !23)
// CHECK:STDOUT: !229 = !DILocation(line: 9, column: 3, scope: !227)
// CHECK:STDOUT: !230 = !{!228}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !234)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !23)
// CHECK:STDOUT: !233 = !DILocation(line: 9, column: 3, scope: !231)
// CHECK:STDOUT: !234 = !{!232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2f43e328c637647f:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !238)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !23)
// CHECK:STDOUT: !237 = !DILocation(line: 9, column: 3, scope: !235)
// CHECK:STDOUT: !238 = !{!236}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !23)
// CHECK:STDOUT: !241 = !DILocation(line: 9, column: 3, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2970be6ba4c2615c:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !246)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !23)
// CHECK:STDOUT: !245 = !DILocation(line: 9, column: 3, scope: !243)
// CHECK:STDOUT: !246 = !{!244}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !250)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !23)
// CHECK:STDOUT: !249 = !DILocation(line: 9, column: 3, scope: !247)
// CHECK:STDOUT: !250 = !{!248}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !254)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !23)
// CHECK:STDOUT: !253 = !DILocation(line: 9, column: 3, scope: !251)
// CHECK:STDOUT: !254 = !{!252}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !258)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !23)
// CHECK:STDOUT: !257 = !DILocation(line: 9, column: 3, scope: !255)
// CHECK:STDOUT: !258 = !{!256}
// CHECK:STDOUT: !259 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core", scope: null, file: !188, line: 9, type: !123, spFlags: DISPFlagDefinition, unit: !189, retainedNodes: !262)
// CHECK:STDOUT: !260 = !DILocalVariable(arg: 1, scope: !259, type: !23)
// CHECK:STDOUT: !261 = !DILocation(line: 9, column: 3, scope: !259)
// CHECK:STDOUT: !262 = !{!260}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.75a9ebb2041c173b", scope: null, file: !188, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !189)
// CHECK:STDOUT: !264 = !DILocation(line: 9, column: 3, scope: !263)
// CHECK:STDOUT: !265 = !DILocation(line: 9, column: 22, scope: !263)
// CHECK:STDOUT: !266 = !DILocation(line: 9, column: 21, scope: !263)
// CHECK:STDOUT: !267 = !DILocation(line: 9, column: 28, scope: !263)
// CHECK:STDOUT: !268 = !DILocation(line: 9, column: 41, scope: !263)
// CHECK:STDOUT: !269 = !DILocation(line: 9, column: 50, scope: !263)
// CHECK:STDOUT: !271 = !DILocation(line: 8, column: 1, scope: !263)
// CHECK:STDOUT: !272 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f5d9b693014ce43e", scope: null, file: !188, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !189)
// CHECK:STDOUT: !273 = !DILocation(line: 9, column: 3, scope: !272)
// CHECK:STDOUT: !274 = !DILocation(line: 9, column: 28, scope: !272)
// CHECK:STDOUT: !275 = !DILocation(line: 9, column: 41, scope: !272)
// CHECK:STDOUT: !276 = !DILocation(line: 9, column: 22, scope: !272)
// CHECK:STDOUT: !277 = !DILocation(line: 9, column: 50, scope: !272)
// CHECK:STDOUT: !278 = !DILocation(line: 9, column: 21, scope: !272)
// CHECK:STDOUT: !280 = !DILocation(line: 8, column: 1, scope: !272)
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84270bb113cb754b", scope: null, file: !188, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !189)
// CHECK:STDOUT: !282 = !DILocation(line: 9, column: 3, scope: !281)
// CHECK:STDOUT: !283 = !DILocation(line: 9, column: 22, scope: !281)
// CHECK:STDOUT: !284 = !DILocation(line: 9, column: 21, scope: !281)
// CHECK:STDOUT: !285 = !DILocation(line: 9, column: 28, scope: !281)
// CHECK:STDOUT: !286 = !DILocation(line: 9, column: 41, scope: !281)
// CHECK:STDOUT: !287 = !DILocation(line: 9, column: 50, scope: !281)
// CHECK:STDOUT: !289 = !DILocation(line: 8, column: 1, scope: !281)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !292 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.84588f41d61dafba(i32 %n), !dbg !296
// CHECK:STDOUT:   ret void, !dbg !299
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !301 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !306
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !306
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !307
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !309 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !313
// CHECK:STDOUT:   ret void, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.84588f41d61dafba(i32 %x) #0 !dbg !318 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !326 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !331
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!291}
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
// CHECK:STDOUT: !122 = !{null, !23}
// CHECK:STDOUT: !123 = !DISubroutineType(types: !122)
// CHECK:STDOUT: !290 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !291 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !290, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !292 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !290, line: 8, type: !79, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !300)
// CHECK:STDOUT: !293 = !DILocalVariable(arg: 1, scope: !292, type: !20)
// CHECK:STDOUT: !296 = !DILocation(line: 9, column: 16, scope: !292)
// CHECK:STDOUT: !299 = !DILocation(line: 8, column: 1, scope: !292)
// CHECK:STDOUT: !300 = !{!293}
// CHECK:STDOUT: !301 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !290, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !308)
// CHECK:STDOUT: !302 = !DILocalVariable(arg: 1, scope: !301, type: !23)
// CHECK:STDOUT: !306 = !DILocation(line: 15, column: 38, scope: !301)
// CHECK:STDOUT: !307 = !DILocation(line: 15, column: 31, scope: !301)
// CHECK:STDOUT: !308 = !{!302}
// CHECK:STDOUT: !309 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !290, line: 19, type: !123, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !317)
// CHECK:STDOUT: !310 = !DILocalVariable(arg: 1, scope: !309, type: !23)
// CHECK:STDOUT: !313 = !DILocation(line: 20, column: 16, scope: !309)
// CHECK:STDOUT: !316 = !DILocation(line: 19, column: 1, scope: !309)
// CHECK:STDOUT: !317 = !{!310}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.84588f41d61dafba", scope: null, file: !290, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !325)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !20)
// CHECK:STDOUT: !324 = !DILocation(line: 5, column: 3, scope: !318)
// CHECK:STDOUT: !325 = !{!319}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !290, line: 4, type: !25, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !333)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !23)
// CHECK:STDOUT: !331 = !DILocation(line: 5, column: 10, scope: !326)
// CHECK:STDOUT: !332 = !DILocation(line: 5, column: 3, scope: !326)
// CHECK:STDOUT: !333 = !{!327}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !336 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.8122a03446648385(i32 %n), !dbg !340
// CHECK:STDOUT:   ret void, !dbg !341
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !343 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !348
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !348
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !351 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.3280a9d16838b780(ptr %c), !dbg !355
// CHECK:STDOUT:   ret void, !dbg !356
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.8122a03446648385(i32 %x) #0 !dbg !358 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.3280a9d16838b780(ptr %x) #0 !dbg !366 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !369
// CHECK:STDOUT:   ret void, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!335}
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
// CHECK:STDOUT: !122 = !{null, !23}
// CHECK:STDOUT: !123 = !DISubroutineType(types: !122)
// CHECK:STDOUT: !334 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !335 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !334, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !336 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !334, line: 8, type: !79, spFlags: DISPFlagDefinition, unit: !335, retainedNodes: !342)
// CHECK:STDOUT: !337 = !DILocalVariable(arg: 1, scope: !336, type: !20)
// CHECK:STDOUT: !340 = !DILocation(line: 9, column: 3, scope: !336)
// CHECK:STDOUT: !341 = !DILocation(line: 8, column: 1, scope: !336)
// CHECK:STDOUT: !342 = !{!337}
// CHECK:STDOUT: !343 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !334, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !335, retainedNodes: !350)
// CHECK:STDOUT: !344 = !DILocalVariable(arg: 1, scope: !343, type: !23)
// CHECK:STDOUT: !348 = !DILocation(line: 15, column: 38, scope: !343)
// CHECK:STDOUT: !349 = !DILocation(line: 15, column: 31, scope: !343)
// CHECK:STDOUT: !350 = !{!344}
// CHECK:STDOUT: !351 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !334, line: 19, type: !123, spFlags: DISPFlagDefinition, unit: !335, retainedNodes: !357)
// CHECK:STDOUT: !352 = !DILocalVariable(arg: 1, scope: !351, type: !23)
// CHECK:STDOUT: !355 = !DILocation(line: 20, column: 3, scope: !351)
// CHECK:STDOUT: !356 = !DILocation(line: 19, column: 1, scope: !351)
// CHECK:STDOUT: !357 = !{!352}
// CHECK:STDOUT: !358 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.8122a03446648385", scope: null, file: !334, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !335, retainedNodes: !365)
// CHECK:STDOUT: !359 = !DILocalVariable(arg: 1, scope: !358, type: !20)
// CHECK:STDOUT: !364 = !DILocation(line: 4, column: 1, scope: !358)
// CHECK:STDOUT: !365 = !{!359}
// CHECK:STDOUT: !366 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.3280a9d16838b780", scope: null, file: !334, line: 4, type: !123, spFlags: DISPFlagDefinition, unit: !335, retainedNodes: !373)
// CHECK:STDOUT: !367 = !DILocalVariable(arg: 1, scope: !366, type: !23)
// CHECK:STDOUT: !369 = !DILocation(line: 5, column: 14, scope: !366)
// CHECK:STDOUT: !372 = !DILocation(line: 4, column: 1, scope: !366)
// CHECK:STDOUT: !373 = !{!367}
// CHECK:STDOUT:
