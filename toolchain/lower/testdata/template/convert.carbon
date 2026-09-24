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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba() #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !96
// CHECK:STDOUT:   store i32 0, ptr %w.var, align 4, !dbg !96
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !98
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
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !74, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 5, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !74, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 5, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !74, line: 4, type: !77, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !96 = !DILocation(line: 5, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !DILocation(line: 4, column: 1, scope: !95)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 0), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %x) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !119
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !119
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !119
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!100}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !99 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !100 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !99, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !99, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !20)
// CHECK:STDOUT: !103 = !DILocation(line: 5, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !99, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !20)
// CHECK:STDOUT: !107 = !DILocation(line: 5, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !99, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !20)
// CHECK:STDOUT: !111 = !DILocation(line: 5, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !99, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !100)
// CHECK:STDOUT: !114 = !DILocation(line: 9, column: 3, scope: !113)
// CHECK:STDOUT: !115 = !DILocation(line: 8, column: 1, scope: !113)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !99, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !123)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !20)
// CHECK:STDOUT: !119 = !DILocation(line: 5, column: 3, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 4, column: 1, scope: !116)
// CHECK:STDOUT: !123 = !{!117}
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
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fb3d597a6b1c4ee1(), !dbg !140
// CHECK:STDOUT:   call void @_CF.Main.cd5dea75f888aed4(), !dbg !141
// CHECK:STDOUT:   call void @_CF.Main.be297cc7300419fe(), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core"(ptr %self), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %self), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.12212b84320f2fd6:core.Destroy.Core"(ptr %self), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fb3d597a6b1c4ee1() #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !181
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !181
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !182
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !183
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !182
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !184
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !182
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !185
// CHECK:STDOUT:   call void @"_CSelfDestruct.92e03c783d6af646:core.Destroy.Core"(ptr %w.var), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.cd5dea75f888aed4() #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !189
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !189
// CHECK:STDOUT:   %.f75.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !190
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.f75.array.index), !dbg !191
// CHECK:STDOUT:   %.efc.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !190
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.efc.array.index), !dbg !192
// CHECK:STDOUT:   %.d3b.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !190
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.d3b.array.index), !dbg !193
// CHECK:STDOUT:   call void @"_CSelfDestruct.4c02ca82d7b9fa3c:core.Destroy.Core"(ptr %w.var), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.be297cc7300419fe() #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !197
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !197
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !197
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !197
// CHECK:STDOUT:   %tuple.9aa = alloca { ptr, ptr, ptr }, align 8, !dbg !201
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !197
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !197
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !198
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !197
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !199
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !197
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !200
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.9aa), !dbg !201
// CHECK:STDOUT:   %tuple.9aa1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 0, !dbg !201
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.9aa1, align 8, !dbg !201
// CHECK:STDOUT:   %tuple.9aa2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 1, !dbg !201
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.9aa2, align 8, !dbg !201
// CHECK:STDOUT:   %tuple.9aa3 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 2, !dbg !201
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.9aa3, align 8, !dbg !201
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.9aa), !dbg !197
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %w.var), !dbg !197
// CHECK:STDOUT:   ret void, !dbg !203
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!125}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !124 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !125 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !124, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !128 = !{!23, !23}
// CHECK:STDOUT: !129 = !DISubroutineType(types: !128)
// CHECK:STDOUT: !130 = !{null, !23}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !124, line: 14, type: !129, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !23)
// CHECK:STDOUT: !137 = !DILocation(line: 14, column: 45, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !124, line: 17, type: !77, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !140 = !DILocation(line: 18, column: 3, scope: !139)
// CHECK:STDOUT: !141 = !DILocation(line: 20, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !DILocation(line: 22, column: 3, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 17, column: 1, scope: !139)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !147)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !23)
// CHECK:STDOUT: !146 = !DILocation(line: 8, column: 3, scope: !144)
// CHECK:STDOUT: !147 = !{!145}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !151)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !23)
// CHECK:STDOUT: !150 = !DILocation(line: 8, column: 3, scope: !148)
// CHECK:STDOUT: !151 = !{!149}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !155)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !23)
// CHECK:STDOUT: !154 = !DILocation(line: 8, column: 3, scope: !152)
// CHECK:STDOUT: !155 = !{!153}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !159)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !23)
// CHECK:STDOUT: !158 = !DILocation(line: 8, column: 3, scope: !156)
// CHECK:STDOUT: !159 = !{!157}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.92e03c783d6af646:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !23)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 3, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4c02ca82d7b9fa3c:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !167)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !23)
// CHECK:STDOUT: !166 = !DILocation(line: 8, column: 3, scope: !164)
// CHECK:STDOUT: !167 = !{!165}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4c02ca82d7b9fa3c:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !171)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !23)
// CHECK:STDOUT: !170 = !DILocation(line: 8, column: 3, scope: !168)
// CHECK:STDOUT: !171 = !{!169}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !175)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !23)
// CHECK:STDOUT: !174 = !DILocation(line: 8, column: 3, scope: !172)
// CHECK:STDOUT: !175 = !{!173}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !179)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !23)
// CHECK:STDOUT: !178 = !DILocation(line: 8, column: 3, scope: !176)
// CHECK:STDOUT: !179 = !{!177}
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fb3d597a6b1c4ee1", scope: null, file: !124, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !181 = !DILocation(line: 8, column: 3, scope: !180)
// CHECK:STDOUT: !182 = !DILocation(line: 8, column: 21, scope: !180)
// CHECK:STDOUT: !183 = !DILocation(line: 8, column: 22, scope: !180)
// CHECK:STDOUT: !184 = !DILocation(line: 8, column: 30, scope: !180)
// CHECK:STDOUT: !185 = !DILocation(line: 8, column: 38, scope: !180)
// CHECK:STDOUT: !187 = !DILocation(line: 7, column: 1, scope: !180)
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.cd5dea75f888aed4", scope: null, file: !124, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !189 = !DILocation(line: 8, column: 3, scope: !188)
// CHECK:STDOUT: !190 = !DILocation(line: 8, column: 21, scope: !188)
// CHECK:STDOUT: !191 = !DILocation(line: 8, column: 22, scope: !188)
// CHECK:STDOUT: !192 = !DILocation(line: 8, column: 30, scope: !188)
// CHECK:STDOUT: !193 = !DILocation(line: 8, column: 38, scope: !188)
// CHECK:STDOUT: !195 = !DILocation(line: 7, column: 1, scope: !188)
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.be297cc7300419fe", scope: null, file: !124, line: 7, type: !77, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !197 = !DILocation(line: 8, column: 3, scope: !196)
// CHECK:STDOUT: !198 = !DILocation(line: 8, column: 22, scope: !196)
// CHECK:STDOUT: !199 = !DILocation(line: 8, column: 30, scope: !196)
// CHECK:STDOUT: !200 = !DILocation(line: 8, column: 38, scope: !196)
// CHECK:STDOUT: !201 = !DILocation(line: 8, column: 21, scope: !196)
// CHECK:STDOUT: !203 = !DILocation(line: 7, column: 1, scope: !196)
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.75a9ebb2041c173b(), !dbg !207
// CHECK:STDOUT:   call void @_CF.Main.f5d9b693014ce43e(), !dbg !208
// CHECK:STDOUT:   call void @_CF.Main.84270bb113cb754b(), !dbg !209
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !225
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.1.a), !dbg !225
// CHECK:STDOUT:   %.loc9_3.2.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !225
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.2.b), !dbg !225
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self), !dbg !229
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core"(ptr %self) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core"(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core"(ptr %self), !dbg !237
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.12212b84320f2fd6:core.Destroy.Core"(ptr %self), !dbg !245
// CHECK:STDOUT:   ret void, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !249
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.1.a), !dbg !249
// CHECK:STDOUT:   %.loc9_3.2.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !249
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.2.b), !dbg !249
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self) #0 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self), !dbg !253
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core"(ptr %self) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2f43e328c637647f:core.Destroy.Core"(ptr %self) #0 !dbg !259 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core"(ptr %self), !dbg !261
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core"(ptr %self) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !265
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.1.a), !dbg !265
// CHECK:STDOUT:   %.loc9_3.2.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !265
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.2.b), !dbg !265
// CHECK:STDOUT:   ret void, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2970be6ba4c2615c:core.Destroy.Core"(ptr %self) #0 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core"(ptr %self), !dbg !269
// CHECK:STDOUT:   ret void, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #0 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !273
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.1.a), !dbg !273
// CHECK:STDOUT:   %.loc9_3.2.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !273
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.2.b), !dbg !273
// CHECK:STDOUT:   ret void, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #0 !dbg !275 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self), !dbg !277
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core"(ptr %self) #0 !dbg !279 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core"(ptr %self) #0 !dbg !283 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core"(ptr %self), !dbg !285
// CHECK:STDOUT:   ret void, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.75a9ebb2041c173b() #0 !dbg !287 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !288
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !288
// CHECK:STDOUT:   %tuple.elem0.a28.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !290
// CHECK:STDOUT:   %.d62.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.a28.tuple.elem, i32 0, i32 0, !dbg !289
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.d62.a), !dbg !291
// CHECK:STDOUT:   %.162.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.a28.tuple.elem, i32 0, i32 1, !dbg !289
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.162.b), !dbg !292
// CHECK:STDOUT:   %tuple.elem1.79b.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !290
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.79b.tuple.elem), !dbg !293
// CHECK:STDOUT:   call void @"_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core"(ptr %w.var), !dbg !288
// CHECK:STDOUT:   ret void, !dbg !295
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f5d9b693014ce43e() #0 !dbg !296 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !297
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !297
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !297
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !297
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !297
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !297
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !298
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !297
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !299
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !297
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !301
// CHECK:STDOUT:   %tuple.elem0.25a.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !302
// CHECK:STDOUT:   %.e68.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.25a.tuple.elem, i32 0, i32 0, !dbg !300
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.e68.a, ptr %.7a56fd.1.temp), !dbg !300
// CHECK:STDOUT:   %.9f2.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.25a.tuple.elem, i32 0, i32 1, !dbg !300
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.9f2.b, ptr %.7a56fd.2.temp), !dbg !300
// CHECK:STDOUT:   %tuple.elem1.0e1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !302
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.0e1.tuple.elem, ptr %.7a56fd.3.temp), !dbg !302
// CHECK:STDOUT:   call void @"_CSelfDestruct.2f43e328c637647f:core.Destroy.Core"(ptr %w.var), !dbg !297
// CHECK:STDOUT:   ret void, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84270bb113cb754b() #0 !dbg !305 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !306
// CHECK:STDOUT:   %.7a56fd.4.temp = alloca {}, align 1, !dbg !306
// CHECK:STDOUT:   %.7a56fd.5.temp = alloca {}, align 1, !dbg !306
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !306
// CHECK:STDOUT:   %tuple.elem0.d0d.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !308
// CHECK:STDOUT:   %.b76.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.d0d.tuple.elem, i32 0, i32 0, !dbg !307
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b76.a), !dbg !309
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.4.temp), !dbg !306
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.4.temp), !dbg !310
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.5.temp), !dbg !306
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.5.temp), !dbg !311
// CHECK:STDOUT:   %.25e.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.d0d.tuple.elem, i32 0, i32 1, !dbg !307
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.25e.b, ptr %.7a56fd.4.temp), !dbg !307
// CHECK:STDOUT:   %tuple.elem1.c76.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !308
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.c76.tuple.elem, ptr %.7a56fd.5.temp), !dbg !308
// CHECK:STDOUT:   call void @"_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core"(ptr %w.var), !dbg !306
// CHECK:STDOUT:   ret void, !dbg !313
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!205}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !76 = !{null}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !130 = !{null, !23}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !204 = !DIFile(filename: "initialize_from_nested_literal.carbon", directory: "")
// CHECK:STDOUT: !205 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !204, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !204, line: 24, type: !77, spFlags: DISPFlagDefinition, unit: !205)
// CHECK:STDOUT: !207 = !DILocation(line: 25, column: 3, scope: !206)
// CHECK:STDOUT: !208 = !DILocation(line: 27, column: 3, scope: !206)
// CHECK:STDOUT: !209 = !DILocation(line: 29, column: 3, scope: !206)
// CHECK:STDOUT: !210 = !DILocation(line: 24, column: 1, scope: !206)
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !23)
// CHECK:STDOUT: !213 = !DILocation(line: 9, column: 3, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !23)
// CHECK:STDOUT: !217 = !DILocation(line: 9, column: 3, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !23)
// CHECK:STDOUT: !221 = !DILocation(line: 9, column: 3, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !226)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !23)
// CHECK:STDOUT: !225 = !DILocation(line: 9, column: 3, scope: !223)
// CHECK:STDOUT: !226 = !{!224}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !23)
// CHECK:STDOUT: !229 = !DILocation(line: 9, column: 3, scope: !227)
// CHECK:STDOUT: !230 = !{!228}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !234)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !23)
// CHECK:STDOUT: !233 = !DILocation(line: 9, column: 3, scope: !231)
// CHECK:STDOUT: !234 = !{!232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !238)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !23)
// CHECK:STDOUT: !237 = !DILocation(line: 9, column: 3, scope: !235)
// CHECK:STDOUT: !238 = !{!236}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !23)
// CHECK:STDOUT: !241 = !DILocation(line: 9, column: 3, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !246)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !23)
// CHECK:STDOUT: !245 = !DILocation(line: 9, column: 3, scope: !243)
// CHECK:STDOUT: !246 = !{!244}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !250)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !23)
// CHECK:STDOUT: !249 = !DILocation(line: 9, column: 3, scope: !247)
// CHECK:STDOUT: !250 = !{!248}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !254)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !23)
// CHECK:STDOUT: !253 = !DILocation(line: 9, column: 3, scope: !251)
// CHECK:STDOUT: !254 = !{!252}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !258)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !23)
// CHECK:STDOUT: !257 = !DILocation(line: 9, column: 3, scope: !255)
// CHECK:STDOUT: !258 = !{!256}
// CHECK:STDOUT: !259 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2f43e328c637647f:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !262)
// CHECK:STDOUT: !260 = !DILocalVariable(arg: 1, scope: !259, type: !23)
// CHECK:STDOUT: !261 = !DILocation(line: 9, column: 3, scope: !259)
// CHECK:STDOUT: !262 = !{!260}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !266)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !23)
// CHECK:STDOUT: !265 = !DILocation(line: 9, column: 3, scope: !263)
// CHECK:STDOUT: !266 = !{!264}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2970be6ba4c2615c:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !270)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !23)
// CHECK:STDOUT: !269 = !DILocation(line: 9, column: 3, scope: !267)
// CHECK:STDOUT: !270 = !{!268}
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !274)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !23)
// CHECK:STDOUT: !273 = !DILocation(line: 9, column: 3, scope: !271)
// CHECK:STDOUT: !274 = !{!272}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !278)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !23)
// CHECK:STDOUT: !277 = !DILocation(line: 9, column: 3, scope: !275)
// CHECK:STDOUT: !278 = !{!276}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !282)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !23)
// CHECK:STDOUT: !281 = !DILocation(line: 9, column: 3, scope: !279)
// CHECK:STDOUT: !282 = !{!280}
// CHECK:STDOUT: !283 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core", scope: null, file: !204, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !205, retainedNodes: !286)
// CHECK:STDOUT: !284 = !DILocalVariable(arg: 1, scope: !283, type: !23)
// CHECK:STDOUT: !285 = !DILocation(line: 9, column: 3, scope: !283)
// CHECK:STDOUT: !286 = !{!284}
// CHECK:STDOUT: !287 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.75a9ebb2041c173b", scope: null, file: !204, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !205)
// CHECK:STDOUT: !288 = !DILocation(line: 9, column: 3, scope: !287)
// CHECK:STDOUT: !289 = !DILocation(line: 9, column: 22, scope: !287)
// CHECK:STDOUT: !290 = !DILocation(line: 9, column: 21, scope: !287)
// CHECK:STDOUT: !291 = !DILocation(line: 9, column: 28, scope: !287)
// CHECK:STDOUT: !292 = !DILocation(line: 9, column: 41, scope: !287)
// CHECK:STDOUT: !293 = !DILocation(line: 9, column: 50, scope: !287)
// CHECK:STDOUT: !295 = !DILocation(line: 8, column: 1, scope: !287)
// CHECK:STDOUT: !296 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f5d9b693014ce43e", scope: null, file: !204, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !205)
// CHECK:STDOUT: !297 = !DILocation(line: 9, column: 3, scope: !296)
// CHECK:STDOUT: !298 = !DILocation(line: 9, column: 28, scope: !296)
// CHECK:STDOUT: !299 = !DILocation(line: 9, column: 41, scope: !296)
// CHECK:STDOUT: !300 = !DILocation(line: 9, column: 22, scope: !296)
// CHECK:STDOUT: !301 = !DILocation(line: 9, column: 50, scope: !296)
// CHECK:STDOUT: !302 = !DILocation(line: 9, column: 21, scope: !296)
// CHECK:STDOUT: !304 = !DILocation(line: 8, column: 1, scope: !296)
// CHECK:STDOUT: !305 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84270bb113cb754b", scope: null, file: !204, line: 8, type: !77, spFlags: DISPFlagDefinition, unit: !205)
// CHECK:STDOUT: !306 = !DILocation(line: 9, column: 3, scope: !305)
// CHECK:STDOUT: !307 = !DILocation(line: 9, column: 22, scope: !305)
// CHECK:STDOUT: !308 = !DILocation(line: 9, column: 21, scope: !305)
// CHECK:STDOUT: !309 = !DILocation(line: 9, column: 28, scope: !305)
// CHECK:STDOUT: !310 = !DILocation(line: 9, column: 41, scope: !305)
// CHECK:STDOUT: !311 = !DILocation(line: 9, column: 50, scope: !305)
// CHECK:STDOUT: !313 = !DILocation(line: 8, column: 1, scope: !305)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !316 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.84588f41d61dafba(i32 %n), !dbg !320
// CHECK:STDOUT:   ret void, !dbg !323
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !325 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !330
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !330
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !333 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !337
// CHECK:STDOUT:   ret void, !dbg !340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.84588f41d61dafba(i32 %x) #0 !dbg !342 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !348
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !350 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !355
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !356
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!315}
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
// CHECK:STDOUT: !130 = !{null, !23}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !314 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !315 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !314, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !316 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !314, line: 8, type: !79, spFlags: DISPFlagDefinition, unit: !315, retainedNodes: !324)
// CHECK:STDOUT: !317 = !DILocalVariable(arg: 1, scope: !316, type: !20)
// CHECK:STDOUT: !320 = !DILocation(line: 9, column: 16, scope: !316)
// CHECK:STDOUT: !323 = !DILocation(line: 8, column: 1, scope: !316)
// CHECK:STDOUT: !324 = !{!317}
// CHECK:STDOUT: !325 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !314, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !315, retainedNodes: !332)
// CHECK:STDOUT: !326 = !DILocalVariable(arg: 1, scope: !325, type: !23)
// CHECK:STDOUT: !330 = !DILocation(line: 15, column: 38, scope: !325)
// CHECK:STDOUT: !331 = !DILocation(line: 15, column: 31, scope: !325)
// CHECK:STDOUT: !332 = !{!326}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !314, line: 19, type: !131, spFlags: DISPFlagDefinition, unit: !315, retainedNodes: !341)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !333, type: !23)
// CHECK:STDOUT: !337 = !DILocation(line: 20, column: 16, scope: !333)
// CHECK:STDOUT: !340 = !DILocation(line: 19, column: 1, scope: !333)
// CHECK:STDOUT: !341 = !{!334}
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.84588f41d61dafba", scope: null, file: !314, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !315, retainedNodes: !349)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 1, scope: !342, type: !20)
// CHECK:STDOUT: !348 = !DILocation(line: 5, column: 3, scope: !342)
// CHECK:STDOUT: !349 = !{!343}
// CHECK:STDOUT: !350 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !314, line: 4, type: !25, spFlags: DISPFlagDefinition, unit: !315, retainedNodes: !357)
// CHECK:STDOUT: !351 = !DILocalVariable(arg: 1, scope: !350, type: !23)
// CHECK:STDOUT: !355 = !DILocation(line: 5, column: 10, scope: !350)
// CHECK:STDOUT: !356 = !DILocation(line: 5, column: 3, scope: !350)
// CHECK:STDOUT: !357 = !{!351}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !360 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.8122a03446648385(i32 %n), !dbg !364
// CHECK:STDOUT:   ret void, !dbg !365
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !367 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !372
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !372
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !373
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !375 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.3280a9d16838b780(ptr %c), !dbg !379
// CHECK:STDOUT:   ret void, !dbg !380
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.8122a03446648385(i32 %x) #0 !dbg !382 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.3280a9d16838b780(ptr %x) #0 !dbg !390 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !393
// CHECK:STDOUT:   ret void, !dbg !396
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!359}
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
// CHECK:STDOUT: !130 = !{null, !23}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !358 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !359 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !358, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !360 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !358, line: 8, type: !79, spFlags: DISPFlagDefinition, unit: !359, retainedNodes: !366)
// CHECK:STDOUT: !361 = !DILocalVariable(arg: 1, scope: !360, type: !20)
// CHECK:STDOUT: !364 = !DILocation(line: 9, column: 3, scope: !360)
// CHECK:STDOUT: !365 = !DILocation(line: 8, column: 1, scope: !360)
// CHECK:STDOUT: !366 = !{!361}
// CHECK:STDOUT: !367 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !358, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !359, retainedNodes: !374)
// CHECK:STDOUT: !368 = !DILocalVariable(arg: 1, scope: !367, type: !23)
// CHECK:STDOUT: !372 = !DILocation(line: 15, column: 38, scope: !367)
// CHECK:STDOUT: !373 = !DILocation(line: 15, column: 31, scope: !367)
// CHECK:STDOUT: !374 = !{!368}
// CHECK:STDOUT: !375 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !358, line: 19, type: !131, spFlags: DISPFlagDefinition, unit: !359, retainedNodes: !381)
// CHECK:STDOUT: !376 = !DILocalVariable(arg: 1, scope: !375, type: !23)
// CHECK:STDOUT: !379 = !DILocation(line: 20, column: 3, scope: !375)
// CHECK:STDOUT: !380 = !DILocation(line: 19, column: 1, scope: !375)
// CHECK:STDOUT: !381 = !{!376}
// CHECK:STDOUT: !382 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.8122a03446648385", scope: null, file: !358, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !359, retainedNodes: !389)
// CHECK:STDOUT: !383 = !DILocalVariable(arg: 1, scope: !382, type: !20)
// CHECK:STDOUT: !388 = !DILocation(line: 4, column: 1, scope: !382)
// CHECK:STDOUT: !389 = !{!383}
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.3280a9d16838b780", scope: null, file: !358, line: 4, type: !131, spFlags: DISPFlagDefinition, unit: !359, retainedNodes: !397)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !23)
// CHECK:STDOUT: !393 = !DILocation(line: 5, column: 14, scope: !390)
// CHECK:STDOUT: !396 = !DILocation(line: 4, column: 1, scope: !390)
// CHECK:STDOUT: !397 = !{!391}
// CHECK:STDOUT:
