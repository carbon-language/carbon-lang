// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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

// --- initialize_from_template_method_call.carbon

library "[[@TEST_NAME]]";

class B {
  fn MakeB() -> B {
    return {};
  }
}

class A {
  var field: B;
}

fn MakeA(template T: type) -> A {
   return {
     .field = T.MakeB(),
   };
}

fn Run() {
  MakeA(B);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest1.Main(i32 %n) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.55456748a8a2eb70(i32 %n), !dbg !48
// CHECK:STDOUT:   ret i32 %F.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   %.loc16_42.2 = load i32, ptr %.loc16_42.1.n, align 4, !dbg !56
// CHECK:STDOUT:   ret i32 %.loc16_42.2, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest2.Main(ptr %c) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.9522ff908e97e267(ptr %c), !dbg !65
// CHECK:STDOUT:   ret i32 %F.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.55456748a8a2eb70(i32 %x) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9522ff908e97e267(ptr %x) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core"(ptr %x), !dbg !84
// CHECK:STDOUT:   ret i32 %C.as.ImplicitAs.impl.Convert.call, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !34, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !36)
// CHECK:STDOUT: !48 = !DILocation(line: 10, column: 10, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 3, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !58)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 38, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 31, scope: !51)
// CHECK:STDOUT: !58 = !{!52}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !34, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !67)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 10, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 3, scope: !59)
// CHECK:STDOUT: !67 = !{!60}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !34, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !78)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !36)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 3, scope: !68)
// CHECK:STDOUT: !78 = !{!69}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !34, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !89)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !39)
// CHECK:STDOUT: !84 = !DILocation(line: 5, column: 16, scope: !79)
// CHECK:STDOUT: !88 = !DILocation(line: 6, column: 3, scope: !79)
// CHECK:STDOUT: !89 = !{!80}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_dependent_target_from_literal.carbon'
// CHECK:STDOUT: source_filename = "initialize_dependent_target_from_literal.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70() #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !98
// CHECK:STDOUT:   store i32 0, ptr %w.var, align 4, !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %w.var), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !103 {
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !90 = !DIFile(filename: "initialize_dependent_target_from_literal.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !90, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 9, column: 3, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 8, column: 1, scope: !94)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !90, line: 4, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 5, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !102 = !DILocation(line: 4, column: 1, scope: !97)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !26, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !36)
// CHECK:STDOUT: !108 = !DILocation(line: 30, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 0), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70(i32 %x) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !118
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !118
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %w.var), !dbg !118
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!111}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !110 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !110, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !110, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !111)
// CHECK:STDOUT: !113 = !DILocation(line: 9, column: 3, scope: !112)
// CHECK:STDOUT: !114 = !DILocation(line: 8, column: 1, scope: !112)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !110, line: 4, type: !101, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !122)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !36)
// CHECK:STDOUT: !118 = !DILocation(line: 5, column: 3, scope: !115)
// CHECK:STDOUT: !121 = !DILocation(line: 4, column: 1, scope: !115)
// CHECK:STDOUT: !122 = !{!116}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !26, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !36)
// CHECK:STDOUT: !128 = !DILocation(line: 30, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
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
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fb3d597a6b1c4ee1(), !dbg !144
// CHECK:STDOUT:   call void @_CF.Main.689d8af583927bab(), !dbg !145
// CHECK:STDOUT:   call void @_CF.Main.be297cc7300419fe(), !dbg !146
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fb3d597a6b1c4ee1() #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !149
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !149
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !150
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !151
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !150
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !152
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !150
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !153
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6add1b2daa1190e3(ptr %w.var), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.689d8af583927bab() #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !159
// CHECK:STDOUT:   %.b44.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !160
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b44.array.index), !dbg !161
// CHECK:STDOUT:   %.5b6.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !160
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.5b6.array.index), !dbg !162
// CHECK:STDOUT:   %.97d.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !160
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.97d.array.index), !dbg !163
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a8283838dc120d4f(ptr %w.var), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.be297cc7300419fe() #0 !dbg !166 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !167
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !167
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !167
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !167
// CHECK:STDOUT:   %tuple.9aa = alloca { ptr, ptr, ptr }, align 8, !dbg !171
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !167
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !167
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !168
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !167
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !169
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !167
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !170
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.9aa), !dbg !171
// CHECK:STDOUT:   %tuple.9aa1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 0, !dbg !171
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.9aa1, align 8, !dbg !171
// CHECK:STDOUT:   %tuple.9aa2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 1, !dbg !171
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.9aa2, align 8, !dbg !171
// CHECK:STDOUT:   %tuple.9aa3 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 2, !dbg !171
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.9aa3, align 8, !dbg !171
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.9aa), !dbg !167
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %w.var), !dbg !167
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6add1b2daa1190e3(ptr %t) #0 !dbg !174 {
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a8283838dc120d4f(ptr %t) #0 !dbg !181 {
// CHECK:STDOUT:   ret void, !dbg !186
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %t) #0 !dbg !188 {
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!131}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !130 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !131 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !130, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !134 = !{!39, !39}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !130, line: 14, type: !135, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !142)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !39)
// CHECK:STDOUT: !141 = !DILocation(line: 14, column: 45, scope: !136)
// CHECK:STDOUT: !142 = !{!137}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !130, line: 17, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !144 = !DILocation(line: 18, column: 3, scope: !143)
// CHECK:STDOUT: !145 = !DILocation(line: 20, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !DILocation(line: 22, column: 3, scope: !143)
// CHECK:STDOUT: !147 = !DILocation(line: 17, column: 1, scope: !143)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fb3d597a6b1c4ee1", scope: null, file: !130, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !149 = !DILocation(line: 8, column: 3, scope: !148)
// CHECK:STDOUT: !150 = !DILocation(line: 8, column: 21, scope: !148)
// CHECK:STDOUT: !151 = !DILocation(line: 8, column: 22, scope: !148)
// CHECK:STDOUT: !152 = !DILocation(line: 8, column: 30, scope: !148)
// CHECK:STDOUT: !153 = !DILocation(line: 8, column: 38, scope: !148)
// CHECK:STDOUT: !155 = !{null, !39}
// CHECK:STDOUT: !156 = !DISubroutineType(types: !155)
// CHECK:STDOUT: !157 = !DILocation(line: 7, column: 1, scope: !148)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.689d8af583927bab", scope: null, file: !130, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !159 = !DILocation(line: 8, column: 3, scope: !158)
// CHECK:STDOUT: !160 = !DILocation(line: 8, column: 21, scope: !158)
// CHECK:STDOUT: !161 = !DILocation(line: 8, column: 22, scope: !158)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 30, scope: !158)
// CHECK:STDOUT: !163 = !DILocation(line: 8, column: 38, scope: !158)
// CHECK:STDOUT: !165 = !DILocation(line: 7, column: 1, scope: !158)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.be297cc7300419fe", scope: null, file: !130, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !167 = !DILocation(line: 8, column: 3, scope: !166)
// CHECK:STDOUT: !168 = !DILocation(line: 8, column: 22, scope: !166)
// CHECK:STDOUT: !169 = !DILocation(line: 8, column: 30, scope: !166)
// CHECK:STDOUT: !170 = !DILocation(line: 8, column: 38, scope: !166)
// CHECK:STDOUT: !171 = !DILocation(line: 8, column: 21, scope: !166)
// CHECK:STDOUT: !173 = !DILocation(line: 7, column: 1, scope: !166)
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6add1b2daa1190e3", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !180)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !39)
// CHECK:STDOUT: !179 = !DILocation(line: 30, column: 1, scope: !174)
// CHECK:STDOUT: !180 = !{!175}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a8283838dc120d4f", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !187)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !39)
// CHECK:STDOUT: !186 = !DILocation(line: 30, column: 1, scope: !181)
// CHECK:STDOUT: !187 = !{!182}
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3773d92a34eaed6b", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !194)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !188, type: !39)
// CHECK:STDOUT: !193 = !DILocation(line: 30, column: 1, scope: !188)
// CHECK:STDOUT: !194 = !{!189}
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.75a9ebb2041c173b(), !dbg !198
// CHECK:STDOUT:   call void @_CF.Main.f5d9b693014ce43e(), !dbg !199
// CHECK:STDOUT:   call void @_CF.Main.84270bb113cb754b(), !dbg !200
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.75a9ebb2041c173b() #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !203
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !203
// CHECK:STDOUT:   %tuple.elem0.16b.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !205
// CHECK:STDOUT:   %.ae2.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.16b.tuple.elem, i32 0, i32 0, !dbg !204
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.ae2.a), !dbg !206
// CHECK:STDOUT:   %.ffd.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.16b.tuple.elem, i32 0, i32 1, !dbg !204
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.ffd.b), !dbg !207
// CHECK:STDOUT:   %tuple.elem1.b82.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !205
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.b82.tuple.elem), !dbg !208
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f5aa614245b4c289(ptr %w.var), !dbg !203
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f5d9b693014ce43e() #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !212
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !212
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !212
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !212
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !212
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !212
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !212
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !214
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !212
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !216
// CHECK:STDOUT:   %tuple.elem0.c3c.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !217
// CHECK:STDOUT:   %.ce4.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.c3c.tuple.elem, i32 0, i32 0, !dbg !215
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.ce4.a, ptr %.7a56fd.1.temp), !dbg !215
// CHECK:STDOUT:   %.0ae.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.c3c.tuple.elem, i32 0, i32 1, !dbg !215
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.0ae.b, ptr %.7a56fd.2.temp), !dbg !215
// CHECK:STDOUT:   %tuple.elem1.1e9.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !217
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.1e9.tuple.elem, ptr %.7a56fd.3.temp), !dbg !217
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ac60fe411ae2f070(ptr %w.var), !dbg !212
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84270bb113cb754b() #0 !dbg !220 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !221
// CHECK:STDOUT:   %.7a56fd.4.temp = alloca {}, align 1, !dbg !221
// CHECK:STDOUT:   %.7a56fd.5.temp = alloca {}, align 1, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !221
// CHECK:STDOUT:   %tuple.elem0.297.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !223
// CHECK:STDOUT:   %.014.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.297.tuple.elem, i32 0, i32 0, !dbg !222
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.014.a), !dbg !224
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.4.temp), !dbg !221
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.4.temp), !dbg !225
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.5.temp), !dbg !221
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.5.temp), !dbg !226
// CHECK:STDOUT:   %.94f.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.297.tuple.elem, i32 0, i32 1, !dbg !222
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.94f.b, ptr %.7a56fd.4.temp), !dbg !222
// CHECK:STDOUT:   %tuple.elem1.76f.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !223
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.76f.tuple.elem, ptr %.7a56fd.5.temp), !dbg !223
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f34cebc0ce9f922d(ptr %w.var), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f5aa614245b4c289(ptr %t) #0 !dbg !229 {
// CHECK:STDOUT:   ret void, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ac60fe411ae2f070(ptr %t) #0 !dbg !236 {
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f34cebc0ce9f922d(ptr %t) #0 !dbg !243 {
// CHECK:STDOUT:   ret void, !dbg !248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!196}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !155 = !{null, !39}
// CHECK:STDOUT: !156 = !DISubroutineType(types: !155)
// CHECK:STDOUT: !195 = !DIFile(filename: "initialize_from_nested_literal.carbon", directory: "")
// CHECK:STDOUT: !196 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !195, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !195, line: 24, type: !93, spFlags: DISPFlagDefinition, unit: !196)
// CHECK:STDOUT: !198 = !DILocation(line: 25, column: 3, scope: !197)
// CHECK:STDOUT: !199 = !DILocation(line: 27, column: 3, scope: !197)
// CHECK:STDOUT: !200 = !DILocation(line: 29, column: 3, scope: !197)
// CHECK:STDOUT: !201 = !DILocation(line: 24, column: 1, scope: !197)
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.75a9ebb2041c173b", scope: null, file: !195, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !196)
// CHECK:STDOUT: !203 = !DILocation(line: 9, column: 3, scope: !202)
// CHECK:STDOUT: !204 = !DILocation(line: 9, column: 22, scope: !202)
// CHECK:STDOUT: !205 = !DILocation(line: 9, column: 21, scope: !202)
// CHECK:STDOUT: !206 = !DILocation(line: 9, column: 28, scope: !202)
// CHECK:STDOUT: !207 = !DILocation(line: 9, column: 41, scope: !202)
// CHECK:STDOUT: !208 = !DILocation(line: 9, column: 50, scope: !202)
// CHECK:STDOUT: !210 = !DILocation(line: 8, column: 1, scope: !202)
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f5d9b693014ce43e", scope: null, file: !195, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !196)
// CHECK:STDOUT: !212 = !DILocation(line: 9, column: 3, scope: !211)
// CHECK:STDOUT: !213 = !DILocation(line: 9, column: 28, scope: !211)
// CHECK:STDOUT: !214 = !DILocation(line: 9, column: 41, scope: !211)
// CHECK:STDOUT: !215 = !DILocation(line: 9, column: 22, scope: !211)
// CHECK:STDOUT: !216 = !DILocation(line: 9, column: 50, scope: !211)
// CHECK:STDOUT: !217 = !DILocation(line: 9, column: 21, scope: !211)
// CHECK:STDOUT: !219 = !DILocation(line: 8, column: 1, scope: !211)
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84270bb113cb754b", scope: null, file: !195, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !196)
// CHECK:STDOUT: !221 = !DILocation(line: 9, column: 3, scope: !220)
// CHECK:STDOUT: !222 = !DILocation(line: 9, column: 22, scope: !220)
// CHECK:STDOUT: !223 = !DILocation(line: 9, column: 21, scope: !220)
// CHECK:STDOUT: !224 = !DILocation(line: 9, column: 28, scope: !220)
// CHECK:STDOUT: !225 = !DILocation(line: 9, column: 41, scope: !220)
// CHECK:STDOUT: !226 = !DILocation(line: 9, column: 50, scope: !220)
// CHECK:STDOUT: !228 = !DILocation(line: 8, column: 1, scope: !220)
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f5aa614245b4c289", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !196, retainedNodes: !235)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !39)
// CHECK:STDOUT: !234 = !DILocation(line: 30, column: 1, scope: !229)
// CHECK:STDOUT: !235 = !{!230}
// CHECK:STDOUT: !236 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ac60fe411ae2f070", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !196, retainedNodes: !242)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !236, type: !39)
// CHECK:STDOUT: !241 = !DILocation(line: 30, column: 1, scope: !236)
// CHECK:STDOUT: !242 = !{!237}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f34cebc0ce9f922d", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !196, retainedNodes: !249)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !39)
// CHECK:STDOUT: !248 = !DILocation(line: 30, column: 1, scope: !243)
// CHECK:STDOUT: !249 = !{!244}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !252 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.55456748a8a2eb70(i32 %n), !dbg !256
// CHECK:STDOUT:   ret void, !dbg !259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !261 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !266
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !266
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !273
// CHECK:STDOUT:   ret void, !dbg !276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.55456748a8a2eb70(i32 %x) #0 !dbg !278 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !284
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !286 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %x), !dbg !291
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!251}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !155 = !{null, !39}
// CHECK:STDOUT: !156 = !DISubroutineType(types: !155)
// CHECK:STDOUT: !250 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !251 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !250, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !250, line: 8, type: !101, spFlags: DISPFlagDefinition, unit: !251, retainedNodes: !260)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !36)
// CHECK:STDOUT: !256 = !DILocation(line: 9, column: 16, scope: !252)
// CHECK:STDOUT: !259 = !DILocation(line: 8, column: 1, scope: !252)
// CHECK:STDOUT: !260 = !{!253}
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.55456748a8a2eb70.Core", scope: null, file: !250, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !251, retainedNodes: !268)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !39)
// CHECK:STDOUT: !266 = !DILocation(line: 15, column: 38, scope: !261)
// CHECK:STDOUT: !267 = !DILocation(line: 15, column: 31, scope: !261)
// CHECK:STDOUT: !268 = !{!262}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !250, line: 19, type: !156, spFlags: DISPFlagDefinition, unit: !251, retainedNodes: !277)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !39)
// CHECK:STDOUT: !273 = !DILocation(line: 20, column: 16, scope: !269)
// CHECK:STDOUT: !276 = !DILocation(line: 19, column: 1, scope: !269)
// CHECK:STDOUT: !277 = !{!270}
// CHECK:STDOUT: !278 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.55456748a8a2eb70", scope: null, file: !250, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !251, retainedNodes: !285)
// CHECK:STDOUT: !279 = !DILocalVariable(arg: 1, scope: !278, type: !36)
// CHECK:STDOUT: !284 = !DILocation(line: 5, column: 3, scope: !278)
// CHECK:STDOUT: !285 = !{!279}
// CHECK:STDOUT: !286 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !250, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !251, retainedNodes: !293)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 1, scope: !286, type: !39)
// CHECK:STDOUT: !291 = !DILocation(line: 5, column: 10, scope: !286)
// CHECK:STDOUT: !292 = !DILocation(line: 5, column: 3, scope: !286)
// CHECK:STDOUT: !293 = !{!287}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !296 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.0f3e15a21cc8e963(i32 %n), !dbg !300
// CHECK:STDOUT:   ret void, !dbg !301
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !303 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !308
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !308
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !309
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !311 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.9f9f842f4c8e7f62(ptr %c), !dbg !315
// CHECK:STDOUT:   ret void, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.0f3e15a21cc8e963(i32 %x) #0 !dbg !318 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.9f9f842f4c8e7f62(ptr %x) #0 !dbg !326 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %x), !dbg !329
// CHECK:STDOUT:   ret void, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!295}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !155 = !{null, !39}
// CHECK:STDOUT: !156 = !DISubroutineType(types: !155)
// CHECK:STDOUT: !294 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !295 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !294, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !296 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !294, line: 8, type: !101, spFlags: DISPFlagDefinition, unit: !295, retainedNodes: !302)
// CHECK:STDOUT: !297 = !DILocalVariable(arg: 1, scope: !296, type: !36)
// CHECK:STDOUT: !300 = !DILocation(line: 9, column: 3, scope: !296)
// CHECK:STDOUT: !301 = !DILocation(line: 8, column: 1, scope: !296)
// CHECK:STDOUT: !302 = !{!297}
// CHECK:STDOUT: !303 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.55456748a8a2eb70.Core", scope: null, file: !294, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !295, retainedNodes: !310)
// CHECK:STDOUT: !304 = !DILocalVariable(arg: 1, scope: !303, type: !39)
// CHECK:STDOUT: !308 = !DILocation(line: 15, column: 38, scope: !303)
// CHECK:STDOUT: !309 = !DILocation(line: 15, column: 31, scope: !303)
// CHECK:STDOUT: !310 = !{!304}
// CHECK:STDOUT: !311 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !294, line: 19, type: !156, spFlags: DISPFlagDefinition, unit: !295, retainedNodes: !317)
// CHECK:STDOUT: !312 = !DILocalVariable(arg: 1, scope: !311, type: !39)
// CHECK:STDOUT: !315 = !DILocation(line: 20, column: 3, scope: !311)
// CHECK:STDOUT: !316 = !DILocation(line: 19, column: 1, scope: !311)
// CHECK:STDOUT: !317 = !{!312}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.0f3e15a21cc8e963", scope: null, file: !294, line: 4, type: !101, spFlags: DISPFlagDefinition, unit: !295, retainedNodes: !325)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !36)
// CHECK:STDOUT: !324 = !DILocation(line: 4, column: 1, scope: !318)
// CHECK:STDOUT: !325 = !{!319}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.9f9f842f4c8e7f62", scope: null, file: !294, line: 4, type: !156, spFlags: DISPFlagDefinition, unit: !295, retainedNodes: !333)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !39)
// CHECK:STDOUT: !329 = !DILocation(line: 5, column: 14, scope: !326)
// CHECK:STDOUT: !332 = !DILocation(line: 4, column: 1, scope: !326)
// CHECK:STDOUT: !333 = !{!327}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_template_method_call.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_template_method_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @B.val.loc6_14 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeB.B.Main(ptr sret({}) %return) #0 !dbg !338 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc6_14, i64 0, i1 false), !dbg !341
// CHECK:STDOUT:   ret void, !dbg !341
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !342 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_10.1.temp = alloca { {} }, align 1, !dbg !343
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_10.1.temp), !dbg !343
// CHECK:STDOUT:   call void @_CMakeA.Main.a6a60dfee2d7d273(ptr %.loc21_10.1.temp), !dbg !343
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %.loc21_10.1.temp), !dbg !343
// CHECK:STDOUT:   ret i32 0, !dbg !344
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMakeA.Main.a6a60dfee2d7d273(ptr sret({ {} }) %return) #0 !dbg !345 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.f4aa6f.1.field = getelementptr inbounds nuw { {} }, ptr %return, i32 0, i32 0, !dbg !349
// CHECK:STDOUT:   call void @_CMakeB.B.Main(ptr %.f4aa6f.1.field), !dbg !348
// CHECK:STDOUT:   ret void, !dbg !350
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %t) #0 !dbg !351 {
// CHECK:STDOUT:   ret void, !dbg !356
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!335}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !132 = !{!39}
// CHECK:STDOUT: !133 = !DISubroutineType(types: !132)
// CHECK:STDOUT: !155 = !{null, !39}
// CHECK:STDOUT: !156 = !DISubroutineType(types: !155)
// CHECK:STDOUT: !334 = !DIFile(filename: "initialize_from_template_method_call.carbon", directory: "")
// CHECK:STDOUT: !335 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !334, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !336 = !{!36}
// CHECK:STDOUT: !337 = !DISubroutineType(types: !336)
// CHECK:STDOUT: !338 = distinct !DISubprogram(name: "MakeB", linkageName: "_CMakeB.B.Main", scope: null, file: !334, line: 5, type: !133, spFlags: DISPFlagDefinition, unit: !335)
// CHECK:STDOUT: !341 = !DILocation(line: 6, column: 5, scope: !338)
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !334, line: 20, type: !337, spFlags: DISPFlagDefinition, unit: !335)
// CHECK:STDOUT: !343 = !DILocation(line: 21, column: 3, scope: !342)
// CHECK:STDOUT: !344 = !DILocation(line: 20, column: 1, scope: !342)
// CHECK:STDOUT: !345 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.a6a60dfee2d7d273", scope: null, file: !334, line: 14, type: !133, spFlags: DISPFlagDefinition, unit: !335)
// CHECK:STDOUT: !348 = !DILocation(line: 16, column: 15, scope: !345)
// CHECK:STDOUT: !349 = !DILocation(line: 15, column: 11, scope: !345)
// CHECK:STDOUT: !350 = !DILocation(line: 15, column: 4, scope: !345)
// CHECK:STDOUT: !351 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e517637b3b58cb2c", scope: null, file: !26, line: 30, type: !156, spFlags: DISPFlagDefinition, unit: !335, retainedNodes: !357)
// CHECK:STDOUT: !352 = !DILocalVariable(arg: 1, scope: !351, type: !39)
// CHECK:STDOUT: !356 = !DILocation(line: 30, column: 1, scope: !351)
// CHECK:STDOUT: !357 = !{!352}
// CHECK:STDOUT:
