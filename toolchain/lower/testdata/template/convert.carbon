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
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !{!6, !6}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!6, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !35, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !6)
// CHECK:STDOUT: !48 = !DILocation(line: 10, column: 10, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 3, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core", scope: null, file: !35, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !58)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 38, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 31, scope: !51)
// CHECK:STDOUT: !58 = !{!52}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !35, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !67)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 10, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 3, scope: !59)
// CHECK:STDOUT: !67 = !{!60}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !35, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !78)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !6)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 3, scope: !68)
// CHECK:STDOUT: !78 = !{!69}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !35, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !89)
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
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !90 = !DIFile(filename: "initialize_dependent_target_from_literal.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !90, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 9, column: 3, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 8, column: 1, scope: !94)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !90, line: 4, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 5, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !DILocation(line: 4, column: 1, scope: !97)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 0), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70(i32 %x) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !111
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !111
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!102}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !101 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !102 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !101, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !101, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 9, column: 3, scope: !103)
// CHECK:STDOUT: !105 = !{null, !6}
// CHECK:STDOUT: !106 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !107 = !DILocation(line: 8, column: 1, scope: !103)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !101, line: 4, type: !106, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !115)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !6)
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
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fb3d597a6b1c4ee1(), !dbg !130
// CHECK:STDOUT:   call void @_CF.Main.689d8af583927bab(), !dbg !131
// CHECK:STDOUT:   call void @_CF.Main.be297cc7300419fe(), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fb3d597a6b1c4ee1() #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !135
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !137
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !136
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !138
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !136
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.689d8af583927bab() #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !143
// CHECK:STDOUT:   %.b44.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !144
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b44.array.index), !dbg !145
// CHECK:STDOUT:   %.5b6.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !144
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.5b6.array.index), !dbg !146
// CHECK:STDOUT:   %.97d.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !144
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.97d.array.index), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.be297cc7300419fe() #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !151
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !151
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !151
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !151
// CHECK:STDOUT:   %tuple.9aa = alloca { ptr, ptr, ptr }, align 8, !dbg !155
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !151
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !151
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !151
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !151
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.9aa), !dbg !155
// CHECK:STDOUT:   %tuple.9aa1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.9aa1, align 8, !dbg !155
// CHECK:STDOUT:   %tuple.9aa2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 1, !dbg !155
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.9aa2, align 8, !dbg !155
// CHECK:STDOUT:   %tuple.9aa3 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 2, !dbg !155
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.9aa3, align 8, !dbg !155
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.9aa), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !157
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
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !116 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !117 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !116, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !120 = !{!39, !39}
// CHECK:STDOUT: !121 = !DISubroutineType(types: !120)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !116, line: 14, type: !121, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !128)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !39)
// CHECK:STDOUT: !127 = !DILocation(line: 14, column: 45, scope: !122)
// CHECK:STDOUT: !128 = !{!123}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !116, line: 17, type: !93, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !130 = !DILocation(line: 18, column: 3, scope: !129)
// CHECK:STDOUT: !131 = !DILocation(line: 20, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !DILocation(line: 22, column: 3, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 17, column: 1, scope: !129)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fb3d597a6b1c4ee1", scope: null, file: !116, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !135 = !DILocation(line: 8, column: 3, scope: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 8, column: 21, scope: !134)
// CHECK:STDOUT: !137 = !DILocation(line: 8, column: 22, scope: !134)
// CHECK:STDOUT: !138 = !DILocation(line: 8, column: 30, scope: !134)
// CHECK:STDOUT: !139 = !DILocation(line: 8, column: 38, scope: !134)
// CHECK:STDOUT: !141 = !DILocation(line: 7, column: 1, scope: !134)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.689d8af583927bab", scope: null, file: !116, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !143 = !DILocation(line: 8, column: 3, scope: !142)
// CHECK:STDOUT: !144 = !DILocation(line: 8, column: 21, scope: !142)
// CHECK:STDOUT: !145 = !DILocation(line: 8, column: 22, scope: !142)
// CHECK:STDOUT: !146 = !DILocation(line: 8, column: 30, scope: !142)
// CHECK:STDOUT: !147 = !DILocation(line: 8, column: 38, scope: !142)
// CHECK:STDOUT: !149 = !DILocation(line: 7, column: 1, scope: !142)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.be297cc7300419fe", scope: null, file: !116, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !117)
// CHECK:STDOUT: !151 = !DILocation(line: 8, column: 3, scope: !150)
// CHECK:STDOUT: !152 = !DILocation(line: 8, column: 22, scope: !150)
// CHECK:STDOUT: !153 = !DILocation(line: 8, column: 30, scope: !150)
// CHECK:STDOUT: !154 = !DILocation(line: 8, column: 38, scope: !150)
// CHECK:STDOUT: !155 = !DILocation(line: 8, column: 21, scope: !150)
// CHECK:STDOUT: !157 = !DILocation(line: 7, column: 1, scope: !150)
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.75a9ebb2041c173b(), !dbg !161
// CHECK:STDOUT:   call void @_CF.Main.f5d9b693014ce43e(), !dbg !162
// CHECK:STDOUT:   call void @_CF.Main.84270bb113cb754b(), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.75a9ebb2041c173b() #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !166
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !166
// CHECK:STDOUT:   %tuple.elem0.16b.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !168
// CHECK:STDOUT:   %.ae2.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.16b.tuple.elem, i32 0, i32 0, !dbg !167
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.ae2.a), !dbg !169
// CHECK:STDOUT:   %.ffd.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.16b.tuple.elem, i32 0, i32 1, !dbg !167
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.ffd.b), !dbg !170
// CHECK:STDOUT:   %tuple.elem1.b82.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !168
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.b82.tuple.elem), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f5d9b693014ce43e() #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !175
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !175
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !175
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !176
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !175
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !175
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !179
// CHECK:STDOUT:   %tuple.elem0.c3c.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !180
// CHECK:STDOUT:   %.ce4.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.c3c.tuple.elem, i32 0, i32 0, !dbg !178
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.ce4.a, ptr %.7a56fd.1.temp), !dbg !178
// CHECK:STDOUT:   %.0ae.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.c3c.tuple.elem, i32 0, i32 1, !dbg !178
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.0ae.b, ptr %.7a56fd.2.temp), !dbg !178
// CHECK:STDOUT:   %tuple.elem1.1e9.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !180
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.1e9.tuple.elem, ptr %.7a56fd.3.temp), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84270bb113cb754b() #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !184
// CHECK:STDOUT:   %.7a56fd.4.temp = alloca {}, align 1, !dbg !184
// CHECK:STDOUT:   %.7a56fd.5.temp = alloca {}, align 1, !dbg !184
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !184
// CHECK:STDOUT:   %tuple.elem0.297.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   %.014.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.297.tuple.elem, i32 0, i32 0, !dbg !185
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.014.a), !dbg !187
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.4.temp), !dbg !184
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.4.temp), !dbg !188
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.5.temp), !dbg !184
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.5.temp), !dbg !189
// CHECK:STDOUT:   %.94f.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.297.tuple.elem, i32 0, i32 1, !dbg !185
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.94f.b, ptr %.7a56fd.4.temp), !dbg !185
// CHECK:STDOUT:   %tuple.elem1.76f.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !186
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.76f.tuple.elem, ptr %.7a56fd.5.temp), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !191
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!159}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !158 = !DIFile(filename: "initialize_from_nested_literal.carbon", directory: "")
// CHECK:STDOUT: !159 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !158, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !158, line: 24, type: !93, spFlags: DISPFlagDefinition, unit: !159)
// CHECK:STDOUT: !161 = !DILocation(line: 25, column: 3, scope: !160)
// CHECK:STDOUT: !162 = !DILocation(line: 27, column: 3, scope: !160)
// CHECK:STDOUT: !163 = !DILocation(line: 29, column: 3, scope: !160)
// CHECK:STDOUT: !164 = !DILocation(line: 24, column: 1, scope: !160)
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.75a9ebb2041c173b", scope: null, file: !158, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !159)
// CHECK:STDOUT: !166 = !DILocation(line: 9, column: 3, scope: !165)
// CHECK:STDOUT: !167 = !DILocation(line: 9, column: 22, scope: !165)
// CHECK:STDOUT: !168 = !DILocation(line: 9, column: 21, scope: !165)
// CHECK:STDOUT: !169 = !DILocation(line: 9, column: 28, scope: !165)
// CHECK:STDOUT: !170 = !DILocation(line: 9, column: 41, scope: !165)
// CHECK:STDOUT: !171 = !DILocation(line: 9, column: 50, scope: !165)
// CHECK:STDOUT: !173 = !DILocation(line: 8, column: 1, scope: !165)
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f5d9b693014ce43e", scope: null, file: !158, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !159)
// CHECK:STDOUT: !175 = !DILocation(line: 9, column: 3, scope: !174)
// CHECK:STDOUT: !176 = !DILocation(line: 9, column: 28, scope: !174)
// CHECK:STDOUT: !177 = !DILocation(line: 9, column: 41, scope: !174)
// CHECK:STDOUT: !178 = !DILocation(line: 9, column: 22, scope: !174)
// CHECK:STDOUT: !179 = !DILocation(line: 9, column: 50, scope: !174)
// CHECK:STDOUT: !180 = !DILocation(line: 9, column: 21, scope: !174)
// CHECK:STDOUT: !182 = !DILocation(line: 8, column: 1, scope: !174)
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84270bb113cb754b", scope: null, file: !158, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !159)
// CHECK:STDOUT: !184 = !DILocation(line: 9, column: 3, scope: !183)
// CHECK:STDOUT: !185 = !DILocation(line: 9, column: 22, scope: !183)
// CHECK:STDOUT: !186 = !DILocation(line: 9, column: 21, scope: !183)
// CHECK:STDOUT: !187 = !DILocation(line: 9, column: 28, scope: !183)
// CHECK:STDOUT: !188 = !DILocation(line: 9, column: 41, scope: !183)
// CHECK:STDOUT: !189 = !DILocation(line: 9, column: 50, scope: !183)
// CHECK:STDOUT: !191 = !DILocation(line: 8, column: 1, scope: !183)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.55456748a8a2eb70(i32 %n), !dbg !200
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !205 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !210
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !210
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !213 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !217
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.55456748a8a2eb70(i32 %x) #0 !dbg !222 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %x), !dbg !235
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!193}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!6, !6}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!6, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !105 = !{null, !6}
// CHECK:STDOUT: !106 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !192 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !193 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !192, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !194 = !{null, !39}
// CHECK:STDOUT: !195 = !DISubroutineType(types: !194)
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !192, line: 8, type: !106, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !204)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !6)
// CHECK:STDOUT: !200 = !DILocation(line: 9, column: 16, scope: !196)
// CHECK:STDOUT: !203 = !DILocation(line: 8, column: 1, scope: !196)
// CHECK:STDOUT: !204 = !{!197}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.55456748a8a2eb70.Core", scope: null, file: !192, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !212)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !39)
// CHECK:STDOUT: !210 = !DILocation(line: 15, column: 38, scope: !205)
// CHECK:STDOUT: !211 = !DILocation(line: 15, column: 31, scope: !205)
// CHECK:STDOUT: !212 = !{!206}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !192, line: 19, type: !195, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !221)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !39)
// CHECK:STDOUT: !217 = !DILocation(line: 20, column: 16, scope: !213)
// CHECK:STDOUT: !220 = !DILocation(line: 19, column: 1, scope: !213)
// CHECK:STDOUT: !221 = !{!214}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.55456748a8a2eb70", scope: null, file: !192, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !229)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !6)
// CHECK:STDOUT: !228 = !DILocation(line: 5, column: 3, scope: !222)
// CHECK:STDOUT: !229 = !{!223}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !192, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !237)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !39)
// CHECK:STDOUT: !235 = !DILocation(line: 5, column: 10, scope: !230)
// CHECK:STDOUT: !236 = !DILocation(line: 5, column: 3, scope: !230)
// CHECK:STDOUT: !237 = !{!231}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.0f3e15a21cc8e963(i32 %n), !dbg !244
// CHECK:STDOUT:   ret void, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !252
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !252
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.9f9f842f4c8e7f62(ptr %c), !dbg !259
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.0f3e15a21cc8e963(i32 %x) #0 !dbg !262 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !268
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.9f9f842f4c8e7f62(ptr %x) #0 !dbg !270 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %x), !dbg !273
// CHECK:STDOUT:   ret void, !dbg !276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!239}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!6, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !105 = !{null, !6}
// CHECK:STDOUT: !106 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !194 = !{null, !39}
// CHECK:STDOUT: !195 = !DISubroutineType(types: !194)
// CHECK:STDOUT: !238 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !239 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !238, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !238, line: 8, type: !106, spFlags: DISPFlagDefinition, unit: !239, retainedNodes: !246)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 1, scope: !240, type: !6)
// CHECK:STDOUT: !244 = !DILocation(line: 9, column: 3, scope: !240)
// CHECK:STDOUT: !245 = !DILocation(line: 8, column: 1, scope: !240)
// CHECK:STDOUT: !246 = !{!241}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.55456748a8a2eb70.Core", scope: null, file: !238, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !239, retainedNodes: !254)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !39)
// CHECK:STDOUT: !252 = !DILocation(line: 15, column: 38, scope: !247)
// CHECK:STDOUT: !253 = !DILocation(line: 15, column: 31, scope: !247)
// CHECK:STDOUT: !254 = !{!248}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !238, line: 19, type: !195, spFlags: DISPFlagDefinition, unit: !239, retainedNodes: !261)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !39)
// CHECK:STDOUT: !259 = !DILocation(line: 20, column: 3, scope: !255)
// CHECK:STDOUT: !260 = !DILocation(line: 19, column: 1, scope: !255)
// CHECK:STDOUT: !261 = !{!256}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.0f3e15a21cc8e963", scope: null, file: !238, line: 4, type: !106, spFlags: DISPFlagDefinition, unit: !239, retainedNodes: !269)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !6)
// CHECK:STDOUT: !268 = !DILocation(line: 4, column: 1, scope: !262)
// CHECK:STDOUT: !269 = !{!263}
// CHECK:STDOUT: !270 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.9f9f842f4c8e7f62", scope: null, file: !238, line: 4, type: !195, spFlags: DISPFlagDefinition, unit: !239, retainedNodes: !277)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 1, scope: !270, type: !39)
// CHECK:STDOUT: !273 = !DILocation(line: 5, column: 14, scope: !270)
// CHECK:STDOUT: !276 = !DILocation(line: 4, column: 1, scope: !270)
// CHECK:STDOUT: !277 = !{!271}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_template_method_call.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_template_method_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @B.val.loc6_14 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeB.B.Main(ptr sret({}) %return) #0 !dbg !282 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc6_14, i64 0, i1 false), !dbg !285
// CHECK:STDOUT:   ret void, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !286 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_10.1.temp = alloca { {} }, align 1, !dbg !287
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_10.1.temp), !dbg !287
// CHECK:STDOUT:   call void @_CMakeA.Main.a6a60dfee2d7d273(ptr %.loc21_10.1.temp), !dbg !287
// CHECK:STDOUT:   ret i32 0, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMakeA.Main.a6a60dfee2d7d273(ptr sret({ {} }) %return) #0 !dbg !289 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.f4aa6f.1.field = getelementptr inbounds nuw { {} }, ptr %return, i32 0, i32 0, !dbg !293
// CHECK:STDOUT:   call void @_CMakeB.B.Main(ptr %.f4aa6f.1.field), !dbg !292
// CHECK:STDOUT:   ret void, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!279}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !118 = !{!39}
// CHECK:STDOUT: !119 = !DISubroutineType(types: !118)
// CHECK:STDOUT: !278 = !DIFile(filename: "initialize_from_template_method_call.carbon", directory: "")
// CHECK:STDOUT: !279 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !278, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !280 = !{!6}
// CHECK:STDOUT: !281 = !DISubroutineType(types: !280)
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "MakeB", linkageName: "_CMakeB.B.Main", scope: null, file: !278, line: 5, type: !119, spFlags: DISPFlagDefinition, unit: !279)
// CHECK:STDOUT: !285 = !DILocation(line: 6, column: 5, scope: !282)
// CHECK:STDOUT: !286 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !278, line: 20, type: !281, spFlags: DISPFlagDefinition, unit: !279)
// CHECK:STDOUT: !287 = !DILocation(line: 21, column: 3, scope: !286)
// CHECK:STDOUT: !288 = !DILocation(line: 20, column: 1, scope: !286)
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.a6a60dfee2d7d273", scope: null, file: !278, line: 14, type: !119, spFlags: DISPFlagDefinition, unit: !279)
// CHECK:STDOUT: !292 = !DILocation(line: 16, column: 15, scope: !289)
// CHECK:STDOUT: !293 = !DILocation(line: 15, column: 11, scope: !289)
// CHECK:STDOUT: !294 = !DILocation(line: 15, column: 4, scope: !289)
// CHECK:STDOUT:
