// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/form.carbon

// --- ref_form_param.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(ref i32));

fn G() {
  var y: i32 = 0;
  F(ref y);
}

// --- var_form_param.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(var i32));

fn G() {
  F(0);
}

// --- val_form_param.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(val i32));

fn G() {
  F(0);
}

// --- ref_return_form.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(ref i32);

fn G() {
  let unused ref x: i32 = F();
}

// --- var_return_form.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(var i32);

fn G() {
  let unused var x: i32 = F();
}

// --- val_return_form.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(val i32);

fn G() {
  let unused x: i32 = F();
}

// --- val_return_form_definition.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(val i32) {
  return 1;
}

fn G() {
  let unused x: i32 = F();
}

// --- val_return_form_class_definition.carbon
library "[[@TEST_NAME]]";

class C { var a: i32; }

// Returns a pointer to the caller's argument.
fn F(c: C) ->? form(val C) {
  return c;
}

// --- ref_return_class.carbon
library "[[@TEST_NAME]]";

class C {}

fn F() -> ref C;

fn G() {
  F();
}

// --- val_return_class.carbon
library "[[@TEST_NAME]]";

class C {}

fn F() ->? form(val C);

fn G() {
  F();
}

// --- ref_return_class_generic.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {
  fn F(self) -> ref C;
}

fn G[T: I](x: T) {
  x.F();
}

class D {
  var p: C*;
  impl as I {
    fn F(self) -> ref C { return *self.p; }
  }
}

fn H(d: D) {
  G(d);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_form_param.carbon'
// CHECK:STDOUT: source_filename = "ref_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca i32, align 4, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !28
// CHECK:STDOUT:   store i32 0, ptr %y.var, align 4, !dbg !28
// CHECK:STDOUT:   call void @_CF.Main(ptr %y.var), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "ref_form_param.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 3, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 3, scope: !27)
// CHECK:STDOUT: !33 = !DILocation(line: 5, column: 1, scope: !27)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_form_param.carbon'
// CHECK:STDOUT: source_filename = "var_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !37
// CHECK:STDOUT:   store i32 0, ptr %x.var, align 4, !dbg !37
// CHECK:STDOUT:   call void @_CF.Main(ptr %x.var), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !34 = !DIFile(filename: "var_form_param.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !34, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 3, column: 6, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 6, column: 3, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 5, column: 1, scope: !36)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_form_param.carbon'
// CHECK:STDOUT: source_filename = "val_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 0), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !40 = !DIFile(filename: "val_form_param.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !40, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 6, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 5, column: 1, scope: !42)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_form.carbon'
// CHECK:STDOUT: source_filename = "ref_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!46}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !45 = !DIFile(filename: "ref_return_form.carbon", directory: "")
// CHECK:STDOUT: !46 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !45, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !45, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 6, column: 27, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 1, scope: !50)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_return_form.carbon'
// CHECK:STDOUT: source_filename = "var_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !60
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !61
// CHECK:STDOUT:   store i32 %F.call, ptr %x.var, align 4, !dbg !60
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!56}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !55 = !DIFile(filename: "var_return_form.carbon", directory: "")
// CHECK:STDOUT: !56 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !55, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !55, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 6, column: 14, scope: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 6, column: 27, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 5, column: 1, scope: !59)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form.carbon'
// CHECK:STDOUT: source_filename = "val_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !65 = !DIFile(filename: "val_return_form.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !65, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 6, column: 23, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 5, column: 1, scope: !67)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form_definition.carbon'
// CHECK:STDOUT: source_filename = "val_return_form_definition.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!73}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !57 = !{!22}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !72 = !DIFile(filename: "val_return_form_definition.carbon", directory: "")
// CHECK:STDOUT: !73 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !72, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !72, line: 3, type: !58, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 4, column: 3, scope: !74)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !72, line: 7, type: !26, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !79 = !DILocation(line: 8, column: 23, scope: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 7, column: 1, scope: !78)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form_class_definition.carbon'
// CHECK:STDOUT: source_filename = "val_return_form_class_definition.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %c, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !83 = !DIFile(filename: "val_return_form_class_definition.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = !{!47, !47}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !83, line: 6, type: !86, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !94)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !47)
// CHECK:STDOUT: !93 = !DILocation(line: 7, column: 3, scope: !87)
// CHECK:STDOUT: !94 = !{!88}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_class.carbon'
// CHECK:STDOUT: source_filename = "ref_return_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!96}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !95 = !DIFile(filename: "ref_return_class.carbon", directory: "")
// CHECK:STDOUT: !96 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !95, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !95, line: 7, type: !26, spFlags: DISPFlagDefinition, unit: !96)
// CHECK:STDOUT: !98 = !DILocation(line: 8, column: 3, scope: !97)
// CHECK:STDOUT: !99 = !DILocation(line: 7, column: 1, scope: !97)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_class.carbon'
// CHECK:STDOUT: source_filename = "val_return_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!101}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !100 = !DIFile(filename: "val_return_class.carbon", directory: "")
// CHECK:STDOUT: !101 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !100, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !100, line: 7, type: !26, spFlags: DISPFlagDefinition, unit: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 8, column: 3, scope: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 7, column: 1, scope: !102)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_class_generic.carbon'
// CHECK:STDOUT: source_filename = "ref_return_class_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CF.D.Main:I.Main"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_39.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !114
// CHECK:STDOUT:   %.loc16_39.2 = load ptr, ptr %.loc16_39.1.p, align 8, !dbg !114
// CHECK:STDOUT:   ret ptr %.loc16_39.2, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main(ptr %d) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.c3fdce5fe96ec8d5(ptr %d), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.c3fdce5fe96ec8d5(ptr %x) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %I.WithSelf.F.call = call ptr @"_CF.D.Main:I.Main"(ptr %x), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!106}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !85 = !{!47, !47}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !105 = !DIFile(filename: "ref_return_class_generic.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !105, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !107 = !{null, !47}
// CHECK:STDOUT: !108 = !DISubroutineType(types: !107)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "F", linkageName: "_CF.D.Main:I.Main", scope: null, file: !105, line: 16, type: !86, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !117)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !47)
// CHECK:STDOUT: !114 = !DILocation(line: 16, column: 35, scope: !109)
// CHECK:STDOUT: !116 = !DILocation(line: 16, column: 27, scope: !109)
// CHECK:STDOUT: !117 = !{!110}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !105, line: 20, type: !108, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !47)
// CHECK:STDOUT: !122 = !DILocation(line: 21, column: 3, scope: !118)
// CHECK:STDOUT: !123 = !DILocation(line: 20, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3fdce5fe96ec8d5", scope: null, file: !105, line: 9, type: !108, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !130)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !47)
// CHECK:STDOUT: !128 = !DILocation(line: 10, column: 3, scope: !125)
// CHECK:STDOUT: !129 = !DILocation(line: 9, column: 1, scope: !125)
// CHECK:STDOUT: !130 = !{!126}
// CHECK:STDOUT:
