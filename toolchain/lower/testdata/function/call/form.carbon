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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca i32, align 4, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !26
// CHECK:STDOUT:   store i32 0, ptr %y.var, align 4, !dbg !26
// CHECK:STDOUT:   call void @_CF.Main(ptr %y.var), !dbg !30
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %y.var), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "ref_form_param.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 6, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 7, column: 3, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 1, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_form_param.carbon'
// CHECK:STDOUT: source_filename = "var_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !39
// CHECK:STDOUT:   store i32 0, ptr %x.var, align 4, !dbg !39
// CHECK:STDOUT:   call void @_CF.Main(ptr %x.var), !dbg !40
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %x.var), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !36 = !DIFile(filename: "var_form_param.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !36, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 3, column: 6, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 5, column: 1, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !36, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !20)
// CHECK:STDOUT: !44 = !DILocation(line: 3, column: 6, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_form_param.carbon'
// CHECK:STDOUT: source_filename = "val_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 0), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!47}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !46 = !DIFile(filename: "val_form_param.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !46, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !46, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 6, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 5, column: 1, scope: !48)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_form.carbon'
// CHECK:STDOUT: source_filename = "ref_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !51 = !DIFile(filename: "ref_return_form.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !51, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 27, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 5, column: 1, scope: !56)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_return_form.carbon'
// CHECK:STDOUT: source_filename = "var_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !66
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !67
// CHECK:STDOUT:   store i32 %F.call, ptr %x.var, align 4, !dbg !66
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %x.var), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !61 = !DIFile(filename: "var_return_form.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !61, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 6, column: 14, scope: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 6, column: 27, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 5, column: 1, scope: !65)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !61, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 14, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form.carbon'
// CHECK:STDOUT: source_filename = "val_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !75 = !DIFile(filename: "val_return_form.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !75, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !76)
// CHECK:STDOUT: !78 = !DILocation(line: 6, column: 23, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 5, column: 1, scope: !77)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form_definition.carbon'
// CHECK:STDOUT: source_filename = "val_return_form_definition.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !63 = !{!20}
// CHECK:STDOUT: !64 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !82 = !DIFile(filename: "val_return_form_definition.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !82, line: 3, type: !64, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 4, column: 3, scope: !84)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !82, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 8, column: 23, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 7, column: 1, scope: !88)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form_class_definition.carbon'
// CHECK:STDOUT: source_filename = "val_return_form_class_definition.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %c, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!94}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !53 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !93 = !DIFile(filename: "val_return_form_class_definition.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !93, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !95 = !{!53, !53}
// CHECK:STDOUT: !96 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !93, line: 6, type: !96, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !104)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !53)
// CHECK:STDOUT: !103 = !DILocation(line: 7, column: 3, scope: !97)
// CHECK:STDOUT: !104 = !{!98}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_class.carbon'
// CHECK:STDOUT: source_filename = "ref_return_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!106}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !105 = !DIFile(filename: "ref_return_class.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !105, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !105, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 8, column: 3, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 7, column: 1, scope: !107)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_class.carbon'
// CHECK:STDOUT: source_filename = "val_return_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!111}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !110 = !DIFile(filename: "val_return_class.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !110, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !110, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !111)
// CHECK:STDOUT: !113 = !DILocation(line: 8, column: 3, scope: !112)
// CHECK:STDOUT: !114 = !DILocation(line: 7, column: 1, scope: !112)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_class_generic.carbon'
// CHECK:STDOUT: source_filename = "ref_return_class_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CF.D.Main:I.Main"(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_39.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !124
// CHECK:STDOUT:   %.loc16_39.2 = load ptr, ptr %.loc16_39.1.p, align 8, !dbg !124
// CHECK:STDOUT:   ret ptr %.loc16_39.2, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main(ptr %d) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.c3fdce5fe96ec8d5(ptr %d), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.c3fdce5fe96ec8d5(ptr %x) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %I.WithSelf.F.call = call ptr @"_CF.D.Main:I.Main"(ptr %x), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!116}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !53 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !95 = !{!53, !53}
// CHECK:STDOUT: !96 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !115 = !DIFile(filename: "ref_return_class_generic.carbon", directory: "")
// CHECK:STDOUT: !116 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !115, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !117 = !{null, !53}
// CHECK:STDOUT: !118 = !DISubroutineType(types: !117)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "F", linkageName: "_CF.D.Main:I.Main", scope: null, file: !115, line: 16, type: !96, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !127)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !53)
// CHECK:STDOUT: !124 = !DILocation(line: 16, column: 35, scope: !119)
// CHECK:STDOUT: !126 = !DILocation(line: 16, column: 27, scope: !119)
// CHECK:STDOUT: !127 = !{!120}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !115, line: 20, type: !118, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !134)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !53)
// CHECK:STDOUT: !132 = !DILocation(line: 21, column: 3, scope: !128)
// CHECK:STDOUT: !133 = !DILocation(line: 20, column: 1, scope: !128)
// CHECK:STDOUT: !134 = !{!129}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3fdce5fe96ec8d5", scope: null, file: !115, line: 9, type: !118, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !140)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !53)
// CHECK:STDOUT: !138 = !DILocation(line: 10, column: 3, scope: !135)
// CHECK:STDOUT: !139 = !DILocation(line: 9, column: 1, scope: !135)
// CHECK:STDOUT: !140 = !{!136}
// CHECK:STDOUT:
