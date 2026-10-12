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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %y.var), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !32 {
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !38)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !37 = !DILocation(line: 26, column: 1, scope: !32)
// CHECK:STDOUT: !38 = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_form_param.carbon'
// CHECK:STDOUT: source_filename = "var_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !42
// CHECK:STDOUT:   store i32 0, ptr %x.var, align 4, !dbg !42
// CHECK:STDOUT:   call void @_CF.Main(ptr %x.var), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %x.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !45 {
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!40}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !39 = !DIFile(filename: "var_form_param.carbon", directory: "")
// CHECK:STDOUT: !40 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !39, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !39, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 3, column: 6, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 6, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 5, column: 1, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !22, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !51)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !20)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 1, scope: !45)
// CHECK:STDOUT: !51 = !{!46}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_form_param.carbon'
// CHECK:STDOUT: source_filename = "val_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 0), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!53}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !52 = !DIFile(filename: "val_form_param.carbon", directory: "")
// CHECK:STDOUT: !53 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !52, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !52, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 6, column: 3, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 5, column: 1, scope: !54)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_form.carbon'
// CHECK:STDOUT: source_filename = "ref_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!58}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !57 = !DIFile(filename: "ref_return_form.carbon", directory: "")
// CHECK:STDOUT: !58 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !57, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !57, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 6, column: 27, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 5, column: 1, scope: !62)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_return_form.carbon'
// CHECK:STDOUT: source_filename = "var_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !72
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !73
// CHECK:STDOUT:   store i32 %F.call, ptr %x.var, align 4, !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %x.var), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !77 {
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!68}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !67 = !DIFile(filename: "var_return_form.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !67, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !67, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 6, column: 14, scope: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 27, scope: !71)
// CHECK:STDOUT: !76 = !DILocation(line: 5, column: 1, scope: !71)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !22, spFlags: DISPFlagDefinition, unit: !68, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !20)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form.carbon'
// CHECK:STDOUT: source_filename = "val_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!85}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !84 = !DIFile(filename: "val_return_form.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !84, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !84, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !85)
// CHECK:STDOUT: !87 = !DILocation(line: 6, column: 23, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 5, column: 1, scope: !86)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form_definition.carbon'
// CHECK:STDOUT: source_filename = "val_return_form_definition.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !69 = !{!20}
// CHECK:STDOUT: !70 = !DISubroutineType(types: !69)
// CHECK:STDOUT: !91 = !DIFile(filename: "val_return_form_definition.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !91, line: 3, type: !70, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 4, column: 3, scope: !93)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !91, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 8, column: 23, scope: !97)
// CHECK:STDOUT: !101 = !DILocation(line: 7, column: 1, scope: !97)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form_class_definition.carbon'
// CHECK:STDOUT: source_filename = "val_return_form_class_definition.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %c, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!103}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !102 = !DIFile(filename: "val_return_form_class_definition.carbon", directory: "")
// CHECK:STDOUT: !103 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !102, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !104 = !{!59, !59}
// CHECK:STDOUT: !105 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !102, line: 6, type: !105, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !113)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !59)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 3, scope: !106)
// CHECK:STDOUT: !113 = !{!107}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_class.carbon'
// CHECK:STDOUT: source_filename = "ref_return_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!115}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !114 = !DIFile(filename: "ref_return_class.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !114, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !114, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !115)
// CHECK:STDOUT: !117 = !DILocation(line: 8, column: 3, scope: !116)
// CHECK:STDOUT: !118 = !DILocation(line: 7, column: 1, scope: !116)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_class.carbon'
// CHECK:STDOUT: source_filename = "val_return_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!120}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !119 = !DIFile(filename: "val_return_class.carbon", directory: "")
// CHECK:STDOUT: !120 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !119, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !119, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !120)
// CHECK:STDOUT: !122 = !DILocation(line: 8, column: 3, scope: !121)
// CHECK:STDOUT: !123 = !DILocation(line: 7, column: 1, scope: !121)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_class_generic.carbon'
// CHECK:STDOUT: source_filename = "ref_return_class_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CF.D.Main:I.Main"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_39.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !133
// CHECK:STDOUT:   %.loc16_39.2 = load ptr, ptr %.loc16_39.1.p, align 8, !dbg !133
// CHECK:STDOUT:   ret ptr %.loc16_39.2, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main(ptr %d) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.c3fdce5fe96ec8d5(ptr %d), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.c3fdce5fe96ec8d5(ptr %x) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %I.WithSelf.F.call = call ptr @"_CF.D.Main:I.Main"(ptr %x), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!125}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !104 = !{!59, !59}
// CHECK:STDOUT: !105 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !124 = !DIFile(filename: "ref_return_class_generic.carbon", directory: "")
// CHECK:STDOUT: !125 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !124, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !126 = !{null, !59}
// CHECK:STDOUT: !127 = !DISubroutineType(types: !126)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "F", linkageName: "_CF.D.Main:I.Main", scope: null, file: !124, line: 16, type: !105, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !136)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !59)
// CHECK:STDOUT: !133 = !DILocation(line: 16, column: 35, scope: !128)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 27, scope: !128)
// CHECK:STDOUT: !136 = !{!129}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !124, line: 20, type: !127, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !143)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !59)
// CHECK:STDOUT: !141 = !DILocation(line: 21, column: 3, scope: !137)
// CHECK:STDOUT: !142 = !DILocation(line: 20, column: 1, scope: !137)
// CHECK:STDOUT: !143 = !{!138}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3fdce5fe96ec8d5", scope: null, file: !124, line: 9, type: !127, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !149)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !59)
// CHECK:STDOUT: !147 = !DILocation(line: 10, column: 3, scope: !144)
// CHECK:STDOUT: !148 = !DILocation(line: 9, column: 1, scope: !144)
// CHECK:STDOUT: !149 = !{!145}
// CHECK:STDOUT:
