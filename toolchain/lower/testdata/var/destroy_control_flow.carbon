// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/destroy_control_flow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/destroy_control_flow.carbon

// --- destructible.carbon

library "[[@TEST_NAME]]";

class Destructible {
  // TODO: Implement Core.Destructor instead.
  impl as Core.Destroy {
    fn Op(self);
  }
};

fn Use(ref d: Destructible);

// --- if_then.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfThen(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  }
  Use(ref a);
}

// --- if_else.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfElse(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  } else {
    var c: Destructible = {};
    Use(ref c);
  }
  Use(ref a);
}

// --- if_return.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfReturn(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
    return;
  }
}

// --- while_break.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn WhileBreak(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      break;
    }
  }
  Use(ref a);
}

// --- while_continue.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn WhileContinue(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      continue;
    }
  }
  Use(ref a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'destructible.carbon'
// CHECK:STDOUT: source_filename = "destructible.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Destructible.Main:Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Destructible.Main:Destroy.Core"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "destructible.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core", scope: null, file: !34, line: 7, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !43)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 7, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !{!40}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_then.carbon'
// CHECK:STDOUT: source_filename = "if_then.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfThen.Main(i1 %cond) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !49
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !53
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !56
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !56
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !60
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !56
// CHECK:STDOUT:   br label %if.else, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!45}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !44 = !DIFile(filename: "if_then.carbon", directory: "")
// CHECK:STDOUT: !45 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !44, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "IfThen", linkageName: "_CIfThen.Main", scope: null, file: !44, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !66)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !36)
// CHECK:STDOUT: !49 = !DILocation(line: 6, column: 3, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 7, column: 3, scope: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 8, column: 6, scope: !46)
// CHECK:STDOUT: !56 = !DILocation(line: 9, column: 5, scope: !46)
// CHECK:STDOUT: !60 = !DILocation(line: 10, column: 5, scope: !46)
// CHECK:STDOUT: !61 = !DILocation(line: 8, column: 3, scope: !46)
// CHECK:STDOUT: !64 = !DILocation(line: 12, column: 3, scope: !46)
// CHECK:STDOUT: !65 = !DILocation(line: 5, column: 1, scope: !46)
// CHECK:STDOUT: !66 = !{!47}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !44, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !36)
// CHECK:STDOUT: !69 = !DILocation(line: 9, column: 5, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !44, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !36)
// CHECK:STDOUT: !73 = !DILocation(line: 9, column: 5, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !87
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !80
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !80
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !84
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !87
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !87
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !91
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !87
// CHECK:STDOUT:   br label %if.done, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !93
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !93
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !97
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %c.var), !dbg !93
// CHECK:STDOUT:   br label %if.done, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !100
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !75 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !75, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !102)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !36)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 3, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 3, scope: !77)
// CHECK:STDOUT: !86 = !DILocation(line: 8, column: 6, scope: !77)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 5, scope: !77)
// CHECK:STDOUT: !91 = !DILocation(line: 10, column: 5, scope: !77)
// CHECK:STDOUT: !92 = !DILocation(line: 8, column: 3, scope: !77)
// CHECK:STDOUT: !93 = !DILocation(line: 12, column: 5, scope: !77)
// CHECK:STDOUT: !97 = !DILocation(line: 13, column: 5, scope: !77)
// CHECK:STDOUT: !100 = !DILocation(line: 15, column: 3, scope: !77)
// CHECK:STDOUT: !101 = !DILocation(line: 5, column: 1, scope: !77)
// CHECK:STDOUT: !102 = !{!78}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !75, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !36)
// CHECK:STDOUT: !105 = !DILocation(line: 9, column: 5, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !75, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !36)
// CHECK:STDOUT: !109 = !DILocation(line: 9, column: 5, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !116
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !123
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !116
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !116
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !120
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !122
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !123
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !123
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !127
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !123
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!112}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !111 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !112 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !111, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !111, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !130)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !36)
// CHECK:STDOUT: !116 = !DILocation(line: 6, column: 3, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 7, column: 3, scope: !113)
// CHECK:STDOUT: !122 = !DILocation(line: 8, column: 6, scope: !113)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 5, scope: !113)
// CHECK:STDOUT: !127 = !DILocation(line: 10, column: 5, scope: !113)
// CHECK:STDOUT: !128 = !DILocation(line: 11, column: 5, scope: !113)
// CHECK:STDOUT: !129 = !DILocation(line: 5, column: 1, scope: !113)
// CHECK:STDOUT: !130 = !{!114}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !111, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !36)
// CHECK:STDOUT: !133 = !DILocation(line: 9, column: 5, scope: !131)
// CHECK:STDOUT: !134 = !{!132}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !111, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !36)
// CHECK:STDOUT: !137 = !DILocation(line: 9, column: 5, scope: !135)
// CHECK:STDOUT: !138 = !{!136}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !144
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !154
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !144
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !144
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !148
// CHECK:STDOUT:   br label %while.cond, !dbg !149
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !149
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !154
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !154
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !158
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !160
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !161
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !161
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !165
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %c.var), !dbg !161
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !154
// CHECK:STDOUT:   br label %while.done, !dbg !166
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !154
// CHECK:STDOUT:   br label %while.cond, !dbg !167
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !170
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!140}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !139 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !139, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !139, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !172)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !36)
// CHECK:STDOUT: !144 = !DILocation(line: 6, column: 3, scope: !141)
// CHECK:STDOUT: !148 = !DILocation(line: 7, column: 3, scope: !141)
// CHECK:STDOUT: !149 = !DILocation(line: 8, column: 9, scope: !141)
// CHECK:STDOUT: !153 = !DILocation(line: 9, column: 5, scope: !141)
// CHECK:STDOUT: !154 = !DILocation(line: 10, column: 5, scope: !141)
// CHECK:STDOUT: !158 = !DILocation(line: 11, column: 5, scope: !141)
// CHECK:STDOUT: !160 = !DILocation(line: 12, column: 8, scope: !141)
// CHECK:STDOUT: !161 = !DILocation(line: 13, column: 7, scope: !141)
// CHECK:STDOUT: !165 = !DILocation(line: 14, column: 7, scope: !141)
// CHECK:STDOUT: !166 = !DILocation(line: 15, column: 7, scope: !141)
// CHECK:STDOUT: !167 = !DILocation(line: 8, column: 3, scope: !141)
// CHECK:STDOUT: !170 = !DILocation(line: 18, column: 3, scope: !141)
// CHECK:STDOUT: !171 = !DILocation(line: 5, column: 1, scope: !141)
// CHECK:STDOUT: !172 = !{!142}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !139, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !176)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !36)
// CHECK:STDOUT: !175 = !DILocation(line: 13, column: 7, scope: !173)
// CHECK:STDOUT: !176 = !{!174}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !139, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !180)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !36)
// CHECK:STDOUT: !179 = !DILocation(line: 13, column: 7, scope: !177)
// CHECK:STDOUT: !180 = !{!178}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !186
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !196
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !203
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !186
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !186
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !190
// CHECK:STDOUT:   br label %while.cond, !dbg !191
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !191
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !196
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !196
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !200
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !202
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !203
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !203
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !207
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %c.var), !dbg !203
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !196
// CHECK:STDOUT:   br label %while.cond, !dbg !208
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !196
// CHECK:STDOUT:   br label %while.cond, !dbg !209
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !212
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!182}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !181 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !182 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !181, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !181, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !182, retainedNodes: !214)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !36)
// CHECK:STDOUT: !186 = !DILocation(line: 6, column: 3, scope: !183)
// CHECK:STDOUT: !190 = !DILocation(line: 7, column: 3, scope: !183)
// CHECK:STDOUT: !191 = !DILocation(line: 8, column: 9, scope: !183)
// CHECK:STDOUT: !195 = !DILocation(line: 9, column: 5, scope: !183)
// CHECK:STDOUT: !196 = !DILocation(line: 10, column: 5, scope: !183)
// CHECK:STDOUT: !200 = !DILocation(line: 11, column: 5, scope: !183)
// CHECK:STDOUT: !202 = !DILocation(line: 12, column: 8, scope: !183)
// CHECK:STDOUT: !203 = !DILocation(line: 13, column: 7, scope: !183)
// CHECK:STDOUT: !207 = !DILocation(line: 14, column: 7, scope: !183)
// CHECK:STDOUT: !208 = !DILocation(line: 15, column: 7, scope: !183)
// CHECK:STDOUT: !209 = !DILocation(line: 8, column: 3, scope: !183)
// CHECK:STDOUT: !212 = !DILocation(line: 18, column: 3, scope: !183)
// CHECK:STDOUT: !213 = !DILocation(line: 5, column: 1, scope: !183)
// CHECK:STDOUT: !214 = !{!184}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !181, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !182, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !36)
// CHECK:STDOUT: !217 = !DILocation(line: 13, column: 7, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !181, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !182, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !36)
// CHECK:STDOUT: !221 = !DILocation(line: 13, column: 7, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT:
