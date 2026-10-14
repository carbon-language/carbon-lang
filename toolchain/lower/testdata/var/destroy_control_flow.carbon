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
  impl as Core.DestroyImpl {
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
// CHECK:STDOUT: declare void @"_COp.Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.1421ed8cb8ddfaad.Core:Destructible.Main:DestroyImpl.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Destructible.Main:DestroyImpl.Core"(ptr %self), !dbg !42
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
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.1421ed8cb8ddfaad.Core:Destructible.Main:DestroyImpl.Core", scope: null, file: !34, line: 6, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !43)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 6, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !{!40}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_then.carbon'
// CHECK:STDOUT: source_filename = "if_then.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfThen.Main(i1 %cond) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !49
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !53
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !56
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !56
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !56
// CHECK:STDOUT:   br label %if.else, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !64
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %a.var), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %t) #0 !dbg !67 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f8de890abec9a19e, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!45}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8de890abec9a19e", scope: null, file: !28, line: 30, type: !38, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !36)
// CHECK:STDOUT: !70 = !DILocation(line: 31, column: 5, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 30, column: 1, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !79
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !86
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !79
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !79
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !83
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !85
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !86
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !86
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !86
// CHECK:STDOUT:   br label %if.done, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !92
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !92
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !96
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %c.var), !dbg !92
// CHECK:STDOUT:   br label %if.done, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %a.var), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %t) #0 !dbg !102 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f8de890abec9a19e, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!75}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !74 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !75 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !74, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !74, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !101)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !36)
// CHECK:STDOUT: !79 = !DILocation(line: 6, column: 3, scope: !76)
// CHECK:STDOUT: !83 = !DILocation(line: 7, column: 3, scope: !76)
// CHECK:STDOUT: !85 = !DILocation(line: 8, column: 6, scope: !76)
// CHECK:STDOUT: !86 = !DILocation(line: 9, column: 5, scope: !76)
// CHECK:STDOUT: !90 = !DILocation(line: 10, column: 5, scope: !76)
// CHECK:STDOUT: !91 = !DILocation(line: 8, column: 3, scope: !76)
// CHECK:STDOUT: !92 = !DILocation(line: 12, column: 5, scope: !76)
// CHECK:STDOUT: !96 = !DILocation(line: 13, column: 5, scope: !76)
// CHECK:STDOUT: !99 = !DILocation(line: 15, column: 3, scope: !76)
// CHECK:STDOUT: !100 = !DILocation(line: 5, column: 1, scope: !76)
// CHECK:STDOUT: !101 = !{!77}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8de890abec9a19e", scope: null, file: !28, line: 30, type: !38, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !36)
// CHECK:STDOUT: !105 = !DILocation(line: 31, column: 5, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 30, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !114
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !121
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !114
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !114
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !118
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !120
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !121
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !121
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !125
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !121
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %a.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %a.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %t) #0 !dbg !129 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f8de890abec9a19e, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!110}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !109 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !110 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !109, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !109, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !128)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !36)
// CHECK:STDOUT: !114 = !DILocation(line: 6, column: 3, scope: !111)
// CHECK:STDOUT: !118 = !DILocation(line: 7, column: 3, scope: !111)
// CHECK:STDOUT: !120 = !DILocation(line: 8, column: 6, scope: !111)
// CHECK:STDOUT: !121 = !DILocation(line: 9, column: 5, scope: !111)
// CHECK:STDOUT: !125 = !DILocation(line: 10, column: 5, scope: !111)
// CHECK:STDOUT: !126 = !DILocation(line: 11, column: 5, scope: !111)
// CHECK:STDOUT: !127 = !DILocation(line: 5, column: 1, scope: !111)
// CHECK:STDOUT: !128 = !{!112}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8de890abec9a19e", scope: null, file: !28, line: 30, type: !38, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !135)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !36)
// CHECK:STDOUT: !132 = !DILocation(line: 31, column: 5, scope: !129)
// CHECK:STDOUT: !134 = !DILocation(line: 30, column: 1, scope: !129)
// CHECK:STDOUT: !135 = !{!130}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !141
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !151
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !158
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !141
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !141
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !145
// CHECK:STDOUT:   br label %while.cond, !dbg !146
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !146
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !151
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !155
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !157
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !158
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !158
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !162
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %c.var), !dbg !158
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !151
// CHECK:STDOUT:   br label %while.done, !dbg !163
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !151
// CHECK:STDOUT:   br label %while.cond, !dbg !164
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !167
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %a.var), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %t) #0 !dbg !170 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f8de890abec9a19e, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!137}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !136 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !137 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !136, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !136, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !137, retainedNodes: !169)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !36)
// CHECK:STDOUT: !141 = !DILocation(line: 6, column: 3, scope: !138)
// CHECK:STDOUT: !145 = !DILocation(line: 7, column: 3, scope: !138)
// CHECK:STDOUT: !146 = !DILocation(line: 8, column: 9, scope: !138)
// CHECK:STDOUT: !150 = !DILocation(line: 9, column: 5, scope: !138)
// CHECK:STDOUT: !151 = !DILocation(line: 10, column: 5, scope: !138)
// CHECK:STDOUT: !155 = !DILocation(line: 11, column: 5, scope: !138)
// CHECK:STDOUT: !157 = !DILocation(line: 12, column: 8, scope: !138)
// CHECK:STDOUT: !158 = !DILocation(line: 13, column: 7, scope: !138)
// CHECK:STDOUT: !162 = !DILocation(line: 14, column: 7, scope: !138)
// CHECK:STDOUT: !163 = !DILocation(line: 15, column: 7, scope: !138)
// CHECK:STDOUT: !164 = !DILocation(line: 8, column: 3, scope: !138)
// CHECK:STDOUT: !167 = !DILocation(line: 18, column: 3, scope: !138)
// CHECK:STDOUT: !168 = !DILocation(line: 5, column: 1, scope: !138)
// CHECK:STDOUT: !169 = !{!139}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8de890abec9a19e", scope: null, file: !28, line: 30, type: !38, spFlags: DISPFlagDefinition, unit: !137, retainedNodes: !176)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !36)
// CHECK:STDOUT: !173 = !DILocation(line: 31, column: 5, scope: !170)
// CHECK:STDOUT: !175 = !DILocation(line: 30, column: 1, scope: !170)
// CHECK:STDOUT: !176 = !{!171}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !182
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !192
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !199
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !182
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !182
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !186
// CHECK:STDOUT:   br label %while.cond, !dbg !187
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !187
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !191
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !192
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !192
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !196
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !198
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !199
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !199
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !203
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %c.var), !dbg !199
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !192
// CHECK:STDOUT:   br label %while.cond, !dbg !204
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %b.var), !dbg !192
// CHECK:STDOUT:   br label %while.cond, !dbg !205
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !208
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %a.var), !dbg !182
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8de890abec9a19e(ptr %t) #0 !dbg !211 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !214
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f8de890abec9a19e, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!178}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !177 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !178 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !177, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !177, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !178, retainedNodes: !210)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !36)
// CHECK:STDOUT: !182 = !DILocation(line: 6, column: 3, scope: !179)
// CHECK:STDOUT: !186 = !DILocation(line: 7, column: 3, scope: !179)
// CHECK:STDOUT: !187 = !DILocation(line: 8, column: 9, scope: !179)
// CHECK:STDOUT: !191 = !DILocation(line: 9, column: 5, scope: !179)
// CHECK:STDOUT: !192 = !DILocation(line: 10, column: 5, scope: !179)
// CHECK:STDOUT: !196 = !DILocation(line: 11, column: 5, scope: !179)
// CHECK:STDOUT: !198 = !DILocation(line: 12, column: 8, scope: !179)
// CHECK:STDOUT: !199 = !DILocation(line: 13, column: 7, scope: !179)
// CHECK:STDOUT: !203 = !DILocation(line: 14, column: 7, scope: !179)
// CHECK:STDOUT: !204 = !DILocation(line: 15, column: 7, scope: !179)
// CHECK:STDOUT: !205 = !DILocation(line: 8, column: 3, scope: !179)
// CHECK:STDOUT: !208 = !DILocation(line: 18, column: 3, scope: !179)
// CHECK:STDOUT: !209 = !DILocation(line: 5, column: 1, scope: !179)
// CHECK:STDOUT: !210 = !{!180}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8de890abec9a19e", scope: null, file: !28, line: 30, type: !38, spFlags: DISPFlagDefinition, unit: !178, retainedNodes: !217)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !36)
// CHECK:STDOUT: !214 = !DILocation(line: 31, column: 5, scope: !211)
// CHECK:STDOUT: !216 = !DILocation(line: 30, column: 1, scope: !211)
// CHECK:STDOUT: !217 = !{!212}
// CHECK:STDOUT:
