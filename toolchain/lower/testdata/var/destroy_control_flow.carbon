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
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Destructible.Main:Destroy.Core"(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "destructible.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core", scope: null, file: !32, line: 7, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !41)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 7, column: 5, scope: !37)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_then.carbon'
// CHECK:STDOUT: source_filename = "if_then.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfThen.Main(i1 %cond) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !47
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !47
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !51
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !54
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !54
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !54
// CHECK:STDOUT:   br label %if.else, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !62
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %a.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.5666652c25a63fe7, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !42 = !DIFile(filename: "if_then.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "IfThen", linkageName: "_CIfThen.Main", scope: null, file: !42, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !64)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !34)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 7, column: 3, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 8, column: 6, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 9, column: 5, scope: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 10, column: 5, scope: !44)
// CHECK:STDOUT: !59 = !DILocation(line: 8, column: 3, scope: !44)
// CHECK:STDOUT: !62 = !DILocation(line: 12, column: 3, scope: !44)
// CHECK:STDOUT: !63 = !DILocation(line: 5, column: 1, scope: !44)
// CHECK:STDOUT: !64 = !{!45}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !42, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !34)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 5, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5666652c25a63fe7", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !34)
// CHECK:STDOUT: !73 = !DILocation(line: 26, column: 5, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 24, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !81
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !88
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !81
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !81
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !85
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !87
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !88
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !88
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !92
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !88
// CHECK:STDOUT:   br label %if.done, !dbg !93
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !94
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !94
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %c.var), !dbg !94
// CHECK:STDOUT:   br label %if.done, !dbg !93
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !101
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %a.var), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.5666652c25a63fe7, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!77}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !76 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !76, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !76, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !103)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !34)
// CHECK:STDOUT: !81 = !DILocation(line: 6, column: 3, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 7, column: 3, scope: !78)
// CHECK:STDOUT: !87 = !DILocation(line: 8, column: 6, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 9, column: 5, scope: !78)
// CHECK:STDOUT: !92 = !DILocation(line: 10, column: 5, scope: !78)
// CHECK:STDOUT: !93 = !DILocation(line: 8, column: 3, scope: !78)
// CHECK:STDOUT: !94 = !DILocation(line: 12, column: 5, scope: !78)
// CHECK:STDOUT: !98 = !DILocation(line: 13, column: 5, scope: !78)
// CHECK:STDOUT: !101 = !DILocation(line: 15, column: 3, scope: !78)
// CHECK:STDOUT: !102 = !DILocation(line: 5, column: 1, scope: !78)
// CHECK:STDOUT: !103 = !{!79}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !76, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !34)
// CHECK:STDOUT: !106 = !DILocation(line: 9, column: 5, scope: !104)
// CHECK:STDOUT: !107 = !{!105}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5666652c25a63fe7", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !114)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !34)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 5, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 24, column: 3, scope: !108)
// CHECK:STDOUT: !114 = !{!109}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !120
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !127
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !120
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !120
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !124
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !126
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !127
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !127
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !131
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !127
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %a.var), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %a.var), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !135 {
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.5666652c25a63fe7, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!116}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !115 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !116 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !115, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !115, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !134)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !34)
// CHECK:STDOUT: !120 = !DILocation(line: 6, column: 3, scope: !117)
// CHECK:STDOUT: !124 = !DILocation(line: 7, column: 3, scope: !117)
// CHECK:STDOUT: !126 = !DILocation(line: 8, column: 6, scope: !117)
// CHECK:STDOUT: !127 = !DILocation(line: 9, column: 5, scope: !117)
// CHECK:STDOUT: !131 = !DILocation(line: 10, column: 5, scope: !117)
// CHECK:STDOUT: !132 = !DILocation(line: 11, column: 5, scope: !117)
// CHECK:STDOUT: !133 = !DILocation(line: 5, column: 1, scope: !117)
// CHECK:STDOUT: !134 = !{!118}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !115, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !34)
// CHECK:STDOUT: !137 = !DILocation(line: 9, column: 5, scope: !135)
// CHECK:STDOUT: !138 = !{!136}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5666652c25a63fe7", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !145)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !34)
// CHECK:STDOUT: !143 = !DILocation(line: 26, column: 5, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 24, column: 3, scope: !139)
// CHECK:STDOUT: !145 = !{!140}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !151
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !161
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !168
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !151
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !155
// CHECK:STDOUT:   br label %while.cond, !dbg !156
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !156
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !161
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !161
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !165
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !167
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !168
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !168
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !172
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %c.var), !dbg !168
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !161
// CHECK:STDOUT:   br label %while.done, !dbg !173
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !161
// CHECK:STDOUT:   br label %while.cond, !dbg !174
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !177
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %a.var), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %self) #0 !dbg !184 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.5666652c25a63fe7, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!147}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !146 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !147 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !146, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !146, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !179)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !34)
// CHECK:STDOUT: !151 = !DILocation(line: 6, column: 3, scope: !148)
// CHECK:STDOUT: !155 = !DILocation(line: 7, column: 3, scope: !148)
// CHECK:STDOUT: !156 = !DILocation(line: 8, column: 9, scope: !148)
// CHECK:STDOUT: !160 = !DILocation(line: 9, column: 5, scope: !148)
// CHECK:STDOUT: !161 = !DILocation(line: 10, column: 5, scope: !148)
// CHECK:STDOUT: !165 = !DILocation(line: 11, column: 5, scope: !148)
// CHECK:STDOUT: !167 = !DILocation(line: 12, column: 8, scope: !148)
// CHECK:STDOUT: !168 = !DILocation(line: 13, column: 7, scope: !148)
// CHECK:STDOUT: !172 = !DILocation(line: 14, column: 7, scope: !148)
// CHECK:STDOUT: !173 = !DILocation(line: 15, column: 7, scope: !148)
// CHECK:STDOUT: !174 = !DILocation(line: 8, column: 3, scope: !148)
// CHECK:STDOUT: !177 = !DILocation(line: 18, column: 3, scope: !148)
// CHECK:STDOUT: !178 = !DILocation(line: 5, column: 1, scope: !148)
// CHECK:STDOUT: !179 = !{!149}
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !146, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !183)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !180, type: !34)
// CHECK:STDOUT: !182 = !DILocation(line: 13, column: 7, scope: !180)
// CHECK:STDOUT: !183 = !{!181}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5666652c25a63fe7", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !190)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !34)
// CHECK:STDOUT: !188 = !DILocation(line: 26, column: 5, scope: !184)
// CHECK:STDOUT: !189 = !DILocation(line: 24, column: 3, scope: !184)
// CHECK:STDOUT: !190 = !{!185}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !193 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !196
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !206
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !196
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !196
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !200
// CHECK:STDOUT:   br label %while.cond, !dbg !201
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !201
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !205
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !206
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !206
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !210
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !212
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !213
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !213
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !217
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %c.var), !dbg !213
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !206
// CHECK:STDOUT:   br label %while.cond, !dbg !218
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %b.var), !dbg !206
// CHECK:STDOUT:   br label %while.cond, !dbg !219
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !222
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %a.var), !dbg !196
// CHECK:STDOUT:   ret void, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5666652c25a63fe7(ptr %self) #0 !dbg !229 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.5666652c25a63fe7, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!192}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !191 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !192 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !191, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !191, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !192, retainedNodes: !224)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !34)
// CHECK:STDOUT: !196 = !DILocation(line: 6, column: 3, scope: !193)
// CHECK:STDOUT: !200 = !DILocation(line: 7, column: 3, scope: !193)
// CHECK:STDOUT: !201 = !DILocation(line: 8, column: 9, scope: !193)
// CHECK:STDOUT: !205 = !DILocation(line: 9, column: 5, scope: !193)
// CHECK:STDOUT: !206 = !DILocation(line: 10, column: 5, scope: !193)
// CHECK:STDOUT: !210 = !DILocation(line: 11, column: 5, scope: !193)
// CHECK:STDOUT: !212 = !DILocation(line: 12, column: 8, scope: !193)
// CHECK:STDOUT: !213 = !DILocation(line: 13, column: 7, scope: !193)
// CHECK:STDOUT: !217 = !DILocation(line: 14, column: 7, scope: !193)
// CHECK:STDOUT: !218 = !DILocation(line: 15, column: 7, scope: !193)
// CHECK:STDOUT: !219 = !DILocation(line: 8, column: 3, scope: !193)
// CHECK:STDOUT: !222 = !DILocation(line: 18, column: 3, scope: !193)
// CHECK:STDOUT: !223 = !DILocation(line: 5, column: 1, scope: !193)
// CHECK:STDOUT: !224 = !{!194}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !191, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !192, retainedNodes: !228)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !34)
// CHECK:STDOUT: !227 = !DILocation(line: 13, column: 7, scope: !225)
// CHECK:STDOUT: !228 = !{!226}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5666652c25a63fe7", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !192, retainedNodes: !235)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !34)
// CHECK:STDOUT: !233 = !DILocation(line: 26, column: 5, scope: !229)
// CHECK:STDOUT: !234 = !DILocation(line: 24, column: 3, scope: !229)
// CHECK:STDOUT: !235 = !{!230}
// CHECK:STDOUT:
