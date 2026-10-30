// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// Mach-O has a global symbol prefix (`_`), so Clang refers to Carbon functions
// exported to C++ using an `\01`-prefixed asm label name. Check that those
// references resolve to the Carbon definitions in the IR.
// EXTRA-ARGS: --target=arm64-apple-macosx14.0
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/macho_asm_label.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/macho_asm_label.carbon

// --- function.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F(i: i32) -> i32 { return i; }

inline Cpp '''
int G() {
  return Carbon::F(1);
}
''';

fn H() -> i32 { return Cpp.G(); }

// --- constructor.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {
  fn C(i: i32) -> Self {
    return {.i = i};
  }

  var i: i32;
}

inline Cpp '''
int G() {
  Carbon::C c(42);
  return c.i;
}
''';

fn H() -> i32 { return Cpp.G(); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'function.carbon'
// CHECK:STDOUT: source_filename = "function.carbon"
// CHECK:STDOUT: target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
// CHECK:STDOUT: target triple = "arm64-apple-macosx14.0.0"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress ssp uwtable(sync)
// CHECK:STDOUT: define noundef i32 @_Z1Gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6Carbon4MainL1FEi(i32 noundef 1)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind ssp uwtable(sync)
// CHECK:STDOUT: define internal noundef i32 @_ZN6Carbon4MainL1FEi(i32 noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !36
// CHECK:STDOUT:   call void @"\01__CF__carbon_thunk.Main"(ptr noundef nonnull align 4 dereferenceable(4) %.addr, ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %1 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(i32 %i) #2 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %i, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"\01__CF__carbon_thunk.Main"(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #3 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6 = load i32, ptr %_, align 4, !dbg !29
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(i32 %.loc6), !dbg !29
// CHECK:STDOUT:   store i32 %F.call, ptr %_1, align 4, !dbg !29
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CH.Main() #2 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_Z1Gv(), !dbg !34
// CHECK:STDOUT:   ret i32 %G.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { nounwind "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!10}
// CHECK:STDOUT: !llvm.module.flags = !{!37, !38, !39, !7, !8}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !7 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !8 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !DIFile(filename: "function.carbon", directory: "")
// CHECK:STDOUT: !10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !9, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !12 = !{!11, !11}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !11, !11}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{!11}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !9, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !25)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !11)
// CHECK:STDOUT: !24 = !DILocation(line: 6, column: 23, scope: !18)
// CHECK:STDOUT: !25 = !{!19}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Main", scope: null, file: !9, line: 6, type: !15, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !11)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !26, type: !11)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 1, scope: !26)
// CHECK:STDOUT: !30 = !{!27, !28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !9, line: 14, type: !17, spFlags: DISPFlagDefinition, unit: !10)
// CHECK:STDOUT: !34 = !DILocation(line: 14, column: 24, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 14, column: 17, scope: !31)
// CHECK:STDOUT: !36 = !{!2, !2, i64 0}
// CHECK:STDOUT: !37 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !38 = !{i32 7, !"uwtable", i32 1}
// CHECK:STDOUT: !39 = !{i32 7, !"frame-pointer", i32 4}
// CHECK:STDOUT: !40 = !{!"__libc_errno", !2, i64 0}
// CHECK:STDOUT: !41 = !{!40, !2, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'constructor.carbon'
// CHECK:STDOUT: source_filename = "constructor.carbon"
// CHECK:STDOUT: target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
// CHECK:STDOUT: target triple = "arm64-apple-macosx14.0.0"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::Main::C" = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress ssp uwtable(sync)
// CHECK:STDOUT: define noundef i32 @_Z1Gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c = alloca %"class.Carbon::Main::C", align 4
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c) #4
// CHECK:STDOUT:   %call = call noundef ptr @_ZN6Carbon4Main1CC1Ei(ptr noundef nonnull align 4 dereferenceable(4) %c, i32 noundef 42)
// CHECK:STDOUT:   %i = getelementptr inbounds nuw %"class.Carbon::Main::C", ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %i, align 4, !tbaa !4
// CHECK:STDOUT:   %call1 = call noundef ptr @_ZN6Carbon4Main1CD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %c) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %c) #4
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind ssp uwtable(sync)
// CHECK:STDOUT: define internal noundef ptr @_ZN6Carbon4Main1CC1Ei(ptr noundef nonnull returned align 4 dereferenceable(4) %this, i32 noundef %0) unnamed_addr #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !36
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %call = call noundef ptr @"\01__CC__carbon_thunk.C.Main"(ptr noundef nonnull align 4 dereferenceable(4) %this1, ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret ptr %this1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind ssp uwtable(sync)
// CHECK:STDOUT: define linkonce_odr noundef ptr @_ZN6Carbon4Main1CD1Ev(ptr noundef nonnull returned align 4 dead_on_return(4) dereferenceable(4) %this) unnamed_addr #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %call = call noundef ptr @_ZN6Carbon4Main1CD2Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %this1) #4
// CHECK:STDOUT:   ret ptr %this1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CC.C.Main(ptr sret({ i32 }) %return, i32 %i) #4 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.2.i = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   store i32 %i, ptr %.loc8_19.2.i, align 4, !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #5 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %self), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d424893893628d5b:core.SubobjectDestroy.Core"(ptr %self) #4 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_9.1.i = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_9.1.i), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.SubobjectDestroy.Core"(ptr %self) #4 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_9.1.i = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_9.1.i), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr noundef ptr @"\01__CC__carbon_thunk.C.Main"(ptr noundef nonnull returned align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) unnamed_addr #6 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_24.2 = load i32, ptr %_1, align 4, !dbg !77
// CHECK:STDOUT:   call void @_CC.C.Main(ptr %_, i32 %.loc7_24.2), !dbg !77
// CHECK:STDOUT:   ret ptr %_
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CH.Main() #4 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_Z1Gv(), !dbg !82
// CHECK:STDOUT:   ret i32 %G.call, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #4 !dbg !85 {
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.SubobjectDestroy.Core"(ptr %t), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #4 !dbg !92 {
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind ssp uwtable(sync)
// CHECK:STDOUT: define linkonce_odr noundef ptr @_ZN6Carbon4Main1CD2Ev(ptr noundef nonnull returned align 4 dead_on_return(4) dereferenceable(4) %this) unnamed_addr #3 personality ptr @__gxx_personality_v0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   invoke void @"\01__C__destroy_thunk:thunk.C.Main"(ptr noundef nonnull align 4 dereferenceable(4) %this1)
// CHECK:STDOUT:           to label %invoke.cont unwind label %terminate.lpad
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont:                                      ; preds = %entry
// CHECK:STDOUT:   ret ptr %this1
// CHECK:STDOUT:
// CHECK:STDOUT: terminate.lpad:                                   ; preds = %entry
// CHECK:STDOUT:   %0 = landingpad { ptr, i32 }
// CHECK:STDOUT:           catch ptr null
// CHECK:STDOUT:   %1 = extractvalue { ptr, i32 } %0, 0
// CHECK:STDOUT:   call void @__clang_call_terminate(ptr %1) #9
// CHECK:STDOUT:   unreachable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"\01__C__destroy_thunk:thunk.C.Main"(ptr noundef nonnull align 4 dereferenceable(4)) #7
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @__gxx_personality_v0(...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline noreturn nounwind ssp uwtable(sync)
// CHECK:STDOUT: define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) #8 {
// CHECK:STDOUT:   %2 = call ptr @__cxa_begin_catch(ptr %0) #4
// CHECK:STDOUT:   call void @_ZSt9terminatev() #9
// CHECK:STDOUT:   unreachable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @__cxa_begin_catch(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZSt9terminatev()
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #6 = { nounwind "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #7 = { "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #8 = { noinline noreturn nounwind ssp uwtable(sync) "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" "tune-cpu"="apple-m5" }
// CHECK:STDOUT: attributes #9 = { noreturn nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!37, !38, !39, !7, !8}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"_ZTSN6Carbon4Main1CE", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !2, i64 0}
// CHECK:STDOUT: !7 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !8 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!11}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !36 = !{!2, !2, i64 0}
// CHECK:STDOUT: !37 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !38 = !{i32 7, !"uwtable", i32 1}
// CHECK:STDOUT: !39 = !{i32 7, !"frame-pointer", i32 4}
// CHECK:STDOUT: !40 = !{!"__libc_errno", !2, i64 0}
// CHECK:STDOUT: !41 = !{!40, !2, i64 0}
// CHECK:STDOUT: !42 = !DIFile(filename: "constructor.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !45 = !{!44, !11}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !{null, !44}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !{null, !44, !11}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "C", linkageName: "_CC.C.Main", scope: null, file: !42, line: 7, type: !46, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !59)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !11)
// CHECK:STDOUT: !57 = !DILocation(line: 8, column: 12, scope: !51)
// CHECK:STDOUT: !58 = !DILocation(line: 8, column: 5, scope: !51)
// CHECK:STDOUT: !59 = !{!52}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !42, line: 6, type: !48, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !44)
// CHECK:STDOUT: !62 = !DILocation(line: 6, column: 1, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d424893893628d5b:core.SubobjectDestroy.Core", scope: null, file: !42, line: 6, type: !48, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !69)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !44)
// CHECK:STDOUT: !66 = !DILocation(line: 6, column: 1, scope: !64)
// CHECK:STDOUT: !67 = !{null, !11}
// CHECK:STDOUT: !68 = !DISubroutineType(types: !67)
// CHECK:STDOUT: !69 = !{!65}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.SubobjectDestroy.Core", scope: null, file: !42, line: 6, type: !48, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !44)
// CHECK:STDOUT: !72 = !DILocation(line: 6, column: 1, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "C__carbon_thunk", linkageName: "_CC__carbon_thunk.C.Main", scope: null, file: !42, line: 7, type: !50, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !78)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !44)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !74, type: !11)
// CHECK:STDOUT: !77 = !DILocation(line: 7, column: 3, scope: !74)
// CHECK:STDOUT: !78 = !{!75, !76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !42, line: 21, type: !17, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !82 = !DILocation(line: 21, column: 24, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 21, column: 17, scope: !79)
// CHECK:STDOUT: !84 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !84, line: 36, type: !48, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !91)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !44)
// CHECK:STDOUT: !89 = !DILocation(line: 38, column: 5, scope: !85)
// CHECK:STDOUT: !90 = !DILocation(line: 36, column: 1, scope: !85)
// CHECK:STDOUT: !91 = !{!86}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !84, line: 36, type: !68, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !98)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !11)
// CHECK:STDOUT: !97 = !DILocation(line: 36, column: 1, scope: !92)
// CHECK:STDOUT: !98 = !{!93}
// CHECK:STDOUT: !99 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !100 = !{!"p1 _ZTSN6Carbon4Main1CE", !99, i64 0}
// CHECK:STDOUT: !101 = !{!100, !100, i64 0}
// CHECK:STDOUT:
