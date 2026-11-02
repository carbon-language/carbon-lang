// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=speed foo.carbon --target=x86_64-unknown-linux-gnu --target-cpu=x86-64-v2 --target-cpu-tune=znver4 --target-cpu-features=+avx2,-popcnt --clang-arg=-mlzcnt --phase=optimize --dump-llvm-ir --exclude-dump-file-prefix=%{core}
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/target_cpu.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/target_cpu.carbon

// --- foo.carbon

import Cpp inline '''
#ifndef __AVX2__
#error "__AVX2__ should be defined via --target-cpu-features=+avx2"
#endif
#ifdef __POPCNT__
#error "__POPCNT__ should be disabled via --target-cpu-features=-popcnt"
#endif
#ifndef __LAHF_SAHF__
#error "__LAHF_SAHF__ should be defined via --target-cpu=x86-64-v2"
#endif
#ifndef __LZCNT__
#error "__LZCNT__ should be defined via --clang-arg=-mlzcnt"
#endif
''';

fn VectorizedWithAvx2(a: array(i32, 65536)*) {
  // With --target-cpu-features=+avx2, this loop should vectorize using
  // 256-bit (<8 x i32>) vectors instead of baseline 128-bit (<4 x i32>).
  var n: i32 = 0;
  while (n < 65536) {
    (*a)[n] *= 2;
    ++n;
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
// CHECK:STDOUT: define void @_CVectorizedWithAvx2.Main(ptr nofree captures(none) %a) local_unnamed_addr #0 !dbg !6 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %vector.body, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: vector.body:                                      ; preds = %vector.body, %entry
// CHECK:STDOUT:   %index = phi i64 [ 0, %entry ], [ %index.next.3, %vector.body ], !dbg !84
// CHECK:STDOUT:   %0 = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %index, !dbg !16
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw i8, ptr %0, i64 32, !dbg !16
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw i8, ptr %0, i64 64, !dbg !16
// CHECK:STDOUT:   %3 = getelementptr inbounds nuw i8, ptr %0, i64 96, !dbg !16
// CHECK:STDOUT:   %wide.load = load <8 x i32>, ptr %0, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load5 = load <8 x i32>, ptr %1, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load6 = load <8 x i32>, ptr %2, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load7 = load <8 x i32>, ptr %3, align 4, !dbg !16
// CHECK:STDOUT:   %4 = shl <8 x i32> %wide.load, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %5 = shl <8 x i32> %wide.load5, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %6 = shl <8 x i32> %wide.load6, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %7 = shl <8 x i32> %wide.load7, splat (i32 1), !dbg !16
// CHECK:STDOUT:   store <8 x i32> %4, ptr %0, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %5, ptr %1, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %6, ptr %2, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %7, ptr %3, align 4, !dbg !16
// CHECK:STDOUT:   %8 = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %index, !dbg !16
// CHECK:STDOUT:   %9 = getelementptr inbounds nuw i8, ptr %8, i64 128, !dbg !16
// CHECK:STDOUT:   %10 = getelementptr inbounds nuw i8, ptr %8, i64 160, !dbg !16
// CHECK:STDOUT:   %11 = getelementptr inbounds nuw i8, ptr %8, i64 192, !dbg !16
// CHECK:STDOUT:   %12 = getelementptr inbounds nuw i8, ptr %8, i64 224, !dbg !16
// CHECK:STDOUT:   %wide.load.1 = load <8 x i32>, ptr %9, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load5.1 = load <8 x i32>, ptr %10, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load6.1 = load <8 x i32>, ptr %11, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load7.1 = load <8 x i32>, ptr %12, align 4, !dbg !16
// CHECK:STDOUT:   %13 = shl <8 x i32> %wide.load.1, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %14 = shl <8 x i32> %wide.load5.1, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %15 = shl <8 x i32> %wide.load6.1, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %16 = shl <8 x i32> %wide.load7.1, splat (i32 1), !dbg !16
// CHECK:STDOUT:   store <8 x i32> %13, ptr %9, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %14, ptr %10, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %15, ptr %11, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %16, ptr %12, align 4, !dbg !16
// CHECK:STDOUT:   %17 = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %index, !dbg !16
// CHECK:STDOUT:   %18 = getelementptr inbounds nuw i8, ptr %17, i64 256, !dbg !16
// CHECK:STDOUT:   %19 = getelementptr inbounds nuw i8, ptr %17, i64 288, !dbg !16
// CHECK:STDOUT:   %20 = getelementptr inbounds nuw i8, ptr %17, i64 320, !dbg !16
// CHECK:STDOUT:   %21 = getelementptr inbounds nuw i8, ptr %17, i64 352, !dbg !16
// CHECK:STDOUT:   %wide.load.2 = load <8 x i32>, ptr %18, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load5.2 = load <8 x i32>, ptr %19, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load6.2 = load <8 x i32>, ptr %20, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load7.2 = load <8 x i32>, ptr %21, align 4, !dbg !16
// CHECK:STDOUT:   %22 = shl <8 x i32> %wide.load.2, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %23 = shl <8 x i32> %wide.load5.2, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %24 = shl <8 x i32> %wide.load6.2, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %25 = shl <8 x i32> %wide.load7.2, splat (i32 1), !dbg !16
// CHECK:STDOUT:   store <8 x i32> %22, ptr %18, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %23, ptr %19, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %24, ptr %20, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %25, ptr %21, align 4, !dbg !16
// CHECK:STDOUT:   %26 = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %index, !dbg !16
// CHECK:STDOUT:   %27 = getelementptr inbounds nuw i8, ptr %26, i64 384, !dbg !16
// CHECK:STDOUT:   %28 = getelementptr inbounds nuw i8, ptr %26, i64 416, !dbg !16
// CHECK:STDOUT:   %29 = getelementptr inbounds nuw i8, ptr %26, i64 448, !dbg !16
// CHECK:STDOUT:   %30 = getelementptr inbounds nuw i8, ptr %26, i64 480, !dbg !16
// CHECK:STDOUT:   %wide.load.3 = load <8 x i32>, ptr %27, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load5.3 = load <8 x i32>, ptr %28, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load6.3 = load <8 x i32>, ptr %29, align 4, !dbg !16
// CHECK:STDOUT:   %wide.load7.3 = load <8 x i32>, ptr %30, align 4, !dbg !16
// CHECK:STDOUT:   %31 = shl <8 x i32> %wide.load.3, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %32 = shl <8 x i32> %wide.load5.3, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %33 = shl <8 x i32> %wide.load6.3, splat (i32 1), !dbg !16
// CHECK:STDOUT:   %34 = shl <8 x i32> %wide.load7.3, splat (i32 1), !dbg !16
// CHECK:STDOUT:   store <8 x i32> %31, ptr %27, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %32, ptr %28, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %33, ptr %29, align 4, !dbg !16
// CHECK:STDOUT:   store <8 x i32> %34, ptr %30, align 4, !dbg !16
// CHECK:STDOUT:   %index.next.3 = add nuw nsw i64 %index, 128, !dbg !84
// CHECK:STDOUT:   %35 = icmp eq i64 %index.next.3, 65536, !dbg !11
// CHECK:STDOUT:   br i1 %35, label %while.done, label %vector.body, !dbg !11, !llvm.loop !92
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %vector.body
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder i64 %index, { 3, 0, 1, 2, 4 }
// CHECK:STDOUT:   uselistorder ptr %0, { 3, 4, 0, 1, 2 }
// CHECK:STDOUT:   uselistorder ptr %1, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %2, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %3, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %10, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %11, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %12, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %19, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %20, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %21, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %28, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %29, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %30, { 1, 0 }
// CHECK:STDOUT:   uselistorder i64 %index.next.3, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "tune-cpu"="znver4" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!1}
// CHECK:STDOUT: !llvm.module.flags = !{!66, !67, !68, !74, !75}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!73}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !0, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !4 = !{null, !3}
// CHECK:STDOUT: !5 = !DISubroutineType(types: !4)
// CHECK:STDOUT: !6 = distinct !DISubprogram(name: "VectorizedWithAvx2", linkageName: "_CVectorizedWithAvx2.Main", scope: null, file: !0, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !23)
// CHECK:STDOUT: !7 = !DILocalVariable(arg: 1, scope: !6, type: !3)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 9, scope: !6)
// CHECK:STDOUT: !16 = !DILocation(line: 22, column: 5, scope: !6)
// CHECK:STDOUT: !19 = !{null, !2}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 17, column: 1, scope: !6)
// CHECK:STDOUT: !23 = !{!7}
// CHECK:STDOUT: !24 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !24, line: 396, type: !20, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !32)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !2)
// CHECK:STDOUT: !29 = !{null, !2, !2}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !32 = !{!26}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !24, line: 331, type: !30, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !48)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !2)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 2, scope: !41, type: !2)
// CHECK:STDOUT: !48 = !{!42, !43}
// CHECK:STDOUT: !66 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !67 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !68 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !69 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !70 = !{!"omnipotent char", !69, i64 0}
// CHECK:STDOUT: !71 = !{!"int", !70, i64 0}
// CHECK:STDOUT: !72 = !{!"__libc_errno", !71, i64 0}
// CHECK:STDOUT: !73 = !{!72, !71, i64 0}
// CHECK:STDOUT: !74 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !75 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !82 = distinct !DILocation(line: 23, column: 5, scope: !6)
// CHECK:STDOUT: !83 = distinct !DILocation(line: 398, column: 5, scope: !25, inlinedAt: !82)
// CHECK:STDOUT: !84 = distinct !DILocation(line: 331, column: 3, scope: !41, inlinedAt: !83)
// CHECK:STDOUT: !89 = !{!"llvm.loop.isvectorized", i32 1}
// CHECK:STDOUT: !91 = !{!"llvm.loop.unroll.runtime.disable"}
// CHECK:STDOUT: !92 = distinct !{!92, !89, !91}
// CHECK:STDOUT:
