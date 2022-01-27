; RUN: llc -verify-machineinstrs -O3 -use-registers-for-deopt-values -restrict-statepoint-remat=true -pass-remarks-filter=regalloc -pass-remarks-output=%t.yaml -stop-after=greedy -o - < %s 2>&1 | FileCheck %s
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

target triple = "x86_64-unknown-linux-gnu"

;CHECK-NOT: error: ran out of registers during register allocation

;YAML: --- !Missed
;YAML: Pass:            regalloc
;YAML: Name:            SpillReloadCopies
;YAML: Function:        barney
;YAML: Args:
;YAML:   - NumSpills:       '10'
;YAML:   - String:          ' spills '
;YAML:   - TotalSpillsCost: '7.000000e+00'
;YAML:   - String:          ' total spills cost '
;YAML:   - NumReloads:      '7'
;YAML:   - String:          ' reloads '
;YAML:   - TotalReloadsCost: '3.109004e-15'
;YAML:   - String:          ' total reloads cost '
;YAML:   - NumZeroCostFoldedReloads: '20'
;YAML:   - String:          ' zero cost folded reloads '
;YAML:   - NumVRCopies:     '2'
;YAML:   - String:          ' virtual registers copies '
;YAML:   - TotalCopiesCost: '8.882868e-16'
;YAML:   - String:          ' total copies cost '
;YAML:   - String:          generated in function

define void @barney(i8 addrspace(1)* %arg, double %arg1, double %arg2, double %arg3, double %arg4, double %arg5, double %arg6, double %arg7, double %arg8, double %arg9, double %arg10, double %arg11, double %arg12) gc "statepoint-example" personality i32* ()* @widget {
bb:
  %tmp = call coldcc token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* nonnull @blam, i32 0, i32 0, i32 0, i32 0) [ "deopt"(i32 0, i32 1, i32 0, i32 0, i32 0, i32 26, i32 0, i32 0, i8 addrspace(1)* %arg, i32 4, double %arg1, i32 7, i8* null, i32 4, double %arg2, i32 7, i8* null, i32 4, double %arg3, i32 7, i8* null, i32 4, double %arg4, i32 7, i8* null, i32 4, double %arg5, i32 7, i8* null, i32 4, double %arg6, i32 7, i8* null, i32 4, double %arg7, i32 7, i8* null, i32 4, double %arg8, i32 7, i8* null, i32 4, double %arg9, i32 7, i8* null, i32 4, double %arg10, i32 7, i8* null, i32 4, double %arg11, i32 7, i8* null, i32 4, double %arg12, i32 7, i8* null, i32 7, i8* null), "gc-live"(i8 addrspace(1)* %arg) ]
  br i1 undef, label %bb13, label %bb15

bb13:                                             ; preds = %bb
  %tmp14 = call token (i64, i32, i32 (i8 addrspace(1)*, double, double, double, double, double, double, double, double, double)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32p1i8f64f64f64f64f64f64f64f64f64f(i64 2, i32 5, i32 (i8 addrspace(1)*, double, double, double, double, double, double, double, double, double)* nonnull @quux, i32 10, i32 0, i8 addrspace(1)* nonnull null, double %arg1, double %arg2, double %arg3, double %arg5, double %arg6, double %arg7, double %arg9, double %arg10, double %arg11, i32 0, i32 0) [ "deopt"(i32 0, i32 2, i32 0, i32 70, i32 0, i32 26, i32 0, i32 0, i8 addrspace(1)* null, i32 4, double %arg1, i32 7, i8* null, i32 4, double %arg2, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 4, double %arg4, i32 7, i8* null, i32 4, double %arg5, i32 7, i8* null, i32 4, double %arg6, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 4, double %arg8, i32 7, i8* null, i32 4, double %arg9, i32 7, i8* null, i32 4, double %arg10, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 4, double %arg12, i32 7, i8* null, i32 7, i8* null), "gc-live"(i8 addrspace(1)* null) ]
  br label %bb15

bb15:                                             ; preds = %bb13, %bb
  %tmp16 = phi double [ %arg4, %bb13 ], [ 1.000000e+00, %bb ]
  %tmp17 = phi double [ %arg8, %bb13 ], [ 1.000000e+00, %bb ]
  %tmp18 = phi double [ %arg12, %bb13 ], [ 1.000000e+00, %bb ]
  br i1 undef, label %bb25, label %bb19

bb19:                                             ; preds = %bb15
  %tmp20 = invoke token (i64, i32, i32 (i32, i8 addrspace(1)*, i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32i32p1i8i32f(i64 1, i32 16, i32 (i32, i8 addrspace(1)*, i32)* nonnull @eggs, i32 3, i32 0, i32 undef, i8 addrspace(1)* nonnull undef, i32 0, i32 0, i32 0) [ "deopt"(i32 0, i32 2, i32 0, i32 97, i32 0, i32 26, i32 0, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 0, i32 2, i32 3, i32 0, i32 20, i32 0, i32 0, i8 addrspace(1)* undef, i32 4, double %arg1, i32 7, i8* null, i32 4, double %arg2, i32 7, i8* null, i32 4, double %tmp16, i32 7, i8* null, i32 4, double %arg5, i32 7, i8* null, i32 4, double %arg6, i32 7, i8* null, i32 4, double %tmp17, i32 7, i8* null, i32 4, double %arg9, i32 7, i8* null, i32 4, double %arg10, i32 7, i8* null, i32 4, double %tmp18, i32 7, i8* null, i32 7, i8* null), "gc-live"(i8 addrspace(1)* undef) ]
          to label %bb21 unwind label %bb23

bb21:                                             ; preds = %bb19
  %tmp22 = call token (i64, i32, void (i8 addrspace(1)*, double, double, double, double, double, double, double, double, double, i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i8f64f64f64f64f64f64f64f64f64i32f(i64 2, i32 5, void (i8 addrspace(1)*, double, double, double, double, double, double, double, double, double, i32)* nonnull @ham, i32 11, i32 0, i8 addrspace(1)* nonnull undef, double %arg1, double %arg2, double %tmp16, double %arg5, double %arg6, double %tmp17, double %arg9, double %arg10, double %tmp18, i32 51, i32 0, i32 0) [ "deopt"(i32 0, i32 2, i32 0, i32 97, i32 0, i32 26, i32 0, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 7, i8* null, i32 2, i32 2, i32 46, i32 0, i32 20, i32 0, i32 0, i8 addrspace(1)* undef, i32 4, double %arg1, i32 7, i8* null, i32 4, double %arg2, i32 7, i8* null, i32 4, double %tmp16, i32 7, i8* null, i32 4, double %arg5, i32 7, i8* null, i32 4, double %arg6, i32 7, i8* null, i32 4, double %tmp17, i32 7, i8* null, i32 4, double %arg9, i32 7, i8* null, i32 4, double %arg10, i32 7, i8* null, i32 4, double %tmp18, i32 7, i8* null, i32 3, i32 51), "gc-live"(i8 addrspace(1)* undef) ]
  unreachable

bb23:                                             ; preds = %bb19
  %tmp24 = landingpad token
          cleanup
  ret void

bb25:                                             ; preds = %bb15
  ret void
}

declare i32* @widget()
declare i32 @quux(i8 addrspace(1)*, double, double, double, double, double, double, double, double, double)
declare void @blam()
declare i32 @eggs(i32, i8 addrspace(1)*, i32)
declare void @ham(i8 addrspace(1)*, double, double, double, double, double, double, double, double, double, i32)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 , i32 , void ()*, i32 , i32 , ...)
declare token @llvm.experimental.gc.statepoint.p0f_i32p1i8f64f64f64f64f64f64f64f64f64f(i64 , i32 , i32 (i8 addrspace(1)*, double, double, double, double, double, double, double, double, double)*, i32 , i32 , ...)
declare token @llvm.experimental.gc.statepoint.p0f_i32i32p1i8i32f(i64 , i32 , i32 (i32, i8 addrspace(1)*, i32)*, i32 , i32 , ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidp1i8f64f64f64f64f64f64f64f64f64i32f(i64 , i32 , void (i8 addrspace(1)*, double, double, double, double, double, double, double, double, double, i32)*, i32 , i32 , ...)

;CHECK: body:             |
;CHECK:   bb.0.bb:
;CHECK:     successors: %bb.2(0x40000000), %bb.1(0x40000000)
;CHECK:     liveins: $rdi, $xmm0, $xmm1, $xmm2, $xmm3, $xmm4, $xmm5, $xmm6, $xmm7
;CHECK:     %49:fr64 = COPY $xmm7
;CHECK:     %10:fr64 = COPY $xmm6
;CHECK:     %41:fr64 = COPY $xmm5
;CHECK:     %45:fr64 = COPY $xmm4
;CHECK:     %53:fr64 = COPY $xmm3
;CHECK:     %6:fr64 = COPY $xmm2
;CHECK:     %58:fr64 = COPY $xmm1
;CHECK:     %62:fr64 = COPY $xmm0
;CHECK:     %3:gr64 = COPY $rdi
;CHECK:     %76:fr64 = MOVSDrm_alt %fixed-stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.0)
;CHECK:     %14:fr64 = MOVSDrm_alt %fixed-stack.1, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.1, align 16)
;CHECK:     %66:fr64 = MOVSDrm_alt %fixed-stack.2, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.2)
;CHECK:     %71:fr64 = MOVSDrm_alt %fixed-stack.3, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.3, align 16)
;CHECK:     MOV64mr %stack.0, 1, $noreg, 0, $noreg, %3 :: (store (s64) into %stack.0)
;CHECK:     ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:     STATEPOINT 2882400000, 0, 0, target-flags(x86-plt) @blam, 2, 9, 2, 0, 2, 59, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 26, 2, 0, 2, 0, 1, 8, %stack.0, 0, 2, 4, %62, 2, 7, 2, 0, 2, 4, %58, 2, 7, 2, 0, 2, 4, %6, 2, 7, 2, 0, 2, 4, %53, 2, 7, 2, 0, 2, 4, %45, 2, 7, 2, 0, 2, 4, %41, 2, 7, 2, 0, 2, 4, %10, 2, 7, 2, 0, 2, 4, %49, 2, 7, 2, 0, 2, 4, %71, 2, 7, 2, 0, 2, 4, %66, 2, 7, 2, 0, 2, 4, %14, 2, 7, 2, 0, 2, 4, %76, 2, 7, 2, 0, 2, 7, 2, 0, 2, 1, 1, 8, %stack.0, 0, 2, 0, 2, 1, 0, 0, csr_64_mostregs, implicit-def $rsp, implicit-def $ssp :: (volatile load store (s64) on %stack.0)
;CHECK:     ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:     %17:gr32 = MOV32r0 implicit-def dead $eflags
;CHECK:     TEST8rr %17.sub_8bit, %17.sub_8bit, implicit-def $eflags
;CHECK:     MOVSDmr %stack.1, 1, $noreg, 0, $noreg, %41 :: (store (s64) into %stack.1)
;CHECK:     MOVSDmr %stack.2, 1, $noreg, 0, $noreg, %45 :: (store (s64) into %stack.2)
;CHECK:     MOVSDmr %stack.5, 1, $noreg, 0, $noreg, %58 :: (store (s64) into %stack.5)
;CHECK:     MOVSDmr %stack.6, 1, $noreg, 0, $noreg, %62 :: (store (s64) into %stack.6)
;CHECK:     JCC_1 %bb.2, 4, implicit killed $eflags
;CHECK:   bb.1:
;CHECK:     successors: %bb.3(0x80000000)
;CHECK:     %54:fr64 = MOVSDrm_alt $rip, 1, $noreg, %const.0, $noreg :: (load (s64) from constant-pool)
;CHECK:     MOVSDmr %stack.3, 1, $noreg, 0, $noreg, %54 :: (store (s64) into %stack.3)
;CHECK:     MOVSDmr %stack.4, 1, $noreg, 0, $noreg, %54 :: (store (s64) into %stack.4)
;CHECK:     MOVSDmr %stack.7, 1, $noreg, 0, $noreg, %54 :: (store (s64) into %stack.7)
;CHECK:     JMP_1 %bb.3
;CHECK:   bb.2.bb13:
;CHECK:     successors: %bb.3(0x80000000)
;CHECK:     ADJCALLSTACKDOWN64 8, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:     MOVSDmr $rsp, 1, $noreg, 0, $noreg, %14 :: (store (s64) into stack)
;CHECK:     dead $edi = MOV32r0 implicit-def dead $eflags, implicit-def $rdi
;CHECK:     $xmm0 = COPY %62
;CHECK:     $xmm1 = COPY %58
;CHECK:     $xmm2 = COPY %6
;CHECK:     $xmm3 = COPY %45
;CHECK:     $xmm4 = COPY %41
;CHECK:     $xmm5 = COPY %10
;CHECK:     $xmm6 = COPY %71
;CHECK:     $xmm7 = COPY %66
;CHECK:     MOVSDmr %stack.3, 1, $noreg, 0, $noreg, %49 :: (store (s64) into %stack.3)
;CHECK:     MOVSDmr %stack.4, 1, $noreg, 0, $noreg, %53 :: (store (s64) into %stack.4)
;CHECK:     MOVSDmr %stack.7, 1, $noreg, 0, $noreg, %76 :: (store (s64) into %stack.7)
;CHECK:     STATEPOINT 2, 5, 9, undef %22:gr64, $rdi, $xmm0, $xmm1, $xmm2, $xmm3, $xmm4, $xmm5, $xmm6, $xmm7, 2, 0, 2, 0, 2, 59, 2, 0, 2, 2, 2, 0, 2, 70, 2, 0, 2, 26, 2, 0, 2, 0, 2, 0, 2, 4, 1, 8, %stack.6, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.5, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.4, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.1, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.3, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.3, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.0, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 1, 2, 0, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def dead $eax :: (load (s64) from %stack.1), (load (s64) from %stack.2), (load (s64) from %stack.3), (load (s64) from %stack.4), (load (s64) from %stack.5), (load (s64) from %stack.6), (load (s64) from %fixed-stack.2), (load (s64) from %fixed-stack.3, align 16), (load (s64) from %fixed-stack.0)
;CHECK:     ADJCALLSTACKUP64 8, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:   bb.3.bb15:
;CHECK:     successors: %bb.7(0x7ffff800), %bb.4(0x00000800)
;CHECK:     %24:gr32 = MOV32r0 implicit-def dead $eflags
;CHECK:     TEST8rr %24.sub_8bit, %24.sub_8bit, implicit-def $eflags
;CHECK:     JCC_1 %bb.7, 5, implicit killed $eflags
;CHECK:     JMP_1 %bb.4
;CHECK:   bb.4.bb19:
;CHECK:     successors: %bb.5(0x00000000), %bb.6(0x80000000)
;CHECK:     EH_LABEL <mcsymbol >
;CHECK:     ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:     $edx = MOV32r0 implicit-def dead $eflags
;CHECK:     STATEPOINT 1, 16, 3, undef %29:gr64, undef $edi, undef $rsi, $edx, 2, 0, 2, 0, 2, 105, 2, 0, 2, 2, 2, 0, 2, 97, 2, 0, 2, 26, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 0, 2, 2, 2, 3, 2, 0, 2, 20, 2, 0, 2, 0, 2, 4278124286, 2, 4, 1, 8, %stack.6, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.5, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.4, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.1, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.3, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.3, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.7, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 1, 2, 4278124286, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def dead $eax :: (load (s64) from %stack.1), (load (s64) from %stack.2), (load (s64) from %stack.3), (load (s64) from %stack.4), (load (s64) from %stack.5), (load (s64) from %stack.6), (load (s64) from %fixed-stack.2), (load (s64) from %fixed-stack.3, align 16), (load (s64) from %stack.7)
;CHECK:     ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:     EH_LABEL <mcsymbol >
;CHECK:     JMP_1 %bb.5
;CHECK:   bb.5.bb21:
;CHECK:     successors:
;CHECK:     ADJCALLSTACKDOWN64 8, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:     %81:fr64 = MOVSDrm_alt %stack.7, 1, $noreg, 0, $noreg :: (load (s64) from %stack.7)
;CHECK:     MOVSDmr $rsp, 1, $noreg, 0, $noreg, %81 :: (store (s64) into stack)
;CHECK:     $xmm0 = MOVSDrm_alt %stack.6, 1, $noreg, 0, $noreg :: (load (s64) from %stack.6)
;CHECK:     $xmm1 = MOVSDrm_alt %stack.5, 1, $noreg, 0, $noreg :: (load (s64) from %stack.5)
;CHECK:     $xmm2 = MOVSDrm_alt %stack.4, 1, $noreg, 0, $noreg :: (load (s64) from %stack.4)
;CHECK:     $xmm3 = MOVSDrm_alt %stack.2, 1, $noreg, 0, $noreg :: (load (s64) from %stack.2)
;CHECK:     $xmm4 = MOVSDrm_alt %stack.1, 1, $noreg, 0, $noreg :: (load (s64) from %stack.1)
;CHECK:     $xmm5 = MOVSDrm_alt %stack.3, 1, $noreg, 0, $noreg :: (load (s64) from %stack.3)
;CHECK:     %74:fr64 = MOVSDrm_alt %fixed-stack.3, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.3, align 16)
;CHECK:     %95:fr64 = COPY %74
;CHECK:     $xmm6 = COPY %95
;CHECK:     $esi = MOV32ri 51
;CHECK:     %69:fr64 = MOVSDrm_alt %fixed-stack.2, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.2)
;CHECK:     %97:fr64 = COPY %69
;CHECK:     $xmm7 = COPY %97
;CHECK:     STATEPOINT 2, 5, 10, undef %36:gr64, undef $rdi, $xmm0, $xmm1, $xmm2, $xmm3, $xmm4, $xmm5, $xmm6, $xmm7, killed $esi, 2, 0, 2, 0, 2, 105, 2, 0, 2, 2, 2, 0, 2, 97, 2, 0, 2, 26, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 7, 2, 0, 2, 2, 2, 2, 2, 46, 2, 0, 2, 20, 2, 0, 2, 0, 2, 4278124286, 2, 4, 1, 8, %stack.6, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.5, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.4, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.1, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.3, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.3, 0, 2, 7, 2, 0, 2, 4, 1, 8, %fixed-stack.2, 0, 2, 7, 2, 0, 2, 4, 1, 8, %stack.7, 0, 2, 7, 2, 0, 2, 3, 2, 51, 2, 1, 2, 4278124286, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp :: (load (s64) from %stack.7), (load (s64) from %stack.6), (load (s64) from %stack.5), (load (s64) from %stack.4), (load (s64) from %stack.2), (load (s64) from %stack.1), (load (s64) from %stack.3), (load (s64) from %fixed-stack.3, align 16), (load (s64) from %fixed-stack.2)
;CHECK:     ADJCALLSTACKUP64 8, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
;CHECK:   bb.6.bb23 (landing-pad):
;CHECK:     liveins: $rax, $rdx
;CHECK:     EH_LABEL <mcsymbol >
;CHECK:     RET 0
;CHECK:   bb.7.bb25:
;CHECK:     RET 0
