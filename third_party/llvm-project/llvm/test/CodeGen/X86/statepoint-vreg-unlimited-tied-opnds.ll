; RUN: llc -max-registers-for-gc-values=18 -stop-before greedy < %s | FileCheck --check-prefix=CHECK-VREG %s
; RUN: llc -max-registers-for-gc-values=18 -stop-after virtregrewriter < %s | FileCheck --check-prefix=CHECK-PREG %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare dso_local void @func()

define i32 @test_spill(
    i32 addrspace(1)* %arg00, i32 addrspace(1)* %arg01, i32 addrspace(1)* %arg02, i32 addrspace(1)* %arg03, i32 addrspace(1)* %arg04, i32 addrspace(1)* %arg05,
    i32 addrspace(1)* %arg06, i32 addrspace(1)* %arg07, i32 addrspace(1)* %arg08, i32 addrspace(1)* %arg09, i32 addrspace(1)* %arg10, i32 addrspace(1)* %arg11,
    i32 addrspace(1)* %arg12, i32 addrspace(1)* %arg13, i32 addrspace(1)* %arg14, i32 addrspace(1)* %arg15, i32 addrspace(1)* %arg16, i32 addrspace(1)* %arg17
    ) gc "statepoint-example" {
; CHECK-VREG-LABEL: test_spill
; CHECK-VREG:     %18:gr64 = COPY $r9
; CHECK-VREG:     %19:gr64 = COPY $r8
; CHECK-VREG:     %20:gr64 = COPY $rcx
; CHECK-VREG:     %21:gr64 = COPY $rdx
; CHECK-VREG:     %22:gr64 = COPY $rsi
; CHECK-VREG:     %23:gr64 = COPY $rdi
; CHECK-VREG:     %17:gr64 = MOV64rm %fixed-stack.11, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.11, align 16)
; CHECK-VREG:     %16:gr64 = MOV64rm %fixed-stack.10, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.10)
; CHECK-VREG:     %15:gr64 = MOV64rm %fixed-stack.9, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.9, align 16)
; CHECK-VREG:     %14:gr64 = MOV64rm %fixed-stack.8, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.8)
; CHECK-VREG:     %13:gr64 = MOV64rm %fixed-stack.7, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.7, align 16)
; CHECK-VREG:     %12:gr64 = MOV64rm %fixed-stack.6, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.6)
; CHECK-VREG:     %11:gr64 = MOV64rm %fixed-stack.5, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.5, align 16)
; CHECK-VREG:     %10:gr64 = MOV64rm %fixed-stack.4, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.4)
; CHECK-VREG:     %9:gr64 = MOV64rm %fixed-stack.3, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.3, align 16)
; CHECK-VREG:     %8:gr64 = MOV64rm %fixed-stack.2, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.2)
; CHECK-VREG:     %7:gr64 = MOV64rm %fixed-stack.1, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.1, align 16)
; CHECK-VREG:     %6:gr64 = MOV64rm %fixed-stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.0)
; CHECK-VREG:     %6:gr64, %7:gr64, %8:gr64, %9:gr64, %10:gr64, %11:gr64, %12:gr64, %13:gr64, %14:gr64, %15:gr64, %16:gr64, %17:gr64, %18:gr64, %19:gr64, %20:gr64, %21:gr64, %22:gr64, %23:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 18, %6(tied-def 0), %7(tied-def 1), %8(tied-def 2), %9(tied-def 3), %10(tied-def 4), %11(tied-def 5), %12(tied-def 6), %13(tied-def 7), %14(tied-def 8), %15(tied-def 9), %16(tied-def 10), %17(tied-def 11), %18(tied-def 12), %19(tied-def 13), %20(tied-def 14), %21(tied-def 15), %22(tied-def 16), %23(tied-def 17), 2, 0, 2, 18, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:     %38:gr32 = MOV32rm %23, 1, $noreg, 4, $noreg :: (load (s32) from %ir.gep00, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %22, 1, $noreg, 8, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep01, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %21, 1, $noreg, 12, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep02, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %20, 1, $noreg, 16, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep03, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %19, 1, $noreg, 20, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep04, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %18, 1, $noreg, 24, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep05, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %17, 1, $noreg, 28, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep06, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %16, 1, $noreg, 32, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep07, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %15, 1, $noreg, 36, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep08, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %14, 1, $noreg, 40, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep09, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %13, 1, $noreg, 44, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep10, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %12, 1, $noreg, 48, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep11, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %11, 1, $noreg, 52, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep12, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %10, 1, $noreg, 56, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep13, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %9, 1, $noreg, 60, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep14, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %8, 1, $noreg, 64, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep15, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %7, 1, $noreg, 68, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep16, addrspace 1)
; CHECK-VREG:     %38:gr32 = ADD32rm %38, %6, 1, $noreg, 72, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep17, addrspace 1)
; CHECK-VREG:     $eax = COPY %38

; CHECK-PREG:     renamable $rbx = COPY $r9
; CHECK-PREG:     MOV64mr %stack.6, 1, $noreg, 0, $noreg, $r8 :: (store (s64) into %stack.6)
; CHECK-PREG:     renamable $r12 = COPY $rcx
; CHECK-PREG:     renamable $r14 = COPY $rdx
; CHECK-PREG:     renamable $r15 = COPY $rsi
; CHECK-PREG:     renamable $r13 = COPY $rdi
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.11, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.11, align 16)
; CHECK-PREG:     MOV64mr %stack.7, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.7)
; CHECK-PREG:     renamable $rbp = MOV64rm %fixed-stack.10, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.10)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.9, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.9, align 16)
; CHECK-PREG:     MOV64mr %stack.11, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.11)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.8, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.8)
; CHECK-PREG:     MOV64mr %stack.5, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.5)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.7, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.7, align 16)
; CHECK-PREG:     MOV64mr %stack.4, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.4)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.6, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.6)
; CHECK-PREG:     MOV64mr %stack.3, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.3)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.5, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.5, align 16)
; CHECK-PREG:     MOV64mr %stack.2, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.2)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.4, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.4)
; CHECK-PREG:     MOV64mr %stack.1, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.1)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.3, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.3, align 16)
; CHECK-PREG:     MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.0)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.2, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.2)
; CHECK-PREG:     MOV64mr %stack.8, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.8)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.1, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.1, align 16)
; CHECK-PREG:     MOV64mr %stack.9, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.9)
; CHECK-PREG:     renamable $rax = MOV64rm %fixed-stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %fixed-stack.0)
; CHECK-PREG:     MOV64mr %stack.10, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.10)
; CHECK-PREG:     renamable $rbp, renamable $rbx, renamable $r12, renamable $r14, renamable $r15, renamable $r13 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 18, 1, 8, %stack.10, 0, 1, 8, %stack.9, 0, 1, 8, %stack.8, 0, 1, 8, %stack.0, 0, 1, 8, %stack.1, 0, 1, 8, %stack.2, 0, 1, 8, %stack.3, 0, 1, 8, %stack.4, 0, 1, 8, %stack.5, 0, 1, 8, %stack.11, 0, killed renamable $rbp(tied-def 0), 1, 8, %stack.7, 0, killed renamable $rbx(tied-def 1), 1, 8, %stack.6, 0, killed renamable $r12(tied-def 2), killed renamable $r14(tied-def 3), killed renamable $r15(tied-def 4), killed renamable $r13(tied-def 5), 2, 0, 2, 18, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, csr_64, implicit-def $rsp, implicit-def $ssp :: (load store (s64) on %stack.0), (load store (s64) on %stack.1), (load store (s64) on %stack.2), (load store (s64) on %stack.3), (load store (s64) on %stack.4), (load store (s64) on %stack.5), (load store (s64) on %stack.6), (load store (s64) on %stack.7), (load store (s64) on %stack.8), (load store (s64) on %stack.9), (load store (s64) on %stack.10), (load store (s64) on %stack.11)
; CHECK-PREG:     renamable $eax = MOV32rm killed renamable $r13, 1, $noreg, 4, $noreg :: (load (s32) from %ir.gep00, addrspace 1)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $r15, 1, $noreg, 8, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep01, addrspace 1)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $r14, 1, $noreg, 12, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep02, addrspace 1)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $r12, 1, $noreg, 16, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep03, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.6, 1, $noreg, 0, $noreg :: (load (s64) from %stack.6)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 20, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep04, addrspace 1)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rbx, 1, $noreg, 24, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep05, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.7, 1, $noreg, 0, $noreg :: (load (s64) from %stack.7)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 28, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep06, addrspace 1)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rbp, 1, $noreg, 32, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep07, addrspace 1)
; CHECK-PREG:     renamable $rcx = MOV64rm %stack.11, 1, $noreg, 0, $noreg :: (load (s64) from %stack.11)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rcx, 1, $noreg, 36, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep08, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.5, 1, $noreg, 0, $noreg :: (load (s64) from %stack.5)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 40, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep09, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.4, 1, $noreg, 0, $noreg :: (load (s64) from %stack.4)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 44, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep10, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.3, 1, $noreg, 0, $noreg :: (load (s64) from %stack.3)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 48, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep11, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.2, 1, $noreg, 0, $noreg :: (load (s64) from %stack.2)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 52, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep12, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.1, 1, $noreg, 0, $noreg :: (load (s64) from %stack.1)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 56, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep13, addrspace 1)
; CHECK-PREG:     renamable $rdi = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdi, 1, $noreg, 60, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep14, addrspace 1)
; CHECK-PREG:     renamable $rsi = MOV64rm %stack.8, 1, $noreg, 0, $noreg :: (load (s64) from %stack.8)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rsi, 1, $noreg, 64, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep15, addrspace 1)
; CHECK-PREG:     renamable $rdx = MOV64rm %stack.9, 1, $noreg, 0, $noreg :: (load (s64) from %stack.9)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rdx, 1, $noreg, 68, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep16, addrspace 1)
; CHECK-PREG:     renamable $rcx = MOV64rm %stack.10, 1, $noreg, 0, $noreg :: (load (s64) from %stack.10)
; CHECK-PREG:     renamable $eax = ADD32rm killed renamable $eax, killed renamable $rcx, 1, $noreg, 72, $noreg, implicit-def dead $eflags :: (load (s32) from %ir.gep17, addrspace 1)

    %token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(i32 addrspace(1)* %arg00, i32 addrspace(1)* %arg01, i32 addrspace(1)* %arg02, i32 addrspace(1)* %arg03, i32 addrspace(1)* %arg04, i32 addrspace(1)* %arg05, i32 addrspace(1)* %arg06, i32 addrspace(1)* %arg07, i32 addrspace(1)* %arg08,
    i32 addrspace(1)* %arg09, i32 addrspace(1)* %arg10, i32 addrspace(1)* %arg11, i32 addrspace(1)* %arg12, i32 addrspace(1)* %arg13, i32 addrspace(1)* %arg14, i32 addrspace(1)* %arg15, i32 addrspace(1)* %arg16, i32 addrspace(1)* %arg17) ]
    %rel00 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 0, i32 0) ; (%arg00, %arg00)
    %rel01 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 1, i32 1) ; (%arg01, %arg01)
    %rel02 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 2, i32 2) ; (%arg02, %arg02)
    %rel03 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 3, i32 3) ; (%arg03, %arg03)
    %rel04 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 4, i32 4) ; (%arg04, %arg04)
    %rel05 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 5, i32 5) ; (%arg05, %arg05)
    %rel06 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 6, i32 6) ; (%arg06, %arg06)
    %rel07 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 7, i32 7) ; (%arg07, %arg07)
    %rel08 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 8, i32 8) ; (%arg08, %arg08)
    %rel09 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 9, i32 9) ; (%arg09, %arg09)
    %rel10 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 10, i32 10) ; (%arg10, %arg10)
    %rel11 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 11, i32 11) ; (%arg11, %arg11)
    %rel12 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 12, i32 12) ; (%arg12, %arg12)
    %rel13 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 13, i32 13) ; (%arg13, %arg13)
    %rel14 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 14, i32 14) ; (%arg14, %arg14)
    %rel15 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 15, i32 15) ; (%arg15, %arg15)
    %rel16 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 16, i32 16) ; (%arg16, %arg16)
    %rel17 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %token, i32 17, i32 17) ; (%arg17, %arg17)
    %gep00 = getelementptr i32, i32 addrspace(1)* %rel00, i64 1
    %gep01 = getelementptr i32, i32 addrspace(1)* %rel01, i64 2
    %gep02 = getelementptr i32, i32 addrspace(1)* %rel02, i64 3
    %gep03 = getelementptr i32, i32 addrspace(1)* %rel03, i64 4
    %gep04 = getelementptr i32, i32 addrspace(1)* %rel04, i64 5
    %gep05 = getelementptr i32, i32 addrspace(1)* %rel05, i64 6
    %gep06 = getelementptr i32, i32 addrspace(1)* %rel06, i64 7
    %gep07 = getelementptr i32, i32 addrspace(1)* %rel07, i64 8
    %gep08 = getelementptr i32, i32 addrspace(1)* %rel08, i64 9
    %gep09 = getelementptr i32, i32 addrspace(1)* %rel09, i64 10
    %gep10 = getelementptr i32, i32 addrspace(1)* %rel10, i64 11
    %gep11 = getelementptr i32, i32 addrspace(1)* %rel11, i64 12
    %gep12 = getelementptr i32, i32 addrspace(1)* %rel12, i64 13
    %gep13 = getelementptr i32, i32 addrspace(1)* %rel13, i64 14
    %gep14 = getelementptr i32, i32 addrspace(1)* %rel14, i64 15
    %gep15 = getelementptr i32, i32 addrspace(1)* %rel15, i64 16
    %gep16 = getelementptr i32, i32 addrspace(1)* %rel16, i64 17
    %gep17 = getelementptr i32, i32 addrspace(1)* %rel17, i64 18
    %val00 = load i32, i32 addrspace(1)* %gep00, align 4
    %val01 = load i32, i32 addrspace(1)* %gep01, align 4
    %sum01 = add i32 %val00, %val01
    %val02 = load i32, i32 addrspace(1)* %gep02, align 4
    %sum02 = add i32 %sum01, %val02
    %val03 = load i32, i32 addrspace(1)* %gep03, align 4
    %sum03 = add i32 %sum02, %val03
    %val04 = load i32, i32 addrspace(1)* %gep04, align 4
    %sum04 = add i32 %sum03, %val04
    %val05 = load i32, i32 addrspace(1)* %gep05, align 4
    %sum05 = add i32 %sum04, %val05
    %val06 = load i32, i32 addrspace(1)* %gep06, align 4
    %sum06 = add i32 %sum05, %val06
    %val07 = load i32, i32 addrspace(1)* %gep07, align 4
    %sum07 = add i32 %sum06, %val07
    %val08 = load i32, i32 addrspace(1)* %gep08, align 4
    %sum08 = add i32 %sum07, %val08
    %val09 = load i32, i32 addrspace(1)* %gep09, align 4
    %sum09 = add i32 %sum08, %val09
    %val10 = load i32, i32 addrspace(1)* %gep10, align 4
    %sum10 = add i32 %sum09, %val10
    %val11 = load i32, i32 addrspace(1)* %gep11, align 4
    %sum11 = add i32 %sum10, %val11
    %val12 = load i32, i32 addrspace(1)* %gep12, align 4
    %sum12 = add i32 %sum11, %val12
    %val13 = load i32, i32 addrspace(1)* %gep13, align 4
    %sum13 = add i32 %sum12, %val13
    %val14 = load i32, i32 addrspace(1)* %gep14, align 4
    %sum14 = add i32 %sum13, %val14
    %val15 = load i32, i32 addrspace(1)* %gep15, align 4
    %sum15 = add i32 %sum14, %val15
    %val16 = load i32, i32 addrspace(1)* %gep16, align 4
    %sum16 = add i32 %sum15, %val16
    %val17 = load i32, i32 addrspace(1)* %gep17, align 4
    %sum17 = add i32 %sum16, %val17
    ret i32 %sum17
}

; Function Attrs: nounwind readonly
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32 immarg, i32 immarg) #0

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 immarg, i32 immarg, void ()*, i32 immarg, i32 immarg, ...)

; Function Attrs: nounwind
declare dso_local void @llvm.stackprotector(i8*, i8**) #1

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind }
