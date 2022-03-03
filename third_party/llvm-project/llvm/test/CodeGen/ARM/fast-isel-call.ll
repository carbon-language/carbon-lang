; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -mattr=+long-calls | FileCheck %s --check-prefix=ARM-LONG --check-prefix=ARM-LONG-MACHO
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -mattr=+long-calls | FileCheck %s --check-prefix=ARM-LONG --check-prefix=ARM-LONG-ELF
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -mattr=+long-calls | FileCheck %s --check-prefix=THUMB-LONG
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -mattr=-fpregs | FileCheck %s --check-prefix=ARM-NOVFP
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -mattr=-fpregs | FileCheck %s --check-prefix=ARM-NOVFP
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -mattr=-fpregs | FileCheck %s --check-prefix=THUMB-NOVFP

; Note that some of these tests assume that relocations are either
; movw/movt or constant pool loads. Different platforms will select
; different approaches.

define i32 @t0(i1 zeroext %a) nounwind {
  %1 = zext i1 %a to i32
  ret i32 %1
}

define i32 @t1(i8 signext %a) nounwind {
  %1 = sext i8 %a to i32
  ret i32 %1
}

define i32 @t2(i8 zeroext %a) nounwind {
  %1 = zext i8 %a to i32
  ret i32 %1
}

define i32 @t3(i16 signext %a) nounwind {
  %1 = sext i16 %a to i32
  ret i32 %1
}

define i32 @t4(i16 zeroext %a) nounwind {
  %1 = zext i16 %a to i32
  ret i32 %1
}

define void @foo(i8 %a, i16 %b) nounwind {
; ARM-LABEL: foo:
; THUMB-LABEL: foo:
;; Materialize i1 1
; ARM: movw [[REG0:r[0-9]+]], #1
; THUMB: movs [[REG0:r[0-9]+]], #1
;; zero-ext
; ARM: and [[REG1:r[0-9]+]], [[REG0]], #1
; THUMB: and [[REG1:r[0-9]+]], [[REG0]], #1
  %1 = call i32 @t0(i1 zeroext 1)
; ARM: sxtb	r0, {{r[0-9]+}}
; THUMB: sxtb	r0, {{r[0-9]+}}
  %2 = call i32 @t1(i8 signext %a)
; ARM: and	r0, {{r[0-9]+}}, #255
; THUMB: and	r0, {{r[0-9]+}}, #255
  %3 = call i32 @t2(i8 zeroext %a)
; ARM: sxth	r0, {{r[0-9]+}}
; THUMB: sxth	r0, {{r[0-9]+}}
  %4 = call i32 @t3(i16 signext %b)
; ARM: uxth	r0, {{r[0-9]+}}
; THUMB: uxth	r0, {{r[0-9]+}}
  %5 = call i32 @t4(i16 zeroext %b)

;; A few test to check materialization
;; Note: i1 1 was materialized with t1 call
; ARM: movw {{r[0-9]+}}, #255
%6 = call i32 @t2(i8 zeroext 255)
; ARM: movw {{r[0-9]+}}, #65535
; THUMB: movw {{r[0-9]+}}, #65535
%7 = call i32 @t4(i16 zeroext 65535)
  ret void
}

define void @foo2() nounwind {
  %1 = call signext i16 @t5()
  %2 = call zeroext i16 @t6()
  %3 = call signext i8 @t7()
  %4 = call zeroext i8 @t8()
  %5 = call zeroext i1 @t9()
  ret void
}

declare signext i16 @t5();
declare zeroext i16 @t6();
declare signext i8 @t7();
declare zeroext i8 @t8();
declare zeroext i1 @t9();

define i32 @t10() {
entry:
; ARM-LABEL: @t10
; ARM-DAG: movw [[R0:l?r[0-9]*]], #0
; ARM-DAG: movw [[R1:l?r[0-9]*]], #248
; ARM-DAG: movw [[R2:l?r[0-9]*]], #187
; ARM-DAG: movw [[R3:l?r[0-9]*]], #28
; ARM-DAG: movw [[R4:l?r[0-9]*]], #40
; ARM-DAG: movw [[R5:l?r[0-9]*]], #186
; ARM-DAG: and [[R0]], [[R0]], #255
; ARM-DAG: and [[R1]], [[R1]], #255
; ARM-DAG: and [[R2]], [[R2]], #255
; ARM-DAG: and [[R3]], [[R3]], #255
; ARM-DAG: and [[R4]], [[R4]], #255
; ARM-DAG: str [[R4]], [sp]
; ARM-DAG: and [[R5]], [[R5]], #255
; ARM-DAG: str [[R5]], [sp, #4]
; ARM: bl {{_?}}bar
; ARM-LONG-LABEL: @t10

; ARM-LONG-MACHO: {{(movw)|(ldr)}} [[R1:l?r[0-9]*]], {{(:lower16:L_bar\$non_lazy_ptr)|(.LCPI)}}
; ARM-LONG-MACHO: {{(movt [[R1]], :upper16:L_bar\$non_lazy_ptr)?}}
; ARM-LONG-MACHO: ldr [[R:r[0-9]+]], [[[R1]]]

; ARM-LONG-ELF: movw [[R1:r[0-9]*]], :lower16:bar
; ARM-LONG-ELF: movt [[R1]], :upper16:bar
; ARM-LONG-ELF: ldr [[R:r[0-9]+]], [[[R1]]]

; ARM-LONG: blx [[R]]
; THUMB-LABEL: @t10
; THUMB-DAG: movs [[R0:l?r[0-9]*]], #0
; THUMB-DAG: movs [[R1:l?r[0-9]*]], #248
; THUMB-DAG: movs [[R2:l?r[0-9]*]], #187
; THUMB-DAG: movs [[R3:l?r[0-9]*]], #28
; THUMB-DAG: movw [[R4:l?r[0-9]*]], #40
; THUMB-DAG: movw [[R5:l?r[0-9]*]], #186
; THUMB-DAG: and [[R0]], [[R0]], #255
; THUMB-DAG: and [[R1]], [[R1]], #255
; THUMB-DAG: and [[R2]], [[R2]], #255
; THUMB-DAG: and [[R3]], [[R3]], #255
; THUMB-DAG: and [[R4]], [[R4]], #255
; THUMB-DAG: str.w [[R4]], [sp]
; THUMB-DAG: and [[R5]], [[R5]], #255
; THUMB-DAG: str.w [[R5]], [sp, #4]
; THUMB: bl {{_?}}bar
; THUMB-LONG-LABEL: @t10
; THUMB-LONG: {{(movw)|(ldr.n)}} [[R1:l?r[0-9]*]], {{(:lower16:L_bar\$non_lazy_ptr)|(.LCPI)}}
; THUMB-LONG: {{(movt [[R1]], :upper16:L_bar\$non_lazy_ptr)?}}
; THUMB-LONG: ldr{{(.w)?}} [[R:r[0-9]+]], [[[R1]]]
; THUMB-LONG: blx [[R]]
  %call = call i32 @bar(i8 zeroext 0, i8 zeroext -8, i8 zeroext -69, i8 zeroext 28, i8 zeroext 40, i8 zeroext -70)
  ret i32 0
}

declare i32 @bar(i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext)

define i32 @bar0(i32 %i) nounwind {
  ret i32 0
}

define void @foo3() uwtable {
; ARM-LABEL: @foo3
; ARM: {{(movw r[0-9]+, :lower16:_?bar0)|(ldr r[0-9]+, .LCPI)}}
; ARM: {{(movt r[0-9]+, :upper16:_?bar0)|(ldr r[0-9]+, \[r[0-9]+\])}}
; ARM: movw    {{r[0-9]+}}, #0
; ARM: blx     {{r[0-9]+}}
; THUMB: {{(movw r[0-9]+, :lower16:_?bar0)|(ldr.n r[0-9]+, .LCPI)}}
; THUMB: {{(movt r[0-9]+, :upper16:_?bar0)|(ldr r[0-9]+, \[r[0-9]+\])}}
; THUMB: movs    {{r[0-9]+}}, #0
; THUMB: blx     {{r[0-9]+}}
  %fptr = alloca i32 (i32)*, align 8
  store i32 (i32)* @bar0, i32 (i32)** %fptr, align 8
  %1 = load i32 (i32)*, i32 (i32)** %fptr, align 8
  %call = call i32 %1(i32 0)
  ret void
}

define i32 @LibCall(i32 %a, i32 %b) {
entry:
; ARM-LABEL: LibCall:
; ARM: bl {{___udivsi3|__aeabi_uidiv}}
; ARM-LONG-LABEL: LibCall:

; ARM-LONG-MACHO: {{(movw r2, :lower16:L___udivsi3\$non_lazy_ptr)|(ldr r2, .LCPI)}}
; ARM-LONG-MACHO: {{(movt r2, :upper16:L___udivsi3\$non_lazy_ptr)?}}
; ARM-LONG-MACHO: ldr r2, [r2]

; ARM-LONG-ELF: movw r2, :lower16:__aeabi_uidiv
; ARM-LONG-ELF: movt r2, :upper16:__aeabi_uidiv

; ARM-LONG: blx r2
; THUMB-LABEL: LibCall:
; THUMB: bl {{___udivsi3|__aeabi_uidiv}}
; THUMB-LONG-LABEL: LibCall
; THUMB-LONG: {{(movw r2, :lower16:L___udivsi3\$non_lazy_ptr)|(ldr.n r2, .LCPI)}}
; THUMB-LONG: {{(movt r2, :upper16:L___udivsi3\$non_lazy_ptr)?}}
; THUMB-LONG: ldr r2, [r2]
; THUMB-LONG: blx r2
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

; Make sure we reuse the original ___udivsi3 rather than creating a new one
; called ___udivsi3.1 or whatever.
define i32 @LibCall2(i32 %a, i32 %b) {
entry:
; ARM-LABEL: LibCall2:
; ARM: bl {{___udivsi3|__aeabi_uidiv}}
; ARM-LONG-LABEL: LibCall2:

; ARM-LONG-MACHO: {{(movw r2, :lower16:L___udivsi3\$non_lazy_ptr)|(ldr r2, .LCPI)}}
; ARM-LONG-MACHO: {{(movt r2, :upper16:L___udivsi3\$non_lazy_ptr)?}}
; ARM-LONG-MACHO: ldr r2, [r2]

; ARM-LONG-ELF: movw r2, :lower16:__aeabi_uidiv
; ARM-LONG-ELF: movt r2, :upper16:__aeabi_uidiv

; ARM-LONG: blx r2
; THUMB-LABEL: LibCall2:
; THUMB: bl {{___udivsi3|__aeabi_uidiv}}
; THUMB-LONG-LABEL: LibCall2
; THUMB-LONG: {{(movw r2, :lower16:L___udivsi3\$non_lazy_ptr)|(ldr.n r2, .LCPI)}}
; THUMB-LONG: {{(movt r2, :upper16:L___udivsi3\$non_lazy_ptr)?}}
; THUMB-LONG: ldr r2, [r2]
; THUMB-LONG: blx r2
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

; Test fastcc

define fastcc void @fast_callee(float %i) ssp {
entry:
; ARM-LABEL: fast_callee:
; ARM: vmov r0, s0
; THUMB-LABEL: fast_callee:
; THUMB: vmov r0, s0
; ARM-NOVFP: fast_callee
; ARM-NOVFP-NOT: s0
; THUMB-NOVFP: fast_callee
; THUMB-NOVFP-NOT: s0
  call void @print(float %i)
  ret void
}

define void @fast_caller() ssp {
entry:
; ARM-LABEL: fast_caller:
; ARM: vldr s0,
; THUMB-LABEL: fast_caller:
; THUMB: vldr s0,
; ARM-NOVFP-LABEL: fast_caller:
; ARM-NOVFP: movw r0, #13107
; ARM-NOVFP: movt r0, #16611
; THUMB-NOVFP-LABEL: fast_caller:
; THUMB-NOVFP: movw r0, #13107
; THUMB-NOVFP: movt r0, #16611
  call fastcc void @fast_callee(float 0x401C666660000000)
  ret void
}

define void @no_fast_callee(float %i) ssp {
entry:
; ARM-LABEL: no_fast_callee:
; ARM: vmov s0, r0
; THUMB-LABEL: no_fast_callee:
; THUMB: vmov s0, r0
; ARM-NOVFP-LABEL: no_fast_callee:
; ARM-NOVFP-NOT: s0
; THUMB-NOVFP-LABEL: no_fast_callee:
; THUMB-NOVFP-NOT: s0
  call void @print(float %i)
  ret void
}

define void @no_fast_caller() ssp {
entry:
; ARM-LABEL: no_fast_caller:
; ARM: vmov r0, s0
; THUMB-LABEL: no_fast_caller:
; THUMB: vmov r0, s0
; ARM-NOVFP-LABEL: no_fast_caller:
; ARM-NOVFP: movw r0, #13107
; ARM-NOVFP: movt r0, #16611
; THUMB-NOVFP-LABEL: no_fast_caller:
; THUMB-NOVFP: movw r0, #13107
; THUMB-NOVFP: movt r0, #16611
  call void @no_fast_callee(float 0x401C666660000000)
  ret void
}

declare void @bar2(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6)

define void @call_undef_args() {
; ARM-LABEL: call_undef_args:
; ARM:       movw  r0, #1
; ARM-NEXT:  movw  r1, #2
; ARM-NEXT:  movw  r2, #3
; ARM-NEXT:  movw  r3, #4
; ARM-NOT:   str {{r[0-9]+}}, [sp]
; ARM:       movw  [[REG:l?r[0-9]*]], #6
; ARM-NEXT:  str [[REG]], [sp, #4]
  call void @bar2(i32 1, i32 2, i32 3, i32 4, i32 undef, i32 6)
  ret void
}

declare void @print(float)
