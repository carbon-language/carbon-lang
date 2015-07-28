; RUN: llc -emulated-tls -mtriple=arm-linux-android \
; RUN:     -relocation-model=pic < %s | FileCheck -check-prefix=ARM32 %s
; RUN: llc -emulated-tls -mtriple=aarch64-linux-android \
; RUN:     -relocation-model=pic < %s | FileCheck -check-prefix=ARM64 %s

; Copied from X86/emutls.ll

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define i32 @my_get_xyz() {
; ARM32-LABEL: my_get_xyz:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl my_emutls_get_address(PLT)
; ARM32-NEXT:   ldr r0, [r0]
; ARM64-LABEL: my_get_xyz:
; ARM64:        adrp x0, :got:my_emutls_v_xyz
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:my_emutls_v_xyz]
; ARM64-NEXT:   bl my_emutls_get_address
; ARM64-NEXT:   ldr  w0, [x0]
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  %call = call i8* @my_emutls_get_address(i8* bitcast (i8** @my_emutls_v_xyz to i8*))
  %0 = bitcast i8* %call to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

@i1 = thread_local global i32 15
@i2 = external thread_local global i32
@i3 = internal thread_local global i32 15
@i4 = hidden thread_local global i32 15
@i5 = external hidden thread_local global i32
@s1 = thread_local global i16 15
@b1 = thread_local global i8 0

define i32 @f1() {
; ARM32-LABEL: f1:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldr r0, [r0]
; ARM64-LABEL: f1:
; ARM64:        adrp x0, :got:__emutls_v.i1
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:__emutls_v.i1]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT:   ldr  w0, [x0]
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  %tmp1 = load i32, i32* @i1
  ret i32 %tmp1
}

define i32* @f2() {
; ARM32-LABEL: f2:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   pop
; ARM64-LABEL: f2:
; ARM64:        adrp x0, :got:__emutls_v.i1
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:__emutls_v.i1]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  ret i32* @i1
}

define i32 @f3() nounwind {
; ARM32-LABEL: f3:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldr r0, [r0]

entry:
  %tmp1 = load i32, i32* @i2
  ret i32 %tmp1
}

define i32* @f4() {
; ARM32-LABEL: f4:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   pop

entry:
  ret i32* @i2
}

define i32 @f5() nounwind {
; ARM32-LABEL: f5:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldr r0, [r0]

entry:
  %tmp1 = load i32, i32* @i3
  ret i32 %tmp1
}

define i32* @f6() {
; ARM32-LABEL: f6:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   pop

entry:
  ret i32* @i3
}

define i32 @f7() {
; ARM32-LABEL: f7:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldr r0, [r0]

entry:
  %tmp1 = load i32, i32* @i4
  ret i32 %tmp1
}

define i32* @f8() {
; ARM32-LABEL: f8:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   pop

entry:
  ret i32* @i4
}

define i32 @f9() {
; ARM32-LABEL: f9:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldr r0, [r0]

entry:
  %tmp1 = load i32, i32* @i5
  ret i32 %tmp1
}

define i32* @f10() {
; ARM32-LABEL: f10:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   pop

entry:
  ret i32* @i5
}

define i16 @f11() {
; ARM32-LABEL: f11:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldrh r0, [r0]

entry:
  %tmp1 = load i16, i16* @s1
  ret i16 %tmp1
}

define i32 @f12() {
; ARM32-LABEL: f12:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldrsh r0, [r0]

entry:
  %tmp1 = load i16, i16* @s1
  %tmp2 = sext i16 %tmp1 to i32
  ret i32 %tmp2
}

define i8 @f13() {
; ARM32-LABEL: f13:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldrb r0, [r0]
; ARM32-NEXT: pop

entry:
  %tmp1 = load i8, i8* @b1
  ret i8 %tmp1
}

define i32 @f14() {
; ARM32-LABEL: f14:
; ARM32:        ldr r0,
; ARM32-NEXT:   ldr r1,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   ldr r0, [r1, r0]
; ARM32-NEXT:   bl __emutls_get_address(PLT)
; ARM32-NEXT:   ldrsb r0, [r0]
; ARM32-NEXT: pop

entry:
  %tmp1 = load i8, i8* @b1
  %tmp2 = sext i8 %tmp1 to i32
  ret i32 %tmp2
}

;;;;;;;;;;;;;; 32-bit __emutls_v. and __emutls_t.

; ARM32       .section .data.rel.local,
; ARM32-LABEL: __emutls_v.i1:
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.i1

; ARM32       .section .rodata,
; ARM32-LABEL: __emutls_t.i1:
; ARM32-NEXT: .long 15

; ARM32-NOT:   __emutls_v.i2

; ARM32       .section .data.rel.local,
; ARM32-LABEL: __emutls_v.i3:
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.i3

; ARM32       .section .rodata,
; ARM32-LABEL: __emutls_t.i3:
; ARM32-NEXT: .long 15

; ARM32       .section .data.rel.local,
; ARM32-LABEL: __emutls_v.i4:
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.i4

; ARM32       .section .rodata,
; ARM32-LABEL: __emutls_t.i4:
; ARM32-NEXT: .long 15

; ARM32-NOT:   __emutls_v.i5:
; ARM32       .hidden __emutls_v.i5
; ARM32-NOT:   __emutls_v.i5:

; ARM32 .section .data.rel.local,
; ARM32-LABEL: __emutls_v.s1:
; ARM32-NEXT: .long 2
; ARM32-NEXT: .long 2
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.s1

; ARM32 .section .rodata,
; ARM32-LABEL: __emutls_t.s1:
; ARM32-NEXT: .short 15

; ARM32 .section .data.rel.local,
; ARM32-LABEL: __emutls_v.b1:
; ARM32-NEXT: .long 1
; ARM32-NEXT: .long 1
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long 0

; ARM32-NOT:   __emutls_t.b1

;;;;;;;;;;;;;; 64-bit __emutls_v. and __emutls_t.

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.i1:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i1

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.i1:
; ARM64-NEXT: .word 15

; ARM64-NOT:   __emutls_v.i2

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.i3:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i3

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.i3:
; ARM64-NEXT: .word 15

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.i4:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i4

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.i4:
; ARM64-NEXT: .word 15

; ARM64-NOT:   __emutls_v.i5:
; ARM64       .hidden __emutls_v.i5
; ARM64-NOT:   __emutls_v.i5:

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.s1:
; ARM64-NEXT: .xword 2
; ARM64-NEXT: .xword 2
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.s1

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.s1:
; ARM64-NEXT: .hword 15

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.b1:
; ARM64-NEXT: .xword 1
; ARM64-NEXT: .xword 1
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword 0

; ARM64-NOT:  __emutls_t.b1
