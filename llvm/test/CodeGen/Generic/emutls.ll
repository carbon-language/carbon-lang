; RUN: llc < %s -emulated-tls -mtriple=arm-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=i686-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_32 %s
; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-android -march=x86 -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_32 %s
; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_64 %s
; RUN: llc < %s -emulated-tls -mtriple=mipsel-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=MIPS_32 %s
; RUN: llc < %s -emulated-tls -mtriple=mips64el-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=MIPS_64 %s
; RUN: llc < %s -emulated-tls -march=ppc64 -relocation-model=pic \
; RUN:     | FileCheck %s
; RUN: llc < %s -emulated-tls -march=ppc32 -relocation-model=pic \
; RUN:     | FileCheck %s
; RUN: llc < %s -emulated-tls -march=x86 -mtriple=i386-linux-gnu -relocation-model=pic \
; RUN:     | FileCheck %s

; Make sure that TLS symbols are emitted in expected order.

@external_x = external thread_local global i32, align 8
@external_y = thread_local global i8 7, align 2
@internal_y = internal thread_local global i64 9, align 16

define i32* @get_external_x() {
entry:
  ret i32* @external_x
}

define i8* @get_external_y() {
entry:
  ret i8* @external_y
}

define i64* @get_internal_y() {
entry:
  ret i64* @internal_y
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; targets independent mode
; CHECK-LABEL: get_external_x:
; CHECK-NOT:   _tls_get_address
; CHECK:       __emutls_get_address
; CHECK-LABEL: get_external_y:
; CHECK:       __emutls_get_address
; CHECK-NOT:   _tls_get_address
; CHECK-LABEL: get_internal_y:

; CHECK-NOT:   __emutls_t.external_x:
; CHECK-NOT:   __emutls_v.external_x:

; CHECK-LABEL: __emutls_v.external_y:
; CHECK-LABEL: __emutls_t.external_y:
; CHECK:       __emutls_t.external_y

; CHECK-LABEL: __emutls_v.internal_y:
; CHECK-LABEL: __emutls_t.internal_y:
; CHECK:       __emutls_t.internal_y

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 32-bit mode
; ARM_32-LABEL:  get_external_x:
; X86_32-LABEL:  get_external_x:
; MIPS-LABEL:    get_external_x:

; ARM_32:        bl __emutls_get_address
; ARM_32:        .long __emutls_v.external_x

; X86_32:        movl __emutls_v.external_x
; X86_32:        calll __emutls_get_address

; ARM_32-LABEL:  get_external_y:
; X86_32-LABEL:  get_external_y:
; MIPS_32-LABEL: get_external_y:

; ARM_32:        bl __emutls_get_address
; ARM_32:        .long __emutls_v.external_y

; X86_32:        movl __emutls_v.external_y
; X86_32:        calll __emutls_get_address

; ARM_32-LABEL:  get_internal_y:
; X86_32-LABEL:  get_internal_y:
; MIPS_32-LABEL: get_internal_y:

; ARM_32:      bl __emutls_get_address
; ARM_32:      .long __emutls_v.internal_y

; X86_32:      movl __emutls_v.internal_y
; X86_32:      calll __emutls_get_address

; MIPS_32:     lw {{.+}}(__emutls_v.internal_y
; MIPS_32:     lw {{.+}}call16(__emutls_get_address

; ARM_32-NOT:   __emutls_t.external_x
; X86_32-NOT:   __emutls_t.external_x
; MIPS_32-NOT:  __emutls_t.external_x

; ARM_32-NOT:   __emutls_v.external_x:
; X86_32-NOT:   __emutls_v.external_x:
; MIPS_32-NOT:  __emutls_v.external_x:

; ARM_32:        .section .data.rel.local
; X86_32:        .section .data.rel.local
; MIPS_32:       .section .data.rel.local

; ARM_32:        .align 2
; X86_32:        .align 4
; MIPS_32:       .align 2

; ARM_32-LABEL:  __emutls_v.external_y:
; X86_32-LABEL:  __emutls_v.external_y:
; MIPS_32-LABEL: __emutls_v.external_y:

; ARM_32-NEXT:   .long 1
; ARM_32-NEXT:   .long 2
; ARM_32-NEXT:   .long 0
; ARM_32-NEXT:   .long __emutls_t.external_y

; X86_32-NEXT:   .long 1
; X86_32-NEXT:   .long 2
; X86_32-NEXT:   .long 0
; X86_32-NEXT:   .long __emutls_t.external_y

; ARM_32:        .section .rodata,
; X86_32:        .section .rodata,
; MIPS_32:       .section .rodata,

; ARM_32-LABEL:  __emutls_t.external_y:
; X86_32-LABEL:  __emutls_t.external_y:
; MIPS_32-LABEL: __emutls_t.external_y:

; ARM_32-NEXT:   .byte 7
; X86_32-NEXT:   .byte 7
; MIPS_32-NEXT:  .byte 7

; ARM_32:        .section .data.rel.local
; X86_32:        .section .data.rel.local
; MIPS_32:       .section .data.rel.local

; ARM_32:        .align 2
; X86_32:        .align 4
; MIPS_32:       .align 2

; ARM_32-LABEL:  __emutls_v.internal_y:
; X86_32-LABEL:  __emutls_v.internal_y:
; MIPS_32-LABEL: __emutls_v.internal_y:

; ARM_32-NEXT:   .long 8
; ARM_32-NEXT:   .long 16
; ARM_32-NEXT:   .long 0
; ARM_32-NEXT:   .long __emutls_t.internal_y

; X86_32-NEXT:   .long 8
; X86_32-NEXT:   .long 16
; X86_32-NEXT:   .long 0
; X86_32-NEXT:   .long __emutls_t.internal_y

; MIPS_32-NEXT:  .4byte 8
; MIPS_32-NEXT:  .4byte 16
; MIPS_32-NEXT:  .4byte 0
; MIPS_32-NEXT:  .4byte __emutls_t.internal_y

; ARM_32-LABEL:  __emutls_t.internal_y:
; X86_32-LABEL:  __emutls_t.internal_y:
; MIPS_32-LABEL: __emutls_t.internal_y:

; ARM_32-NEXT:   .long 9
; ARM_32-NEXT:   .long 0
; X86_32-NEXT:   .quad 9
; MIPS_32-NEXT:  .8byte 9


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 64-bit mode
; X86_64-LABEL:  get_external_x:
; ARM_64-LABEL:  get_external_x:
; MIPS_64-LABEL: get_external_x:

; X86_64:      __emutls_v.external_x
; X86_64:      __emutls_get_address

; ARM_64:      __emutls_v.external_x
; ARM_64:      __emutls_get_address

; X86_64-LABEL:  get_external_y:
; ARM_64-LABEL:  get_external_y:
; MIPS_64-LABEL: get_external_y:

; X86_64:      __emutls_v.external_y
; X86_64:      __emutls_get_address

; ARM_64:      __emutls_v.external_y
; ARM_64:      __emutls_get_address

; X86_64-LABEL:  get_internal_y:
; ARM_64-LABEL:  get_internal_y:
; MIPS_64-LABEL: get_internal_y:

; X86_64:      __emutls_v.internal_y
; X86_64:      __emutls_get_address

; ARM_64:      __emutls_v.internal_y
; ARM_64:      __emutls_get_address

; MIPS_64:     ld {{.+}}(__emutls_v.internal_y
; MIPS_64:     ld {{.+}}call16(__emutls_get_address

; ARM_64-NOT:   __emutls_t.external_x
; X86_64-NOT:   __emutls_t.external_x
; MIPS_64-NOT:  __emutls_t.external_x

; X86_64-NOT:   __emutls_v.external_x:
; ARM_64-NOT:   __emutls_v.external_x:
; MIPS_64-NOT:  __emutls_v.external_x:

; X86_64:        .align 8
; ARM_64:        .align 3

; X86_64-LABEL:  __emutls_v.external_y:
; ARM_64-LABEL:  __emutls_v.external_y:
; MIPS_64-LABEL: __emutls_v.external_y:

; X86_64-NEXT:   .quad 1
; X86_64-NEXT:   .quad 2
; X86_64-NEXT:   .quad 0
; X86_64-NEXT:   .quad __emutls_t.external_y

; ARM_64-NEXT:   .xword 1
; ARM_64-NEXT:   .xword 2
; ARM_64-NEXT:   .xword 0
; ARM_64-NEXT:   .xword __emutls_t.external_y

; X86_64-NOT:    __emutls_v.external_x:
; ARM_64-NOT:    __emutls_v.external_x:
; MIPS_64-NOT:   __emutls_v.external_x:

; ARM_64:        .section .rodata,
; X86_64:        .section .rodata,
; MIPS_64:       .section .rodata,

; X86_64-LABEL:  __emutls_t.external_y:
; ARM_64-LABEL:  __emutls_t.external_y:
; MIPS_64-LABEL: __emutls_t.external_y:

; X86_64-NEXT:   .byte 7
; ARM_64-NEXT:   .byte 7
; MIPS_64-NEXT:  .byte 7

; ARM_64:        .section .data.rel.local
; X86_64:        .section .data.rel.local
; MIPS_64:       .section .data.rel.local

; X86_64:        .align 8
; ARM_64:        .align 3
; MIPS_64:       .align 3

; X86_64-LABEL:  __emutls_v.internal_y:
; ARM_64-LABEL:  __emutls_v.internal_y:
; MIPS_64-LABEL: __emutls_v.internal_y:

; X86_64-NEXT:   .quad 8
; X86_64-NEXT:   .quad 16
; X86_64-NEXT:   .quad 0
; X86_64-NEXT:   .quad __emutls_t.internal_y

; ARM_64-NEXT:   .xword 8
; ARM_64-NEXT:   .xword 16
; ARM_64-NEXT:   .xword 0
; ARM_64-NEXT:   .xword __emutls_t.internal_y

; MIPS_64-NEXT:  .8byte 8
; MIPS_64-NEXT:  .8byte 16
; MIPS_64-NEXT:  .8byte 0
; MIPS_64-NEXT:  .8byte __emutls_t.internal_y

; ARM_64:        .section .rodata,
; X86_64:        .section .rodata,
; MIPS_64:       .section .rodata,

; X86_64-LABEL:  __emutls_t.internal_y:
; ARM_64-LABEL:  __emutls_t.internal_y:
; MIPS_64-LABEL: __emutls_t.internal_y:

; X86_64-NEXT:   .quad 9
; ARM_64-NEXT:   .xword 9
; MIPS_64-NEXT:  .8byte 9
