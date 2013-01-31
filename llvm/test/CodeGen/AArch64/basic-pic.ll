; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -relocation-model=pic %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -relocation-model=pic -filetype=obj %s -o -| llvm-objdump -r - | FileCheck --check-prefix=CHECK-ELF %s

@var = global i32 0

; CHECK-ELF: RELOCATION RECORDS FOR [.text]

define i32 @get_globalvar() {
; CHECK: get_globalvar:

  %val = load i32* @var
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK: ldr x[[GOTLOC:[0-9]+]], [x[[GOTHI]], #:got_lo12:var]
; CHECK: ldr w0, [x[[GOTLOC]]]

; CHECK-ELF: R_AARCH64_ADR_GOT_PAGE var
; CHECK-ELF: R_AARCH64_LD64_GOT_LO12_NC var
  ret i32 %val
}

define i32* @get_globalvaraddr() {
; CHECK: get_globalvaraddr:

  %val = load i32* @var
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK: ldr x0, [x[[GOTHI]], #:got_lo12:var]

; CHECK-ELF: R_AARCH64_ADR_GOT_PAGE var
; CHECK-ELF: R_AARCH64_LD64_GOT_LO12_NC var
  ret i32* @var
}

@hiddenvar = hidden global i32 0

define i32 @get_hiddenvar() {
; CHECK: get_hiddenvar:

  %val = load i32* @hiddenvar
; CHECK: adrp x[[HI:[0-9]+]], hiddenvar
; CHECK: ldr w0, [x[[HI]], #:lo12:hiddenvar]

; CHECK-ELF: R_AARCH64_ADR_PREL_PG_HI21 hiddenvar
; CHECK-ELF: R_AARCH64_LDST32_ABS_LO12_NC hiddenvar
  ret i32 %val
}

define i32* @get_hiddenvaraddr() {
; CHECK: get_hiddenvaraddr:

  %val = load i32* @hiddenvar
; CHECK: adrp [[HI:x[0-9]+]], hiddenvar
; CHECK: add x0, [[HI]], #:lo12:hiddenvar

; CHECK-ELF: R_AARCH64_ADR_PREL_PG_HI21 hiddenvar
; CHECK-ELF: R_AARCH64_ADD_ABS_LO12_NC hiddenvar
  ret i32* @hiddenvar
}

define void()* @get_func() {
; CHECK: get_func:

  ret void()* bitcast(void()*()* @get_func to void()*)
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:get_func
; CHECK: ldr x0, [x[[GOTHI]], #:got_lo12:get_func]

  ; Particularly important that the ADRP gets a relocation, LLVM tends to think
  ; it can relax it because it knows where get_func is. It can't!
; CHECK-ELF: R_AARCH64_ADR_GOT_PAGE get_func
; CHECK-ELF: R_AARCH64_LD64_GOT_LO12_NC get_func
}