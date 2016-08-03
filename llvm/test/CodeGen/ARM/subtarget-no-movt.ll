; RUN: llc -march thumb -mcpu=cortex-a8 -relocation-model=static %s -o - | FileCheck -check-prefix=NO-OPTION %s
; RUN: llc -march thumb -mcpu=cortex-a8 -relocation-model=static %s -o - -mattr=-no-movt | FileCheck -check-prefix=USE-MOVT %s
; RUN: llc -march thumb -mcpu=cortex-a8 -relocation-model=static %s -o - -mattr=+no-movt | FileCheck -check-prefix=NO-USE-MOVT %s
; RUN: llc -march thumb -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 | FileCheck -check-prefix=NO-OPTION %s
; RUN: llc -march thumb -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 -mattr=-no-movt | FileCheck -check-prefix=USE-MOVT %s
; RUN: llc -march thumb -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 -mattr=+no-movt | FileCheck -check-prefix=NO-USE-MOVT %s

; NO-OPTION-LABEL: {{_?}}foo0
; NO-OPTION: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-OPTION: [[L0]]:
; NO-OPTION: .long 2296237089

; USE-MOVT-LABEL: {{_?}}foo0
; USE-MOVT: movw [[R0:r[0-9]+]], #52257
; USE-MOVT: movt [[R0]], #35037

; NO-USE-MOVT-LABEL: {{_?}}foo0
; NO-USE-MOVT: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-USE-MOVT: [[L0]]:
; NO-USE-MOVT: .long 2296237089

define i32 @foo0(i32 %a) #0 {
  %1 = xor i32 -1998730207, %a
  ret i32 %1
}

; NO-OPTION-LABEL: {{_?}}foo1
; NO-OPTION: movw [[R0:r[0-9]+]], #52257
; NO-OPTION: movt [[R0]], #35037

; USE-MOVT-LABEL: {{_?}}foo1
; USE-MOVT: movw [[R0:r[0-9]+]], #52257
; USE-MOVT: movt [[R0]], #35037

; NO-USE-MOVT-LABEL: {{_?}}foo1
; NO-USE-MOVT: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-USE-MOVT: [[L0]]:
; NO-USE-MOVT: .long 2296237089

define i32 @foo1(i32 %a) {
  %1 = xor i32 -1998730207, %a
  ret i32 %1
}

; NO-OPTION-LABEL: {{_?}}foo2
; NO-OPTION: mov.w	r0, #-536813568
; NO-OPTION-LABEL-NOT: .long

; USE-MOVT-LABEL: {{_?}}foo2
; USE-MOVT: mov.w	r0, #-536813568
; USE-MOVT-NOT: .long

; NO-USE-MOVT-LABEL: {{_?}}foo2
; NO-USE-MOVT: mov.w	r0, #-536813568
; NO-USE-MOVT-NOT: .long
define i32 @foo2() {
  %1 = load i32, i32* inttoptr (i32 -536813568 to i32*) ; load from 0xe000e000
  ret i32 %1
}


attributes #0 = { "target-features"="+no-movt" }
