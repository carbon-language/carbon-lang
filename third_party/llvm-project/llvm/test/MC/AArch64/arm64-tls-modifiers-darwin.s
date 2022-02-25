; RUN: llvm-mc -triple=arm64-apple-ios7.0 %s -o - | FileCheck %s
; RUN: llvm-mc -triple=arm64-apple-ios7.0 -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

        adrp x2, _var@TLVPPAGE
        ldr x0, [x15, _var@TLVPPAGEOFF]
        add x30, x0, _var@TLVPPAGEOFF
; CHECK: adrp x2, _var@TLVPPAG
; CHECK: ldr x0, [x15, _var@TLVPPAGEOFF]
; CHECK: add x30, x0, _var@TLVPPAGEOFF

; CHECK-OBJ: 8 ARM64_RELOC_TLVP_LOAD_PAGEOFF12 _var
; CHECK-OBJ: 4 ARM64_RELOC_TLVP_LOAD_PAGEOFF12 _var
; CHECK-OBJ: 0 ARM64_RELOC_TLVP_LOAD_PAGE21 _var
