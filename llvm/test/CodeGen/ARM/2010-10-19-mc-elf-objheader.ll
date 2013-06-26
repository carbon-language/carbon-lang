; RUN: llc  %s -mtriple=arm-linux-gnueabi -filetype=obj -o - | \
; RUN:    llvm-readobj -s -sd | FileCheck  -check-prefix=BASIC %s 
; RUN: llc  %s -mtriple=armv7-linux-gnueabi -march=arm -mcpu=cortex-a8 \
; RUN:    -mattr=-neon,-vfp3,+vfp2 \
; RUN:    -arm-reserve-r9 -filetype=obj -o - | \
; RUN:    llvm-readobj -s -sd | FileCheck  -check-prefix=CORTEXA8 %s


; This tests that the extpected ARM attributes are emitted.
;
; BASIC:        Section {
; BASIC:          Name: .ARM.attributes
; BASIC-NEXT:     Type: SHT_ARM_ATTRIBUTES
; BASIC-NEXT:     Flags [ (0x0)
; BASIC-NEXT:     ]
; BASIC-NEXT:     Address: 0x0
; BASIC-NEXT:     Offset: 0x3C
; BASIC-NEXT:     Size: 28
; BASIC-NEXT:     Link: 0
; BASIC-NEXT:     Info: 0
; BASIC-NEXT:     AddressAlignment: 1
; BASIC-NEXT:     EntrySize: 0
; BASIC-NEXT:     SectionData (
; BASIC-NEXT:       0000: 411B0000 00616561 62690001 11000000
; BASIC-NEXT:       0010: 06011401 15011703 18011901
; BASIC-NEXT:     )

; CORTEXA8:        Name: .ARM.attributes
; CORTEXA8-NEXT:     Type: SHT_ARM_ATTRIBUTES
; CORTEXA8-NEXT:     Flags [ (0x0)
; CORTEXA8-NEXT:     ]
; CORTEXA8-NEXT:     Address: 0x0
; CORTEXA8-NEXT:     Offset: 0x3C
; CORTEXA8-NEXT:     Size: 47
; CORTEXA8-NEXT:     Link: 0
; CORTEXA8-NEXT:     Info: 0
; CORTEXA8-NEXT:     AddressAlignment: 1
; CORTEXA8-NEXT:     EntrySize: 0
; CORTEXA8-NEXT:     SectionData (
; CORTEXA8-NEXT:       0000: 412E0000 00616561 62690001 24000000
; CORTEXA8-NEXT:       0010: 05434F52 5445582D 41380006 0A074108
; CORTEXA8-NEXT:       0020: 0109020A 02140115 01170318 011901
; CORTEXA8-NEXT:     )

define i32 @f(i64 %z) {
       ret i32 0
}
