; This tests that the expected ARM attributes are emitted.

; RUN: llc < %s -mtriple=arm-linux-gnueabi -filetype=obj -o - \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=BASIC
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -march=arm -mcpu=cortex-a8 \
; RUN:          -mattr=-neon,-vfp3,+vfp2 -arm-reserve-r9 -filetype=obj -o - \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CORTEX-A8
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=Vt8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi \
; RUN:          -mattr=-neon,-crypto -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=V8-FPARMv8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi \
; RUN:          -mattr=-fp-armv8,-crypto -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=V8-NEON
; RUN: llc < %s -mtriple=armv8-linux-gnueabi \
; RUN:          -mattr=-crypto -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=V8-FPARMv8-NEON
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CORTEX-A9
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CORTEX-A15
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0 -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CORTEX-M0
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CORTEX-M4
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r5 -filetype=obj \
; RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CORTEX-R5

; BASIC:        Section {
; BASIC:          Name: .ARM.attributes
; BASIC-NEXT:     Type: SHT_ARM_ATTRIBUTES
; BASIC-NEXT:     Flags [ (0x0)
; BASIC-NEXT:     ]
; BASIC-NEXT:     Address: 0x0
; BASIC-NEXT:     Offset: 0x3C
; BASIC-NEXT:     Size: 30
; BASIC-NEXT:     Link: 0
; BASIC-NEXT:     Info: 0
; BASIC-NEXT:     AddressAlignment: 1
; BASIC-NEXT:     EntrySize: 0
; BASIC-NEXT:     SectionData (
; BASIC-NEXT:       0000: 411D0000 00616561 62690001 13000000
; BASIC-NEXT:       0010: 06010801 14011501 17031801 1901
; BASIC-NEXT:     )

; CORTEX-A8:      Name: .ARM.attributes
; CORTEX-A8-NEXT: Type: SHT_ARM_ATTRIBUTES
; CORTEX-A8-NEXT: Flags [ (0x0)
; CORTEX-A8-NEXT: ]
; CORTEX-A8-NEXT: Address: 0x0
; CORTEX-A8-NEXT: Offset: 0x3C
; CORTEX-A8-NEXT: Size: 47
; CORTEX-A8-NEXT: Link: 0
; CORTEX-A8-NEXT: Info: 0
; CORTEX-A8-NEXT: AddressAlignment: 1
; CORTEX-A8-NEXT: EntrySize: 0
; CORTEX-A8-NEXT: SectionData (
; CORTEX-A8-NEXT:   0000: 412E0000 00616561 62690001 24000000
; CORTEX-A8-NEXT:   0010: 05434F52 5445582D 41380006 0A074108
; CORTEX-A8-NEXT:   0020: 0109020A 02140115 01170318 011901
; CORTEX-A8-NEXT: )

; V7:      Name: .ARM.attributes
; V7-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; V7-NEXT: Flags [ (0x0)
; V7-NEXT: ]
; V7-NEXT: Address: 0x0
; V7-NEXT: Offset: 0x3C
; V7-NEXT: Size: 36
; V7-NEXT: Link: 0
; V7-NEXT: Info: 0
; V7-NEXT: AddressAlignment: 1
; V7-NEXT: EntrySize: 0
; V7-NEXT: SectionData (
; V7-NEXT:   0000: 41230000 00616561 62690001 19000000
; V7-NEXT:   0010: 060A0801 09020A03 0C011401 15011703
; V7-NEXT:   0020: 18011901
; V7-NEXT: )

; V8:      Name: .ARM.attributes
; V8-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; V8-NEXT: Flags [ (0x0)
; V8-NEXT: ]
; V8-NEXT: Address: 0x0
; V8-NEXT: Offset: 0x3C
; V8-NEXT: Size: 38
; V8-NEXT: Link: 0
; V8-NEXT: Info: 0
; V8-NEXT: AddressAlignment: 1
; V8-NEXT: EntrySize: 0
; V8-NEXT: SectionData (
; V8-NEXT:   0000: 41250000 00616561 62690001 1B000000
; V8-NEXT:   0010: 060E0801 09020A07 0C031401 15011703
; V8-NEXT:   0020: 18011901 2C02
; V8-NEXT: )

; Vt8:      Name: .ARM.attributes
; Vt8-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; Vt8-NEXT: Flags [ (0x0)
; Vt8-NEXT: ]
; Vt8-NEXT: Address: 0x0
; Vt8-NEXT: Offset: 0x38
; Vt8-NEXT: Size: 38
; Vt8-NEXT: Link: 0
; Vt8-NEXT: Info: 0
; Vt8-NEXT: AddressAlignment: 1
; Vt8-NEXT: EntrySize: 0
; Vt8-NEXT: SectionData (
; Vt8-NEXT:   0000: 41250000 00616561 62690001 1B000000
; Vt8-NEXT:   0010: 060E0801 09020A07 0C031401 15011703
; Vt8-NEXT:   0020: 18011901 2C02
; Vt8-NEXT: )


; V8-FPARMv8:      Name: .ARM.attributes
; V8-FPARMv8-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; V8-FPARMv8-NEXT: Flags [ (0x0)
; V8-FPARMv8-NEXT: ]
; V8-FPARMv8-NEXT: Address: 0x0
; V8-FPARMv8-NEXT: Offset: 0x3C
; V8-FPARMv8-NEXT: Size: 36
; V8-FPARMv8-NEXT: Link: 0
; V8-FPARMv8-NEXT: Info: 0
; V8-FPARMv8-NEXT: AddressAlignment: 1
; V8-FPARMv8-NEXT: EntrySize: 0
; V8-FPARMv8-NEXT: SectionData (
; V8-FPARMv8-NEXT:   0000: 41230000 00616561 62690001 19000000
; V8-FPARMv8-NEXT:   0010: 060E0801 09020A07 14011501 17031801
; V8-FPARMv8-NEXT:   0020: 19012C02
; V8-FPARMv8-NEXT: )


; V8-NEON:      Name: .ARM.attributes
; V8-NEON-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; V8-NEON-NEXT: Flags [ (0x0)
; V8-NEON-NEXT: ]
; V8-NEON-NEXT: Address: 0x0
; V8-NEON-NEXT: Offset: 0x3C
; V8-NEON-NEXT: Size: 38
; V8-NEON-NEXT: Link: 0
; V8-NEON-NEXT: Info: 0
; V8-NEON-NEXT: AddressAlignment: 1
; V8-NEON-NEXT: EntrySize: 0
; V8-NEON-NEXT: SectionData (
; V8-NEON-NEXT:   0000: 41250000 00616561 62690001 1B000000
; V8-NEON-NEXT:   0010: 060E0801 09020A05 0C031401 15011703
; V8-NEON-NEXT:   0020: 18011901 2C02
; V8-NEON-NEXT: )

; V8-FPARMv8-NEON:      Name: .ARM.attributes
; V8-FPARMv8-NEON-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; V8-FPARMv8-NEON-NEXT: Flags [ (0x0)
; V8-FPARMv8-NEON-NEXT: ]
; V8-FPARMv8-NEON-NEXT: Address: 0x0
; V8-FPARMv8-NEON-NEXT: Offset: 0x3C
; V8-FPARMv8-NEON-NEXT: Size: 38
; V8-FPARMv8-NEON-NEXT: Link: 0
; V8-FPARMv8-NEON-NEXT: Info: 0
; V8-FPARMv8-NEON-NEXT: AddressAlignment: 1
; V8-FPARMv8-NEON-NEXT: EntrySize: 0
; V8-FPARMv8-NEON-NEXT: SectionData (
; V8-FPARMv8-NEON-NEXT:   0000: 41250000 00616561 62690001 1B000000
; V8-FPARMv8-NEON-NEXT:   0010: 060E0801 09020A07 0C031401 15011703
; V8-FPARMv8-NEON-NEXT:   0020: 18011901 2C02
; V8-FPARMv8-NEON-NEXT: )

; CORTEX-A9:      Name: .ARM.attributes
; CORTEX-A9-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; CORTEX-A9-NEXT: Flags [ (0x0)
; CORTEX-A9-NEXT: ]
; CORTEX-A9-NEXT: Address: 0x0
; CORTEX-A9-NEXT: Offset: 0x3C
; CORTEX-A9-NEXT: Size: 49
; CORTEX-A9-NEXT: Link: 0
; CORTEX-A9-NEXT: Info: 0
; CORTEX-A9-NEXT: AddressAlignment: 1
; CORTEX-A9-NEXT: EntrySize: 0
; CORTEX-A9-NEXT: SectionData (
; CORTEX-A9-NEXT:   0000: 41300000 00616561 62690001 26000000
; CORTEX-A9-NEXT:   0010: 05434F52 5445582D 41390006 0A074108
; CORTEX-A9-NEXT:   0020: 0109020A 030C0114 01150117 03180119
; CORTEX-A9-NEXT:   0030: 01
; CORTEX-A9-NEXT: )

; CORTEX-A15:      Name: .ARM.attributes
; CORTEX-A15-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; CORTEX-A15-NEXT: Flags [ (0x0)
; CORTEX-A15-NEXT: ]
; CORTEX-A15-NEXT: Address: 0x0
; CORTEX-A15-NEXT: Offset: 0x3C
; CORTEX-A15-NEXT: Size: 52
; CORTEX-A15-NEXT: Link: 0
; CORTEX-A15-NEXT: Info: 0
; CORTEX-A15-NEXT: AddressAlignment: 1
; CORTEX-A15-NEXT: EntrySize: 0
; CORTEX-A15-NEXT: SectionData (
; CORTEX-A15-NEXT:   0000: 41330000 00616561 62690001 29000000
; CORTEX-A15-NEXT:   0010: 05434F52 5445582D 41313500 060A0741
; CORTEX-A15-NEXT:   0020: 08010902 0A050C02 14011501 17031801
; CORTEX-A15-NEXT:   0030: 19012C02
; CORTEX-A15-NEXT: )

; CORTEX-M0:      Name: .ARM.attributes
; CORTEX-M0-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; CORTEX-M0-NEXT: Flags [ (0x0)
; CORTEX-M0-NEXT: ]
; CORTEX-M0-NEXT: Address: 0x0
; CORTEX-M0-NEXT: Offset: 0x38
; CORTEX-M0-NEXT: Size: 45
; CORTEX-M0-NEXT: Link: 0
; CORTEX-M0-NEXT: Info: 0
; CORTEX-M0-NEXT: AddressAlignment: 1
; CORTEX-M0-NEXT: EntrySize: 0
; CORTEX-M0-NEXT: SectionData (
; CORTEX-M0-NEXT:   0000: 412C0000 00616561 62690001 22000000
; CORTEX-M0-NEXT:   0010: 05434F52 5445582D 4D300006 0C074D08
; CORTEX-M0-NEXT:   0020: 00090114 01150117 03180119 01
; CORTEX-M0-NEXT: )

; CORTEX-M4:      Name: .ARM.attributes
; CORTEX-M4-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; CORTEX-M4-NEXT: Flags [ (0x0)
; CORTEX-M4-NEXT: ]
; CORTEX-M4-NEXT: Address: 0x0
; CORTEX-M4-NEXT: Offset: 0x38
; CORTEX-M4-NEXT: Size: 49
; CORTEX-M4-NEXT: Link: 0
; CORTEX-M4-NEXT: Info: 0
; CORTEX-M4-NEXT: AddressAlignment: 1
; CORTEX-M4-NEXT: EntrySize: 0
; CORTEX-M4-NEXT: SectionData (
; CORTEX-M4-NEXT:   0000: 41300000 00616561 62690001 26000000
; CORTEX-M4-NEXT:   0010: 05434F52 5445582D 4D340006 0D074D08
; CORTEX-M4-NEXT:   0020: 0009020A 06140115 01170318 0119012C
; CORTEX-M4-NEXT:   0030: 00
; CORTEX-M4-NEXT: )

; CORTEX-R5:      Name: .ARM.attributes
; CORTEX-R5-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
; CORTEX-R5-NEXT: Flags [ (0x0)
; CORTEX-R5-NEXT: ]
; CORTEX-R5-NEXT: Address: 0x0
; CORTEX-R5-NEXT: Offset: 0x3C
; CORTEX-R5-NEXT: Size: 49
; CORTEX-R5-NEXT: Link: 0
; CORTEX-R5-NEXT: Info: 0
; CORTEX-R5-NEXT: AddressAlignment: 1
; CORTEX-R5-NEXT: EntrySize: 0
; CORTEX-R5-NEXT: SectionData (
; CORTEX-R5-NEXT:   0000: 41300000 00616561 62690001 26000000
; CORTEX-R5-NEXT:   0010: 05434F52 5445582D 52350006 0A075208
; CORTEX-R5-NEXT:   0020: 0109020A 04140115 01170318 0119012C
; CORTEX-R5-NEXT:   0030: 02
; CORTEX-R5-NEXT: )

define i32 @f(i64 %z) {
       ret i32 0
}
