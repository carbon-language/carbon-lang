; RUN: llc < %s -mtriple=i686-pc-linux -filetype=obj | llvm-readobj -s -sr -sd | FileCheck %s

; CHECK:         Index: 8
; CHECK-NEXT:    Name: .eh_frame (41)
; CHECK-NEXT:    Type: SHT_PROGBITS (0x1)
; CHECK-NEXT:    Flags [ (0x2)
; CHECK-NEXT:      SHF_ALLOC (0x2)
; CHECK-NEXT:    ]
; CHECK-NEXT:    Address: 0x0
; CHECK-NEXT:    Offset: 0x64
; CHECK-NEXT:    Size: 60
; CHECK-NEXT:    Link: 0
; CHECK-NEXT:    Info: 0
; CHECK-NEXT:    AddressAlignment: 4
; CHECK-NEXT:    EntrySize: 0
; CHECK-NEXT:    Relocations [
; CHECK-NEXT:    ]
; CHECK-NEXT:    SectionData (
; CHECK-NEXT:      0000: 1C000000 00000000 017A504C 5200017C  |.........zPLR..||
; CHECK-NEXT:      0010: 08070000 00000000 1B0C0404 88010000  |................|
; CHECK-NEXT:      0020: 18000000 24000000 00000000 19000000  |....$...........|
; CHECK-NEXT:      0030: 04000000 00430E10 2E100000           |.....C......|
; CHECK-NEXT:    )

declare i32 @__gxx_personality_v0(...)
declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)

define void @test() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}
