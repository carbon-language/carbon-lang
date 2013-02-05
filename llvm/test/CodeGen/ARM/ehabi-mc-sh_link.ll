; Test the sh_link in Elf32_Shdr.

; The .ARM.exidx section should be linked with corresponding text section.
; The sh_link in Elf32_Shdr should be filled with the section index of
; the text section.

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | elf-dump --dump-section-data \
; RUN:   | FileCheck %s

define void @test1() nounwind {
entry:
  ret void
}

define void @test2() nounwind section ".test_section" {
entry:
  ret void
}

; CHECK: # Section 1
; CHECK-NEXT: (('sh_name', 0x00000010) # '.text'

; CHECK:      (('sh_name', 0x00000005) # '.ARM.exidx'
; CHECK-NEXT:  ('sh_type', 0x70000001)
; CHECK-NEXT:  ('sh_flags', 0x00000082)
; CHECK-NEXT:  ('sh_addr', 0x00000000)
; CHECK-NEXT:  ('sh_offset', 0x0000005c)
; CHECK-NEXT:  ('sh_size', 0x00000008)
; CHECK-NEXT:  ('sh_link',  0x00000001)
; CHECK-NEXT:  ('sh_info',  0x00000000)
; CHECK-NEXT:  ('sh_addralign',  0x00000004)

; CHECK: # Section 7
; CHECK-NEXT: (('sh_name', 0x00000039) # '.test_section'

; CHECK:      (('sh_name', 0x0000002f) # '.ARM.exidx.test_section'
; CHECK-NEXT:  ('sh_type', 0x70000001)
; CHECK-NEXT:  ('sh_flags', 0x00000082)
; CHECK-NEXT:  ('sh_addr', 0x00000000)
; CHECK-NEXT:  ('sh_offset', 0x00000068)
; CHECK-NEXT:  ('sh_size', 0x00000008)
; CHECK-NEXT:  ('sh_link',  0x00000007)
; CHECK-NEXT:  ('sh_info',  0x00000000)
; CHECK-NEXT:  ('sh_addralign',  0x00000004)
