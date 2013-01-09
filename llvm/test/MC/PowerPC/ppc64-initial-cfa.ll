; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj -relocation-model=static %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck %s -check-prefix=STATIC
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj -relocation-model=pic %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck %s -check-prefix=PIC

; FIXME: this file should be in .s form, change when asm parser is available.

define void @f() {
entry:
  ret void
}

; STATIC:      ('sh_name', 0x{{.*}}) # '.eh_frame'
; STATIC-NEXT: ('sh_type', 0x00000001)
; STATIC-NEXT: ('sh_flags', 0x0000000000000002)
; STATIC-NEXT: ('sh_addr', 0x{{.*}})
; STATIC-NEXT: ('sh_offset', 0x{{.*}})
; STATIC-NEXT: ('sh_size', 0x0000000000000028)
; STATIC-NEXT: ('sh_link', 0x00000000)
; STATIC-NEXT: ('sh_info', 0x00000000)
; STATIC-NEXT: ('sh_addralign', 0x0000000000000008)
; STATIC-NEXT: ('sh_entsize', 0x0000000000000000)
; STATIC-NEXT: ('_section_data', '00000010 00000000 017a5200 01784101 1b0c0100 00000010 00000018 00000000 00000010 00000000')

; STATIC:      ('sh_name', 0x{{.*}}) # '.rela.eh_frame'
; STATIC-NEXT: ('sh_type', 0x00000004)
; STATIC-NEXT: ('sh_flags', 0x0000000000000000)
; STATIC-NEXT: ('sh_addr', 0x{{.*}})
; STATIC-NEXT: ('sh_offset', 0x{{.*}})
; STATIC-NEXT: ('sh_size', 0x0000000000000018)
; STATIC-NEXT: ('sh_link', 0x{{.*}})
; STATIC-NEXT: ('sh_info', 0x{{.*}})
; STATIC-NEXT: ('sh_addralign', 0x0000000000000008)
; STATIC-NEXT: ('sh_entsize', 0x0000000000000018)
; STATIC-NEXT: ('_relocations', [

; Static build should create R_PPC64_REL32 relocations
; STATIC-NEXT:  # Relocation 0
; STATIC-NEXT:  (('r_offset', 0x000000000000001c)
; STATIC-NEXT:   ('r_sym', 0x{{.*}})
; STATIC-NEXT:   ('r_type', 0x0000001a)
; STATIC-NEXT:   ('r_addend', 0x0000000000000000)
; STATIC-NEXT:  ),
; STATIC-NEXT: ])


; PIC:      ('sh_name', 0x{{.*}}) # '.eh_frame'
; PIC-NEXT: ('sh_type', 0x00000001)
; PIC-NEXT: ('sh_flags', 0x0000000000000002)
; PIC-NEXT: ('sh_addr', 0x{{.*}})
; PIC-NEXT: ('sh_offset', 0x{{.*}})
; PIC-NEXT: ('sh_size', 0x0000000000000028)
; PIC-NEXT: ('sh_link', 0x00000000)
; PIC-NEXT: ('sh_info', 0x00000000)
; PIC-NEXT: ('sh_addralign', 0x0000000000000008)
; PIC-NEXT: ('sh_entsize', 0x0000000000000000)
; PIC-NEXT: ('_section_data', '00000010 00000000 017a5200 01784101 1b0c0100 00000010 00000018 00000000 00000010 00000000')

; PIC:      ('sh_name', 0x{{.*}}) # '.rela.eh_frame'
; PIC-NEXT: ('sh_type', 0x00000004)
; PIC-NEXT: ('sh_flags', 0x0000000000000000)
; PIC-NEXT: ('sh_addr', 0x{{.*}})
; PIC-NEXT: ('sh_offset', 0x{{.*}})
; PIC-NEXT: ('sh_size', 0x0000000000000018)
; PIC-NEXT: ('sh_link', 0x{{.*}})
; PIC-NEXT: ('sh_info', 0x{{.*}})
; PIC-NEXT: ('sh_addralign', 0x0000000000000008)
; PIC-NEXT: ('sh_entsize', 0x0000000000000018)
; PIC-NEXT: ('_relocations', [

; PIC build should create R_PPC64_REL32 relocations
; PIC-NEXT:  # Relocation 0
; PIC-NEXT:  (('r_offset', 0x000000000000001c)
; PIC-NEXT:   ('r_sym', 0x{{.*}})
; PIC-NEXT:   ('r_type', 0x0000001a)
; PIC-NEXT:   ('r_addend', 0x0000000000000000)
; PIC-NEXT:  ),
; PIC-NEXT: ])
