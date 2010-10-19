; RUN: llc -filetype=obj -mtriple i686-pc-linux-gnu %s -o - | elf-dump | FileCheck -check-prefix=32 %s
; RUN: llc -filetype=obj -mtriple x86_64-pc-linux-gnu %s -o - | elf-dump | FileCheck -check-prefix=64 %s

@.str1 = private constant [6 x i8] c"Hello\00"
@.str2 = private constant [7 x i8] c"World!\00"

define i32 @main() nounwind {
  %1 = call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0))
  %2 = call i32 @puts(i8* getelementptr inbounds ([7 x i8]* @.str2, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind

; 32: ('e_indent[EI_CLASS]', 0x00000001)
; 32: ('e_indent[EI_DATA]', 0x00000001)
; 32: ('e_indent[EI_VERSION]', 0x00000001)
; 32: ('_sections', [
; 32:   # Section 0
; 32:   (('sh_name', 0x00000000) # ''

; 32:   # '.text'

; 32: ('st_bind', 0x00000000)
; 32: ('st_type', 0x00000003)

; 32: ('st_bind', 0x00000000)
; 32: ('st_type', 0x00000003)

; 32: ('st_bind', 0x00000000)
; 32: ('st_type', 0x00000003)

; 32:   # 'main'
; 32:   ('st_bind', 0x00000001)
; 32-NEXT: ('st_type', 0x00000002)

; 32:   # 'puts'
; 32:   ('st_bind', 0x00000001)
; 32-NEXT: ('st_type', 0x00000000)

; 32:   # '.rel.text'

; 32:   ('_relocations', [
; 32:     # Relocation 0x00000000
; 32:     (('r_offset', 0x00000006)
; 32:      ('r_type', 0x00000001)
; 32:     ),
; 32:     # Relocation 0x00000001
; 32:     (('r_offset', 0x0000000b)
; 32:      ('r_type', 0x00000002)
; 32:     ),
; 32:     # Relocation 0x00000002
; 32:     (('r_offset', 0x00000012)
; 32:      ('r_type', 0x00000001)
; 32:     ),
; 32:     # Relocation 0x00000003
; 32:     (('r_offset', 0x00000017)
; 32:      ('r_type', 0x00000002)
; 32:     ),
; 32:   ])

; 64: ('e_indent[EI_CLASS]', 0x00000002)
; 64: ('e_indent[EI_DATA]', 0x00000001)
; 64: ('e_indent[EI_VERSION]', 0x00000001)
; 64: ('_sections', [
; 64:   # Section 0
; 64:   (('sh_name', 0x00000000) # ''

; 64:   # '.text'

; 64: ('st_bind', 0x00000000)
; 64: ('st_type', 0x00000003)

; 64: ('st_bind', 0x00000000)
; 64: ('st_type', 0x00000003)

; 64: ('st_bind', 0x00000000)
; 64: ('st_type', 0x00000003)

; 64:   # 'main'
; 64-NEXT: ('st_bind', 0x00000001)
; 64-NEXT: ('st_type', 0x00000002)

; 64:   # 'puts'
; 64-NEXT: ('st_bind', 0x00000001)
; 64-NEXT: ('st_type', 0x00000000)

; 64:   # '.rela.text'

; 64:   ('_relocations', [
; 64:     # Relocation 0x00000000
; 64:     (('r_offset', 0x00000005)
; 64:      ('r_type', 0x0000000a)
; 64:      ('r_addend', 0x00000000)
; 64:     ),
; 64:     # Relocation 0x00000001
; 64:     (('r_offset', 0x0000000a)
; 64:      ('r_type', 0x00000002)
; 64:      ('r_addend', 0xfffffffc)
; 64:     ),
; 64:     # Relocation 0x00000002
; 64:     (('r_offset', 0x0000000f)
; 64:      ('r_type', 0x0000000a)
; 64:      ('r_addend', 0x00000006)
; 64:     ),
; 64:     # Relocation 0x00000003
; 64:     (('r_offset', 0x00000014)
; 64:      ('r_type', 0x00000002)
; 64:      ('r_addend', 0xfffffffc)
; 64:     ),
; 64:   ])
