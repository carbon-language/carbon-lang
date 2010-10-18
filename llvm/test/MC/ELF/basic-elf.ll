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

; 32: ('e_indent[EI_CLASS]', 0x1)
; 32: ('e_indent[EI_DATA]', 0x1)
; 32: ('e_indent[EI_VERSION]', 0x1)
; 32: ('_sections', [
; 32:   # Section 0
; 32:   (('sh_name', 0x0) # ''

; 32:   # '.text'

; 32: ('st_bind', 0x0)
; 32: ('st_type', 0x3)

; 32: ('st_bind', 0x0)
; 32: ('st_type', 0x3)

; 32: ('st_bind', 0x0)
; 32: ('st_type', 0x3)

; 32:   # 'main'
; 32:   ('st_bind', 0x1)
; 32-NEXT: ('st_type', 0x2)

; 32:   # 'puts'
; 32:   ('st_bind', 0x1)
; 32-NEXT: ('st_type', 0x0)

; 32:   # '.rel.text'

; 32:   ('_relocations', [
; 32:     # Relocation 0x0
; 32:     (('r_offset', 0x6)
; 32:      ('r_type', 0x1)
; 32:     ),
; 32:     # Relocation 0x1
; 32:     (('r_offset', 0xb)
; 32:      ('r_type', 0x2)
; 32:     ),
; 32:     # Relocation 0x2
; 32:     (('r_offset', 0x12)
; 32:      ('r_type', 0x1)
; 32:     ),
; 32:     # Relocation 0x3
; 32:     (('r_offset', 0x17)
; 32:      ('r_type', 0x2)
; 32:     ),
; 32:   ])

; 64: ('e_indent[EI_CLASS]', 0x2)
; 64: ('e_indent[EI_DATA]', 0x1)
; 64: ('e_indent[EI_VERSION]', 0x1)
; 64: ('_sections', [
; 64:   # Section 0
; 64:   (('sh_name', 0x0) # ''

; 64:   # '.text'

; 64: ('st_bind', 0x0)
; 64: ('st_type', 0x3)

; 64: ('st_bind', 0x0)
; 64: ('st_type', 0x3)

; 64: ('st_bind', 0x0)
; 64: ('st_type', 0x3)

; 64:   # 'main'
; 64-NEXT: ('st_bind', 0x1)
; 64-NEXT: ('st_type', 0x2)

; 64:   # 'puts'
; 64-NEXT: ('st_bind', 0x1)
; 64-NEXT: ('st_type', 0x0)

; 64:   # '.rela.text'

; 64:   ('_relocations', [
; 64:     # Relocation 0x0
; 64:     (('r_offset', 0x5)
; 64:      ('r_type', 0xa)
; 64:      ('r_addend', 0x0)
; 64:     ),
; 64:     # Relocation 0x1
; 64:     (('r_offset', 0xa)
; 64:      ('r_type', 0x2)
; 64:      ('r_addend', -0x4)
; 64:     ),
; 64:     # Relocation 0x2
; 64:     (('r_offset', 0xf)
; 64:      ('r_type', 0xa)
; 64:      ('r_addend', 0x6)
; 64:     ),
; 64:     # Relocation 0x3
; 64:     (('r_offset', 0x14)
; 64:      ('r_type', 0x2)
; 64:      ('r_addend', -0x4)
; 64:     ),
; 64:   ])
