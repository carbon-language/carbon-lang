; RUN: llc -filetype=obj -mtriple i686-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

define i32 @f() nounwind optsize ssp {
entry:
  %call = tail call i32 inttoptr (i64 42 to i32 ()*)() nounwind optsize
  %add = add nsw i32 %call, 1
  ret i32 %add
}

; CHECK:      ('_relocations', [
; CHECK-NEXT:  # Relocation 0x00000000
; CHECK-NEXT:  (('r_offset', 0x00000004)
; CHECK-NEXT:   ('r_sym', 0x00000000)
; CHECK-NEXT:   ('r_type', 0x00000002)
; CHECK-NEXT:  ),
; CHECK-NEXT: ])
