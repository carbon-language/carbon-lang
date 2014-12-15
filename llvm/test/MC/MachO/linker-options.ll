; RUN: llc -O0 -mtriple=x86_64-apple-darwin -o - %s > %t
; RUN: FileCheck --check-prefix=CHECK-ASM < %t %s

; CHECK-ASM: .linker_option "-lz"
; CHECK-ASM-NEXT: .linker_option "-framework", "Cocoa"

; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=obj -o - %s | macho-dump > %t
; RUN: FileCheck --check-prefix=CHECK-OBJ < %t %s

; CHECK-OBJ: ('load_commands', [
; CHECK-OBJ:   # Load Command 1
; CHECK-OBJ:  (('command', 45)
; CHECK-OBJ:   ('size', 16)
; CHECK-OBJ:   ('count', 1)
; CHECK-OBJ:   ('_strings', [
; CHECK-OBJ: 	"-lz",
; CHECK-OBJ:   ])
; CHECK-OBJ:  ),
; CHECK-OBJ:   # Load Command 2
; CHECK-OBJ:  (('command', 45)
; CHECK-OBJ:   ('size', 32)
; CHECK-OBJ:   ('count', 2)
; CHECK-OBJ:   ('_strings', [
; CHECK-OBJ: 	"-framework",
; CHECK-OBJ: 	"Cocoa",
; CHECK-OBJ:   ])
; CHECK-OBJ:   # Load Command 3
; CHECK-OBJ:  (('command', 45)
; CHECK-OBJ:   ('size', 24)
; CHECK-OBJ:   ('count', 1)
; CHECK-OBJ:   ('_strings', [
; CHECK-OBJ: 	"-lmath",
; CHECK-OBJ:   ])
; CHECK-OBJ:  ),
; CHECK-OBJ: ])

!0 = !{i32 6, !"Linker Options", !{!{!"-lz"}, !{!"-framework", !"Cocoa"}, !{!"-lmath"}}}

!llvm.module.flags = !{ !0 }
