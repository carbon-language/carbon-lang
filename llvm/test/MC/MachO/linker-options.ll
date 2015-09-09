; RUN: llc -O0 -mtriple=x86_64-apple-darwin -o - %s > %t
; RUN: FileCheck --check-prefix=CHECK-ASM < %t %s

; CHECK-ASM: .linker_option "-lz"
; CHECK-ASM-NEXT: .linker_option "-framework", "Cocoa"

; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=obj -o - %s | llvm-readobj -macho-linker-options > %t
; RUN: FileCheck --check-prefix=CHECK-OBJ < %t %s

; CHECK-OBJ: Linker Options {
; CHECK-OBJ:   Size: 16
; CHECK-OBJ:   Strings [
; CHECK-OBJ:     Value: -lz
; CHECK-OBJ:   ]
; CHECK-OBJ: }
; CHECK-OBJ: Linker Options {
; CHECK-OBJ:   Size: 32
; CHECK-OBJ:   Strings [
; CHECK-OBJ:     Value: -framework
; CHECK-OBJ:     Value: Cocoa
; CHECK-OBJ:   ]
; CHECK-OBJ: }
; CHECK-OBJ: Linker Options {
; CHECK-OBJ:   Size: 24
; CHECK-OBJ:   Strings [
; CHECK-OBJ:     Value: -lmath
; CHECK-OBJ:   ]
; CHECK-OBJ: }

!0 = !{i32 6, !"Linker Options", !{!{!"-lz"}, !{!"-framework", !"Cocoa"}, !{!"-lmath"}}}

!llvm.module.flags = !{ !0 }
