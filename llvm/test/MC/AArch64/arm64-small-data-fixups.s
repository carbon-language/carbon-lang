; RUN: llvm-mc -triple arm64-apple-darwin -filetype=obj -o - %s | macho-dump | FileCheck %s

foo:
  .long 0
bar:
  .long 1

baz:
  .byte foo - bar
  .short foo - bar

; CHECK: # Relocation 0
; CHECK: (('word-0', 0x9),
; CHECK:  ('word-1', 0x1a000002)),
; CHECK: # Relocation 1
; CHECK: (('word-0', 0x9),
; CHECK:  ('word-1', 0xa000001)),
; CHECK: # Relocation 2
; CHECK: (('word-0', 0x8),
; CHECK:  ('word-1', 0x18000002)),
; CHECK: # Relocation 3
; CHECK: (('word-0', 0x8),
; CHECK:  ('word-1', 0x8000001)),

