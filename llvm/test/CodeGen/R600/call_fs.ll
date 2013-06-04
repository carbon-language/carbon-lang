
; RUN: llc < %s -march=r600 -mcpu=redwood -show-mc-encoding -o - | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=rv710 -show-mc-encoding -o - | FileCheck --check-prefix=R600-CHECK %s

; EG-CHECK: @call_fs
; EG-CHECK: .long 257
; EG-CHECK: CALL_FS  ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0xc0,0x84]
; R600-CHECK: @call_fs
; R600-CHECK: .long 257
; R600-CHECK:CALL_FS ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x89]


define void @call_fs() #0 {
  ret void
}

attributes #0 = { "ShaderType"="1" } ; Vertex Shader
