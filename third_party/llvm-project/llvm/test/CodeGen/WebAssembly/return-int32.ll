; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s
; RUN: llc < %s -asm-verbose=false -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: return_i32:
; CHECK-NEXT:  .functype return_i32 (i32) -> (i32){{$}}
; CHECK-NEXT:  local.get  $push0=, 0
; CHECK-NEXT:  end_function{{$}}
define i32 @return_i32(i32 %p) {
  ret i32 %p
}

; CHECK-LABEL: return_i32_twice:
; CHECK:      store
; CHECK-NEXT: i32.const $push[[L0:[^,]+]]=, 1{{$}}
; CHECK-NEXT: return $pop[[L0]]{{$}}
; CHECK:      store
; CHECK-NEXT: i32.const $push{{[^,]+}}=, 3{{$}}
; CHECK-NEXT: end_function{{$}}
define i32 @return_i32_twice(i32 %a) {
  %b = icmp ne i32 %a, 0
  br i1 %b, label %true, label %false

true:
  store i32 0, i32* null
  ret i32 1

false:
  store i32 2, i32* null
  ret i32 3
}
