; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefixes CHECK,SLOW
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck %s

; Test that wasm select instruction is selected from LLVM select instruction.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: select_i32_bool:
; CHECK-NEXT: .functype select_i32_bool (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i32 @select_i32_bool(i1 zeroext %a, i32 %b, i32 %c) {
  %cond = select i1 %a, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_bool_nozext:
; CHECK-NEXT: .functype select_i32_bool_nozext (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const  $push0=, 1{{$}}
; CHECK-NEXT: i32.and    $push1=, $0, $pop0{{$}}
; CHECK-NEXT: i32.select $push2=, $1, $2, $pop1{{$}}
; CHECK-NEXT: return     $pop2{{$}}
define i32 @select_i32_bool_nozext(i1 %a, i32 %b, i32 %c) {
  %cond = select i1 %a, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_eq:
; CHECK-NEXT: .functype select_i32_eq (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i32 @select_i32_eq(i32 %a, i32 %b, i32 %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_ne:
; CHECK-NEXT: .functype select_i32_ne (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i32 @select_i32_ne(i32 %a, i32 %b, i32 %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i64_bool:
; CHECK-NEXT: .functype select_i64_bool (i32, i64, i64) -> (i64){{$}}
; CHECK-NEXT: i64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i64 @select_i64_bool(i1 zeroext %a, i64 %b, i64 %c) {
  %cond = select i1 %a, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_i64_bool_nozext:
; CHECK-NEXT: .functype select_i64_bool_nozext (i32, i64, i64) -> (i64){{$}}
; CHECK-NEXT: i32.const  $push0=, 1{{$}}
; CHECK-NEXT: i32.and    $push1=, $0, $pop0{{$}}
; CHECK-NEXT: i64.select $push2=, $1, $2, $pop1{{$}}
; CHECK-NEXT: return     $pop2{{$}}
define i64 @select_i64_bool_nozext(i1 %a, i64 %b, i64 %c) {
  %cond = select i1 %a, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_i64_eq:
; CHECK-NEXT: .functype select_i64_eq (i32, i64, i64) -> (i64){{$}}
; CHECK-NEXT: i64.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i64 @select_i64_eq(i32 %a, i64 %b, i64 %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_i64_ne:
; CHECK-NEXT: .functype select_i64_ne (i32, i64, i64) -> (i64){{$}}
; CHECK-NEXT: i64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i64 @select_i64_ne(i32 %a, i64 %b, i64 %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_f32_bool:
; CHECK-NEXT: .functype select_f32_bool (i32, f32, f32) -> (f32){{$}}
; CHECK-NEXT: f32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define float @select_f32_bool(i1 zeroext %a, float %b, float %c) {
  %cond = select i1 %a, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f32_bool_nozext:
; CHECK-NEXT: .functype select_f32_bool_nozext (i32, f32, f32) -> (f32){{$}}
; CHECK-NEXT: i32.const  $push0=, 1{{$}}
; CHECK-NEXT: i32.and    $push1=, $0, $pop0{{$}}
; CHECK-NEXT: f32.select $push2=, $1, $2, $pop1{{$}}
; CHECK-NEXT: return     $pop2{{$}}
define float @select_f32_bool_nozext(i1 %a, float %b, float %c) {
  %cond = select i1 %a, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f32_eq:
; CHECK-NEXT: .functype select_f32_eq (i32, f32, f32) -> (f32){{$}}
; CHECK-NEXT: f32.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define float @select_f32_eq(i32 %a, float %b, float %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f32_ne:
; CHECK-NEXT: .functype select_f32_ne (i32, f32, f32) -> (f32){{$}}
; CHECK-NEXT: f32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define float @select_f32_ne(i32 %a, float %b, float %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f64_bool:
; CHECK-NEXT: .functype select_f64_bool (i32, f64, f64) -> (f64){{$}}
; CHECK-NEXT: f64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define double @select_f64_bool(i1 zeroext %a, double %b, double %c) {
  %cond = select i1 %a, double %b, double %c
  ret double %cond
}

; CHECK-LABEL: select_f64_bool_nozext:
; CHECK-NEXT: .functype select_f64_bool_nozext (i32, f64, f64) -> (f64){{$}}
; CHECK-NEXT: i32.const  $push0=, 1{{$}}
; CHECK-NEXT: i32.and    $push1=, $0, $pop0{{$}}
; CHECK-NEXT: f64.select $push2=, $1, $2, $pop1{{$}}
; CHECK-NEXT: return     $pop2{{$}}
define double @select_f64_bool_nozext(i1 %a, double %b, double %c) {
  %cond = select i1 %a, double %b, double %c
  ret double %cond
}

; CHECK-LABEL: select_f64_eq:
; CHECK-NEXT: .functype select_f64_eq (i32, f64, f64) -> (f64){{$}}
; CHECK-NEXT: f64.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define double @select_f64_eq(i32 %a, double %b, double %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, double %b, double %c
  ret double %cond
}

; CHECK-LABEL: select_f64_ne:
; CHECK-NEXT: .functype select_f64_ne (i32, f64, f64) -> (f64){{$}}
; CHECK-NEXT: f64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define double @select_f64_ne(i32 %a, double %b, double %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, double %b, double %c
  ret double %cond
}

; CHECK-LABEL: pr40805_i32:
; CHECK-NEXT: .functype pr40805_i32 (i32, i32, i32) -> (i32){{$}}
; SLOW-NEXT: i32.const  $push0=, 1{{$}}
; SLOW-NEXT: i32.and    $push1=, $0, $pop0{{$}}
; SLOW-NEXT: i32.select $push2=, $1, $2, $pop1{{$}}
; SLOW-NEXT: return     $pop2{{$}}
define i32 @pr40805_i32(i32 %x, i32 %y, i32 %z) {
  %a = and i32 %x, 1
  %b = icmp ne i32 %a, 0
  %c = select i1 %b, i32 %y, i32 %z
  ret i32 %c
}

; CHECK-LABEL: pr40805_i64:
; CHECK-NEXT: .functype pr40805_i64 (i64, i64, i64) -> (i64){{$}}
; SLOW-NEXT: i32.wrap_i64 $push0=, $0{{$}}
; SLOW-NEXT: i32.const  $push1=, 1{{$}}
; SLOW-NEXT: i32.and    $push2=, $pop0, $pop1{{$}}
; SLOW-NEXT: i64.select $push3=, $1, $2, $pop2{{$}}
; SLOW-NEXT: return     $pop3{{$}}
define i64 @pr40805_i64(i64 %x, i64 %y, i64 %z) {
  %a = and i64 %x, 1
  %b = icmp ne i64 %a, 0
  %c = select i1 %b, i64 %y, i64 %z
  ret i64 %c
}

; CHECK-LABEL: pr44012_i32:
; CHECK-NEXT: .functype pr44012_i32 (i32, f32, f32) -> (f32){{$}}
; SLOW-NEXT: i32.const  $push0=, 1{{$}}
; SLOW-NEXT: i32.and    $push1=, $0, $pop0{{$}}
; SLOW-NEXT: f32.select $push2=, $1, $2, $pop1{{$}}
; SLOW-NEXT: return     $pop2{{$}}
define float @pr44012_i32(i32 %x, float %y, float %z) {
  %a = and i32 %x, 1
  %b = icmp ne i32 %a, 0
  %c = select i1 %b, float %y, float %z
  ret float %c
}

; CHECK-LABEL: pr44012_i64:
; CHECK-NEXT: .functype pr44012_i64 (i64, f32, f32) -> (f32){{$}}
; SLOW-NEXT: i32.wrap_i64 $push0=, $0{{$}}
; SLOW-NEXT: i32.const  $push1=, 1{{$}}
; SLOW-NEXT: i32.and    $push2=, $pop0, $pop1{{$}}
; SLOW-NEXT: f32.select $push3=, $1, $2, $pop2{{$}}
; SLOW-NEXT: return     $pop3{{$}}
define float @pr44012_i64(i64 %x, float %y, float %z) {
  %a = and i64 %x, 1
  %b = icmp ne i64 %a, 0
  %c = select i1 %b, float %y, float %z
  ret float %c
}
