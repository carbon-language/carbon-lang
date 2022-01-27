; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that the "returned" attribute is optimized effectively.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: _Z3foov:
; CHECK-NEXT: .functype _Z3foov () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 1{{$}}
; CHECK-NEXT: {{^}} call      $push1=, _Znwm, $pop0{{$}}
; CHECK-NEXT: {{^}} call      $push2=, _ZN5AppleC1Ev, $pop1{{$}}
; CHECK-NEXT: return    $pop2{{$}}
%class.Apple = type { i8 }
declare noalias i8* @_Znwm(i32)
declare %class.Apple* @_ZN5AppleC1Ev(%class.Apple* returned)
define %class.Apple* @_Z3foov() {
entry:
  %call = tail call noalias i8* @_Znwm(i32 1)
  %0 = bitcast i8* %call to %class.Apple*
  %call1 = tail call %class.Apple* @_ZN5AppleC1Ev(%class.Apple* %0)
  ret %class.Apple* %0
}

; CHECK-LABEL: _Z3barPvS_l:
; CHECK-NEXT: .functype _Z3barPvS_l (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: {{^}} call     $push0=, memcpy, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
declare i8* @memcpy(i8* returned, i8*, i32)
define i8* @_Z3barPvS_l(i8* %p, i8* %s, i32 %n) {
entry:
  %call = tail call i8* @memcpy(i8* %p, i8* %s, i32 %n)
  ret i8* %p
}

; Test that the optimization isn't performed on constant arguments.

; CHECK-LABEL: test_constant_arg:
; CHECK:      i32.const   $push0=, global{{$}}
; CHECK-NEXT: {{^}} call        $drop=, returns_arg, $pop0{{$}}
; CHECK-NEXT: return{{$}}
@global = external global i32
@addr = global i32* @global
define void @test_constant_arg() {
  %call = call i32* @returns_arg(i32* @global)
  ret void
}
declare i32* @returns_arg(i32* returned)

; Test that the optimization isn't performed on arguments without the
; "returned" attribute.

; CHECK-LABEL: test_other_skipped:
; CHECK-NEXT: .functype test_other_skipped (i32, i32, f64) -> (){{$}}
; CHECK-NEXT: {{^}} call     $drop=, do_something, $0, $1, $2{{$}}
; CHECK-NEXT: {{^}} call     do_something_with_i32, $1{{$}}
; CHECK-NEXT: {{^}} call     do_something_with_double, $2{{$}}
declare i32 @do_something(i32 returned, i32, double)
declare void @do_something_with_i32(i32)
declare void @do_something_with_double(double)
define void @test_other_skipped(i32 %a, i32 %b, double %c) {
    %call = call i32 @do_something(i32 %a, i32 %b, double %c)
    call void @do_something_with_i32(i32 %b)
    call void @do_something_with_double(double %c)
    ret void
}

; Test that the optimization is performed on arguments other than the first.

; CHECK-LABEL: test_second_arg:
; CHECK-NEXT: .functype test_second_arg (i32, i32) -> (i32){{$}}
; CHECK-NEXT: {{^}} call     $push0=, do_something_else, $0, $1{{$}}
; CHECK-NEXT: return   $pop0{{$}}
declare i32 @do_something_else(i32, i32 returned)
define i32 @test_second_arg(i32 %a, i32 %b) {
    %call = call i32 @do_something_else(i32 %a, i32 %b)
    ret i32 %b
}
