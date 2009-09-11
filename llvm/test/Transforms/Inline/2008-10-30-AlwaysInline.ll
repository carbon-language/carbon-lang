; RUN: opt < %s -always-inline -S | not grep call 

; Ensure that threshold doesn't disrupt always inline.
; RUN: opt < %s -inline-threshold=-2000000001 -always-inline -S | not grep call 


define internal i32 @if0() alwaysinline {
       ret i32 1 
}

define i32 @f0() {
       %r = call i32 @if0()
       ret i32 %r
}
