; RUN: llc < %s

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

define float @foo(i32 signext %a) {
entry:
    %b = bitcast i32 %a to float
    ret float %b
}

define i32 @bar(float %a) {
entry:
    %b = bitcast float %a to i32
    ret i32 %b
}
