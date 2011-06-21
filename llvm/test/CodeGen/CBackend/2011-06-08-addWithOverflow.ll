; RUN: llc < %s -march=c
; Check that uadd and sadd with overflow are handled by C Backend.

%0 = type { i32, i1 }        ; type %0

define i1 @func1(i32 zeroext %v1, i32 zeroext %v2) nounwind {
entry:
    %t = call %0 @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)     ; <%0> [#uses=1]
    %obit = extractvalue %0 %t, 1       ; <i1> [#uses=1]
    br i1 %obit, label %carry, label %normal

normal:     ; preds = %entry
    ret i1 true

carry:      ; preds = %entry
    ret i1 false
}

define i1 @func2(i32 signext %v1, i32 signext %v2) nounwind {
entry:
    %t = call %0 @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)     ; <%0> [#uses=1]
    %obit = extractvalue %0 %t, 1       ; <i1> [#uses=1]
    br i1 %obit, label %carry, label %normal

normal:     ; preds = %entry
    ret i1 true

carry:      ; preds = %entry
    ret i1 false
}

declare %0 @llvm.sadd.with.overflow.i32(i32, i32) nounwind

declare %0 @llvm.uadd.with.overflow.i32(i32, i32) nounwind

