; RUN: llc < %s | grep {dsgf.%} | count 2
; RUN: llc < %s | grep {dsg.%}  | count 2
; RUN: llc < %s | grep {dl.%}   | count 2
; RUN: llc < %s | grep dlg      | count 2

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define i64 @div(i64 %a, i64* %b) nounwind readnone {
entry:
	%b1 = load i64* %b
	%div = sdiv i64 %a, %b1
	ret i64 %div
}

define i64 @div1(i64 %a, i64* %b) nounwind readnone {
entry:
        %b1 = load i64* %b
        %div = udiv i64 %a, %b1
        ret i64 %div
}

define i64 @rem(i64 %a, i64* %b) nounwind readnone {
entry:
        %b1 = load i64* %b
        %div = srem i64 %a, %b1
        ret i64 %div
}

define i64 @rem1(i64 %a, i64* %b) nounwind readnone {
entry:
        %b1 = load i64* %b
        %div = urem i64 %a, %b1
        ret i64 %div
}

define i32 @div2(i32 %a, i32* %b) nounwind readnone {
entry:
        %b1 = load i32* %b
        %div = sdiv i32 %a, %b1
        ret i32 %div
}

define i32 @div3(i32 %a, i32* %b) nounwind readnone {
entry:
        %b1 = load i32* %b
        %div = udiv i32 %a, %b1
        ret i32 %div
}

define i32 @rem2(i32 %a, i32* %b) nounwind readnone {
entry:
        %b1 = load i32* %b
        %div = srem i32 %a, %b1
        ret i32 %div
}

define i32 @rem3(i32 %a, i32* %b) nounwind readnone {
entry:
        %b1 = load i32* %b
        %div = urem i32 %a, %b1
        ret i32 %div
}

