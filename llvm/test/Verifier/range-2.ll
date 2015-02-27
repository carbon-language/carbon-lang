; RUN: llvm-as < %s -o /dev/null

define i8 @f1(i8* %x) {
entry:
  %y = load i8, i8* %x, align 1, !range !0
  ret i8 %y
}
!0 = !{i8 0, i8 1}

define i8 @f2(i8* %x) {
entry:
  %y = load i8, i8* %x, align 1, !range !1
  ret i8 %y
}
!1 = !{i8 255, i8 1}

define i8 @f3(i8* %x) {
entry:
  %y = load i8, i8* %x, align 1, !range !2
  ret i8 %y
}
!2 = !{i8 1, i8 3, i8 5, i8 42}

define i8 @f4(i8* %x) {
entry:
  %y = load i8, i8* %x, align 1, !range !3
  ret i8 %y
}
!3 = !{i8 -1, i8 0, i8 1, i8 2}

define i8 @f5(i8* %x) {
entry:
  %y = load i8, i8* %x, align 1, !range !4
  ret i8 %y
}
!4 = !{i8 -1, i8 0, i8 1, i8 -2}

; We can annotate the range of the return value of a CALL.
define void @call_all(i8* %x) {
entry:
  %v1 = call i8 @f1(i8* %x), !range !0
  %v2 = call i8 @f2(i8* %x), !range !1
  %v3 = call i8 @f3(i8* %x), !range !2
  %v4 = call i8 @f4(i8* %x), !range !3
  %v5 = call i8 @f5(i8* %x), !range !4
  ret void
}

; We can annotate the range of the return value of an INVOKE.
define void @invoke_all(i8* %x) {
entry:
  %v1 = invoke i8 @f1(i8* %x) to label %cont unwind label %lpad, !range !0
  %v2 = invoke i8 @f2(i8* %x) to label %cont unwind label %lpad, !range !1
  %v3 = invoke i8 @f3(i8* %x) to label %cont unwind label %lpad, !range !2
  %v4 = invoke i8 @f4(i8* %x) to label %cont unwind label %lpad, !range !3
  %v5 = invoke i8 @f5(i8* %x) to label %cont unwind label %lpad, !range !4

cont:
  ret void

lpad:
  %4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          filter [0 x i8*] zeroinitializer
  ret void
}
declare i32 @__gxx_personality_v0(...)
