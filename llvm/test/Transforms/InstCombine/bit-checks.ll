; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | \
; RUN:    not grep {tobool}
; END.
define i32 @main(i32 %argc, i8** %argv) nounwind ssp {
entry:
  %and = and i32 %argc, 1                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 2                        ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = and i1 %tobool, %tobool3             ; <i1> [#uses=1]
  %retval.0 = select i1 %or.cond, i32 2, i32 1    ; <i32> [#uses=1]
  ret i32 %retval.0
}

define i32 @main2(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 1                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 2                        ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}