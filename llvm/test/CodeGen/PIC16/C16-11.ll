;RUN: llc < %s -march=pic16

@c612.auto.a.b = internal global i1 false         ; <i1*> [#uses=2]
@c612.auto.A.b = internal global i1 false         ; <i1*> [#uses=2]

define void @c612() nounwind {
entry:
  %tmp3.b = load i1* @c612.auto.a.b               ; <i1> [#uses=1]
  %tmp3 = zext i1 %tmp3.b to i16                  ; <i16> [#uses=1]
  %tmp4.b = load i1* @c612.auto.A.b               ; <i1> [#uses=1]
  %tmp4 = select i1 %tmp4.b, i16 2, i16 0         ; <i16> [#uses=1]
  %cmp5 = icmp ne i16 %tmp3, %tmp4                ; <i1> [#uses=1]
  %conv7 = zext i1 %cmp5 to i8                    ; <i8> [#uses=1]
  tail call void @expectWrap(i8 %conv7, i8 2)
  ret void
}

define void @expectWrap(i8 %boolresult, i8 %errCode) nounwind {
entry:
  %tobool = icmp eq i8 %boolresult, 0             ; <i1> [#uses=1]
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @exit(i16 1)
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

define i16 @main() nounwind {
entry:
  tail call void @c612()
  ret i16 0
}

declare void @exit(i16) noreturn nounwind
