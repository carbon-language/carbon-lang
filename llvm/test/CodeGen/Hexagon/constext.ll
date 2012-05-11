; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Make sure that constant extended instructions are generated.

; Check if add and add-sub instructions are extended.
define i32 @test1(i32 %b, i32* nocapture %c) nounwind {
entry:
%0 = load i32* %c, align 4
%add1 = add nsw i32 %0, 44400
; CHECK: add(r{{[0-9]+}}{{ *}},{{ *}}##44400)
%add = add i32 %b, 33000
%sub = sub i32 %add, %0
; CHECK: add(r{{[0-9]+}},{{ *}}sub(##33000,{{ *}}r{{[0-9]+}})
%add2 = add nsw i32 %add1, %0
store i32 %add1, i32* %c, align 4
  %mul = mul nsw i32 %add2, %sub
  ret i32 %mul
}

; Check if load and store instructions are extended.
define i32 @test2(i32* nocapture %b, i32 %c) nounwind {
entry:
  %arrayidx = getelementptr inbounds i32* %b, i32 7000
  %0 = load i32* %arrayidx, align 4
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memw(r{{[0-9]+}}{{ *}}+{{ *}}##28000) 
  %sub = sub nsw i32 8000, %0
; CHECK: sub(##8000{{ *}},{{ *}}r{{[0-9]+}})
  %cmp = icmp sgt i32 %sub, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %add = add nsw i32 %sub, %c
  br label %return

if.else:
  %arrayidx1 = getelementptr inbounds i32* %b, i32 6000
  store i32 %sub, i32* %arrayidx1, align 4
; CHECK: memw(r{{[0-9]+}}{{ *}}+{{ *}}##24000){{ *}}={{ *}}r{{[0-9]+}}
  br label %return

return:
  %retval.0 = phi i32 [ %add, %if.then ], [ 0, %if.else ]
  ret i32 %retval.0
}

; Check if the transfer, compare and mpyi instructions are extended.
define i32 @test3() nounwind {
entry:
  %call = tail call i32 @b(i32 1235, i32 34567) nounwind
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}##34567
  %sext = shl i32 %call, 16
  %conv1 = ashr exact i32 %sext, 16
  %cmp = icmp slt i32 %sext, 65536
  br i1 %cmp, label %if.then, label %if.else
; CHECK: cmp.gt(r{{[0-9]+}}{{ *}},{{ *}}##65535)

if.then:
  %mul = mul nsw i32 %conv1, 34567
  br label %if.end
; CHECK: r{{[0-9]+}}{{ *}}=+{{ *}}mpyi(r{{[0-9]+}}{{ *}},{{ *}}##34567)

if.else:
  %mul5 = mul nsw i32 %conv1, 1235
  br label %if.end

if.end:
  %a.0 = phi i32 [ %mul, %if.then ], [ %mul5, %if.else ]
  ret i32 %a.0
}

declare i32 @b(i32, i32)
