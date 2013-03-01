; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that the packetizer generates valid packets with constant
; extended add and base+offset store instructions.

; CHECK: {
; CHECK-NEXT: r{{[0-9]+}}{{ *}}={{ *}}add(r{{[0-9]+}}, ##{{[0-9]+}})
; CHECK-NEXT: memw(r{{[0-9]+}}+{{ *}}##{{[0-9]+}}){{ *}}={{ *}}r{{[0-9]+}}.new
; CHECK-NEXT: }

define i32 @test(i32* nocapture %a, i32* nocapture %b, i32 %c) nounwind {
entry:
  %add = add nsw i32 %c, 200002
  %0 = load i32* %a, align 4
  %add1 = add nsw i32 %0, 200000
  %arrayidx2 = getelementptr inbounds i32* %a, i32 3000
  store i32 %add1, i32* %arrayidx2, align 4
  %1 = load i32* %b, align 4
  %add4 = add nsw i32 %1, 200001
  %arrayidx5 = getelementptr inbounds i32* %a, i32 1
  store i32 %add4, i32* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i32* %b, i32 1
  %2 = load i32* %arrayidx7, align 4
  %cmp = icmp sgt i32 %add4, %2
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %arrayidx8 = getelementptr inbounds i32* %a, i32 2
  %3 = load i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32* %b, i32 2000
  %4 = load i32* %arrayidx9, align 4
  %sub = sub nsw i32 %3, %4
  %arrayidx10 = getelementptr inbounds i32* %a, i32 4000
  store i32 %sub, i32* %arrayidx10, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx11 = getelementptr inbounds i32* %b, i32 3200
  store i32 %add, i32* %arrayidx11, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 %add
}
