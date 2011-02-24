; RUN: opt -S -simplifycfg %s | FileCheck %s

define i8* @test1(i8* %x) nounwind {
entry:
  %tmp1 = load i8* %x, align 1
  %cmp = icmp eq i8 %tmp1, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %incdec.ptr = getelementptr inbounds i8* %x, i64 1
  br label %if.end

if.end:
  %x.addr = phi i8* [ %incdec.ptr, %if.then ], [ %x, %entry ]
  ret i8* %x.addr

; CHECK: @test1
; CHECK: %x.addr = select i1 %cmp, i8* %incdec.ptr, i8* %x
; CHECK: ret i8* %x.addr
}
