; RUN: opt -S -simplifycfg < %s | FileCheck %s

%ST = type { i8, i8 }

define i8* @test1(%ST* %x, i8* %y) nounwind {
entry:
  %cmp = icmp eq %ST* %x, null
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %incdec.ptr = getelementptr %ST, %ST* %x, i32 0, i32 1
  br label %if.end

if.end:
  %x.addr = phi i8* [ %incdec.ptr, %if.then ], [ %y, %entry ]
  ret i8* %x.addr

; CHECK-LABEL: @test1(
; CHECK: %incdec.ptr.y = select i1 %cmp, i8* %incdec.ptr, i8* %y
; CHECK: ret i8* %incdec.ptr.y
}
