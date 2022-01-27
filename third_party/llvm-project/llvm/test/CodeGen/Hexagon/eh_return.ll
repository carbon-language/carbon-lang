; RUN: llc -O0 -march=hexagon < %s | FileCheck %s
; Make sure we generate an exception handling return.

; CHECK:         deallocframe
; CHECK-NEXT:  }
; CHECK-NEXT:  {
; CHECK-NEXT:    r29 = add(r29,r28)
; CHECK-NEXT:  }
; CHECK-NEXT:  {
; CHECK-NEXT:    jumpr r31
; CHECK-NEXT:  }

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

%struct.Data = type { i32, i8* }

define i32 @test_eh_return(i32 %a, i32 %b) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %d = alloca %struct.Data, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %b.addr, align 4
  %cmp = icmp sgt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %a.addr, align 4
  %3 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %2, %3
  ret i32 %add

if.else:                                          ; preds = %entry
  %call = call i32 @setup(%struct.Data* %d)
  %_d1 = getelementptr inbounds %struct.Data, %struct.Data* %d, i32 0, i32 0
  %4 = load i32, i32* %_d1, align 4
  %_d2 = getelementptr inbounds %struct.Data, %struct.Data* %d, i32 0, i32 1
  %5 = load i8*, i8** %_d2, align 4
  call void @llvm.eh.return.i32(i32 %4, i8* %5)
  unreachable
}

declare i32 @setup(%struct.Data*)

declare void @llvm.eh.return.i32(i32, i8*) nounwind
