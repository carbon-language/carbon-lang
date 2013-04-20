; RUN: opt -simplifycfg -S < %s | FileCheck %s

define void @ifconvertstore(i32 %m, i32* %A, i32* %B, i32 %C, i32 %D) {
entry:
  %arrayidx = getelementptr inbounds i32* %B, i64 0
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %0, %C
  %arrayidx2 = getelementptr inbounds i32* %A, i64 0

; First store to the location.
  store i32 %add, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32* %B, i64 1
  %1 = load i32* %arrayidx4, align 4
  %add5 = add nsw i32 %1, %D
  %cmp6 = icmp sgt i32 %add5, %C
  br i1 %cmp6, label %if.then, label %ret.end

; Make sure we speculate stores like the following one. It is cheap compared to
; a mispredicated branch.
; CHECK: @ifconvertstore
; CHECK: %add5.add = select i1 %cmp6, i32 %add5, i32 %add
; CHECK: store i32 %add5.add, i32* %arrayidx2, align 4
if.then:
  store i32 %add5, i32* %arrayidx2, align 4
  br label %ret.end

ret.end:
  ret void
}

define void @noifconvertstore1(i32 %m, i32* %A, i32* %B, i32 %C, i32 %D) {
entry:
  %arrayidx = getelementptr inbounds i32* %B, i64 0
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %0, %C
  %arrayidx2 = getelementptr inbounds i32* %A, i64 0

; Store to a different location.
  store i32 %add, i32* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32* %B, i64 1
  %1 = load i32* %arrayidx4, align 4
  %add5 = add nsw i32 %1, %D
  %cmp6 = icmp sgt i32 %add5, %C
  br i1 %cmp6, label %if.then, label %ret.end

; CHECK: @noifconvertstore1
; CHECK-NOT: select
if.then:
  store i32 %add5, i32* %arrayidx2, align 4
  br label %ret.end

ret.end:
  ret void
}

declare void @unknown_fun()

define void @noifconvertstore2(i32 %m, i32* %A, i32* %B, i32 %C, i32 %D) {
entry:
  %arrayidx = getelementptr inbounds i32* %B, i64 0
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %0, %C
  %arrayidx2 = getelementptr inbounds i32* %A, i64 0

; First store to the location.
  store i32 %add, i32* %arrayidx2, align 4
  call void @unknown_fun()
  %arrayidx4 = getelementptr inbounds i32* %B, i64 1
  %1 = load i32* %arrayidx4, align 4
  %add5 = add nsw i32 %1, %D
  %cmp6 = icmp sgt i32 %add5, %C
  br i1 %cmp6, label %if.then, label %ret.end

; CHECK: @noifconvertstore2
; CHECK-NOT: select
if.then:
  store i32 %add5, i32* %arrayidx2, align 4
  br label %ret.end

ret.end:
  ret void
}

