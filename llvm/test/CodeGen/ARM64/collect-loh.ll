; RUN: llc -mtriple=arm64-apple-ios -O2 -arm64-collect-loh -arm64-collect-loh-bb-only=false < %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -O2 -arm64-collect-loh -arm64-collect-loh-bb-only=false < %s -o - | FileCheck %s --check-prefix=CHECK-ELF

; CHECK-ELF-NOT: .loh
; CHECK-ELF-NOT: AdrpAdrp
; CHECK-ELF-NOT: AdrpAdd
; CHECK-ELF-NOT: AdrpLdrGot

@a = internal unnamed_addr global i32 0, align 4
@b = external global i32

; Function Attrs: noinline nounwind ssp
define void @foo(i32 %t) {
entry:
  %tmp = load i32* @a, align 4
  %add = add nsw i32 %tmp, %t
  store i32 %add, i32* @a, align 4
  ret void
}

; Function Attrs: nounwind ssp
; Testcase for <rdar://problem/15438605>, AdrpAdrp reuse is valid only when the first adrp
; dominates the second.
; The first adrp comes from the loading of 'a' and the second the loading of 'b'.
; 'a' is loaded in if.then, 'b' in if.end4, if.then does not dominates if.end4.
; CHECK-LABEL: _test
; CHECK: ret
; CHECK-NOT: .loh AdrpAdrp
define i32 @test(i32 %t) {
entry:
  %cmp = icmp sgt i32 %t, 5
  br i1 %cmp, label %if.then, label %if.end4

if.then:                                          ; preds = %entry
  %tmp = load i32* @a, align 4
  %add = add nsw i32 %tmp, %t
  %cmp1 = icmp sgt i32 %add, 12
  br i1 %cmp1, label %if.then2, label %if.end4

if.then2:                                         ; preds = %if.then
  tail call void @foo(i32 %add)
  %tmp1 = load i32* @a, align 4
  br label %if.end4

if.end4:                                          ; preds = %if.then2, %if.then, %entry
  %t.addr.0 = phi i32 [ %tmp1, %if.then2 ], [ %t, %if.then ], [ %t, %entry ]
  %tmp2 = load i32* @b, align 4
  %add5 = add nsw i32 %tmp2, %t.addr.0
  tail call void @foo(i32 %add5)
  %tmp3 = load i32* @b, align 4
  %add6 = add nsw i32 %tmp3, %t.addr.0
  ret i32 %add6
}
