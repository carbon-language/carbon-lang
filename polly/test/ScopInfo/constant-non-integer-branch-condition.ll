; RUN: opt %loadPolly -analyze -polly-scops %s | FileCheck %s
;
; At some point this caused a problem in the domain generation as we
; assumed any constant branch condition to be valid. However, only constant
; integers are interesting and can be handled.
;
; CHECK: Stmt_entry_split__TO__cleanup
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @main(i32* %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br i1 icmp ne (i32 (...)* @test_weak, i32 (...)* null), label %if.then, label %cleanup

if.then:                                          ; preds = %entry.split
  store i32 0, i32* %A
  br label %cleanup

cleanup:                                          ; preds = %if.then, %entry.split
  ret i32 0
}

declare extern_weak i32 @test_weak(...)
