; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; Test that we don't drop a block that has its address taken.

; CHECK: Ltmp0:                                  ## Block address taken
; CHECK: Ltmp1:                                  ## Block address taken

@a = common global i32 0, align 4
@p = common global i8* null, align 8

define void @foo() noreturn nounwind uwtable ssp {
entry:
  %tmp = load i32, i32* @a, align 4
  %foo = icmp eq i32 0, %tmp
  br i1 %foo, label %sw.bb, label %sw.default

sw.bb:                                            ; preds = %entry
  store i8* blockaddress(@foo, %sw.bb1), i8** @p, align 8
  br label %sw.bb1

sw.bb1:                                           ; preds = %sw.default, %sw.bb, %entry
  store i8* blockaddress(@foo, %sw.default), i8** @p, align 8
  br label %sw.default

sw.default:                                       ; preds = %sw.bb1, %entry
  store i8* blockaddress(@foo, %sw.bb1), i8** @p, align 8
  br label %sw.bb1
}
