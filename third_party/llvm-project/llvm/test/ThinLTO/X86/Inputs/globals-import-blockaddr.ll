target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@label_addr = internal constant [1 x i8*] [i8* blockaddress(@bar, %lb)], align 8

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local [1 x i8*]* @foo() {
  ret [1 x i8*]* @label_addr
}

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local [1 x i8*]* @bar() {
  br label %lb

lb:
  ret [1 x i8*]* @label_addr
}
