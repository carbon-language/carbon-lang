target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$comdat = comdat any

define void @f2() {
  call void @comdat()
  ret void
}

define linkonce_odr void @comdat() comdat {
  ret void
}
