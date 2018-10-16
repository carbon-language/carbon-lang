; ModuleID = 'import_stats2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hot() {
  ret void
}
define void @critical() {
  ret void
}
define void @none() {
  ret void
}
