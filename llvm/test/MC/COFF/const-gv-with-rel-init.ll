; RUN: llc -mtriple x86_64-pc-windows-msvc < %s | FileCheck %s

define void @f() {
  ret void
}

@ptr = constant void ()* @f, section ".CRT$XLB", align 8
; CHECK:  .section  .CRT$XLB,"rd"
