; RUN: not llc -mtriple=armv7-windows-itanium -mcpu=cortex-a9 -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s -check-prefix CHECK-WIN

; RUN: not llc -mtriple=armv7-windows-gnu -mcpu=cortex-a9 -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s -check-prefix CHECK-GNU

; CHECK-WIN: does not support ARM mode execution

; CHECK-GNU: does not support ARM mode execution

