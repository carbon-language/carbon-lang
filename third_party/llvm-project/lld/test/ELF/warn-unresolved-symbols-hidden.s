# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: not ld.lld -shared %t.o -o /dev/null -z defs --warn-unresolved-symbols 2>&1| FileCheck %s

# CHECK: warning: undefined symbol: foo
# CHECK: error: undefined hidden symbol: bar
# CHECK: error: undefined protected symbol: zed

.data
.quad foo
.hidden bar
.quad bar
.protected zed
.quad zed
