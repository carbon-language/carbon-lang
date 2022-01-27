; RUN: llc < %s -march=nvptx -mcpu=sm_30 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s


; CHECK: .version 3.2
; CHECK: .target sm_30

