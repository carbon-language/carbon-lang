; RUN: llc < %s -march=nvptx -mcpu=sm_32 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_32 | FileCheck %s


; CHECK: .version 4.0
; CHECK: .target sm_32

