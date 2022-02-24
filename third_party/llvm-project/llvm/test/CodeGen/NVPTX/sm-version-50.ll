; RUN: llc < %s -march=nvptx -mcpu=sm_50 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_50 | FileCheck %s


; CHECK: .version 4.0
; CHECK: .target sm_50

