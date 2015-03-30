; RUN: llc < %s -march=nvptx -mcpu=sm_37 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_37 | FileCheck %s


; CHECK: .target sm_37

