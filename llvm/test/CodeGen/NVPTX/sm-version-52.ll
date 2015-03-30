; RUN: llc < %s -march=nvptx -mcpu=sm_52 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_52 | FileCheck %s


; CHECK: .target sm_52

