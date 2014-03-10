; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64

@.str = private unnamed_addr constant [6 x i8] c"%d %f\00", align 1
@.str.again = private unnamed_addr constant [6 x i8] c"%d %f\00", align 1

; PTX32-NOT: .str
; PTX64-NOT: .str
