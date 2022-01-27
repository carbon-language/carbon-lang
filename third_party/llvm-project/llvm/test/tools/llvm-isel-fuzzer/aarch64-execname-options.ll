; If the binary looks up libraries using an rpath, we can't test this
; without copying the whole lib dir or polluting the build dir.
; REQUIRES: static-libs
; REQUIRES: aarch64-registered-target

; The above also applies if the binary is built with libc++.
; UNSUPPORTED: libcxx-used

; RUN: echo > %t.input

; workaround for https://openradar.appspot.com/FB8914243
; RUN: rm -f %t.bin--aarch64
; RUN: rm -f %t.bin--aarch64-O1
; RUN: rm -f %t.bin--O3-aarch64

; RUN: cp llvm-isel-fuzzer %t.bin--aarch64
; RUN: %t.bin--aarch64 %t.input 2>&1 | FileCheck -check-prefix=AARCH64 %s
; AARCH64: Injected args: -mtriple=aarch64

; RUN: cp llvm-isel-fuzzer %t.bin--aarch64-O1
; RUN: %t.bin--aarch64-O1 %t.input 2>&1 | FileCheck -check-prefix=OPT-AFTER %s
; OPT-AFTER: Injected args: -mtriple=aarch64 -O1

; RUN: cp llvm-isel-fuzzer %t.bin--O3-aarch64
; RUN: %t.bin--O3-aarch64 %t.input 2>&1 | FileCheck -check-prefix=OPT-BEFORE %s
; OPT-BEFORE: Injected args: -O3 -mtriple=aarch64
