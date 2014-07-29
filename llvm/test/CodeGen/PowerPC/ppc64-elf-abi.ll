; RUN: llc -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefix=CHECK-ELFv1
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mattr=+elfv1 < %s | FileCheck %s -check-prefix=CHECK-ELFv1
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mattr=+elfv2 < %s | FileCheck %s -check-prefix=CHECK-ELFv2
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefix=CHECK-ELFv2
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mattr=+elfv1 < %s | FileCheck %s -check-prefix=CHECK-ELFv1
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mattr=+elfv2 < %s | FileCheck %s -check-prefix=CHECK-ELFv2

; CHECK-ELFv2: .abiversion 2
; CHECK-ELFv1-NOT: .abiversion 2

