; RUN: llc -march=ppc64 < %s | FileCheck %s -check-prefix=CHECK-ELFv1
; RUN: llc -march=ppc64 -mattr=+elfv1 < %s | FileCheck %s -check-prefix=CHECK-ELFv1
; RUN: llc -march=ppc64 -mattr=+elfv2 < %s | FileCheck %s -check-prefix=CHECK-ELFv2
; RUN: llc -march=ppc64le < %s | FileCheck %s -check-prefix=CHECK-ELFv2
; RUN: llc -march=ppc64le -mattr=+elfv1 < %s | FileCheck %s -check-prefix=CHECK-ELFv1
; RUN: llc -march=ppc64le -mattr=+elfv2 < %s | FileCheck %s -check-prefix=CHECK-ELFv2

; CHECK-ELFv2: .abiversion 2
; CHECK-ELFv1-NOT: .abiversion 2

