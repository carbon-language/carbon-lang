; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

; This file contains tests for the AArch64 load/store optimizer.

%padding = type { i8*, i8*, i8*, i8* }
%s.word = type { i32, i32 }
%s.doubleword = type { i64, i32 }
%s.quadword = type { fp128, i32 }
%s.float = type { float, i32 }
%s.double = type { double, i32 }
%struct.word = type { %padding, %s.word }
%struct.doubleword = type { %padding, %s.doubleword }
%struct.quadword = type { %padding, %s.quadword }
%struct.float = type { %padding, %s.float }
%struct.double = type { %padding, %s.double }

; Check the following transform:
;
; (ldr|str) X, [x0, #32]
;  ...
; add x0, x0, #32
;  ->
; (ldr|str) X, [x0, #32]!
;
; with X being either w1, x1, s0, d0 or q0.

declare void @bar_word(%s.word*, i32)

define void @load-pre-indexed-word(%struct.word* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1, i32 0
  %add = load i32* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1
  tail call void @bar_word(%s.word* %c, i32 %add)
  ret void
}

define void @store-pre-indexed-word(%struct.word* %ptr, i32 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-word
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1, i32 0
  store i32 %val, i32* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1
  tail call void @bar_word(%s.word* %c, i32 %val)
  ret void
}

declare void @bar_doubleword(%s.doubleword*, i64)

define void @load-pre-indexed-doubleword(%struct.doubleword* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1, i32 0
  %add = load i64* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1
  tail call void @bar_doubleword(%s.doubleword* %c, i64 %add)
  ret void
}

define void @store-pre-indexed-doubleword(%struct.doubleword* %ptr, i64 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-doubleword
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1, i32 0
  store i64 %val, i64* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1
  tail call void @bar_doubleword(%s.doubleword* %c, i64 %val)
  ret void
}

declare void @bar_quadword(%s.quadword*, fp128)

define void @load-pre-indexed-quadword(%struct.quadword* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1, i32 0
  %add = load fp128* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1
  tail call void @bar_quadword(%s.quadword* %c, fp128 %add)
  ret void
}

define void @store-pre-indexed-quadword(%struct.quadword* %ptr, fp128 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-quadword
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1, i32 0
  store fp128 %val, fp128* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1
  tail call void @bar_quadword(%s.quadword* %c, fp128 %val)
  ret void
}

declare void @bar_float(%s.float*, float)

define void @load-pre-indexed-float(%struct.float* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1, i32 0
  %add = load float* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1
  tail call void @bar_float(%s.float* %c, float %add)
  ret void
}

define void @store-pre-indexed-float(%struct.float* %ptr, float %val) nounwind {
; CHECK-LABEL: store-pre-indexed-float
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1, i32 0
  store float %val, float* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1
  tail call void @bar_float(%s.float* %c, float %val)
  ret void
}

declare void @bar_double(%s.double*, double)

define void @load-pre-indexed-double(%struct.double* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1, i32 0
  %add = load double* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1
  tail call void @bar_double(%s.double* %c, double %add)
  ret void
}

define void @store-pre-indexed-double(%struct.double* %ptr, double %val) nounwind {
; CHECK-LABEL: store-pre-indexed-double
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1, i32 0
  store double %val, double* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1
  tail call void @bar_double(%s.double* %c, double %val)
  ret void
}
