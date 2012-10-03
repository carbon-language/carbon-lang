; RUN: opt < %s -tsan -S | FileCheck %s
; Check that atomic memory operations are converted to calls into ThreadSanitizer runtime.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i8 @atomic8_load_unordered(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a unordered, align 1
  ret i8 %0
}
; CHECK: atomic8_load_unordered
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 100501)

define i8 @atomic8_load_monotonic(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a monotonic, align 1
  ret i8 %0
}
; CHECK: atomic8_load_monotonic
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 100501)

define i8 @atomic8_load_acquire(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a acquire, align 1
  ret i8 %0
}
; CHECK: atomic8_load_acquire
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 100504)

define i8 @atomic8_load_seq_cst(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a seq_cst, align 1
  ret i8 %0
}
; CHECK: atomic8_load_seq_cst
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 100532)

define void @atomic8_store_unordered(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a unordered, align 1
  ret void
}
; CHECK: atomic8_store_unordered
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 100501)

define void @atomic8_store_monotonic(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a monotonic, align 1
  ret void
}
; CHECK: atomic8_store_monotonic
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 100501)

define void @atomic8_store_release(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a release, align 1
  ret void
}
; CHECK: atomic8_store_release
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 100508)

define void @atomic8_store_seq_cst(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a seq_cst, align 1
  ret void
}
; CHECK: atomic8_store_seq_cst
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 100532)

define i16 @atomic16_load_unordered(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a unordered, align 2
  ret i16 %0
}
; CHECK: atomic16_load_unordered
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 100501)

define i16 @atomic16_load_monotonic(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a monotonic, align 2
  ret i16 %0
}
; CHECK: atomic16_load_monotonic
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 100501)

define i16 @atomic16_load_acquire(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a acquire, align 2
  ret i16 %0
}
; CHECK: atomic16_load_acquire
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 100504)

define i16 @atomic16_load_seq_cst(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a seq_cst, align 2
  ret i16 %0
}
; CHECK: atomic16_load_seq_cst
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 100532)

define void @atomic16_store_unordered(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a unordered, align 2
  ret void
}
; CHECK: atomic16_store_unordered
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 100501)

define void @atomic16_store_monotonic(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a monotonic, align 2
  ret void
}
; CHECK: atomic16_store_monotonic
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 100501)

define void @atomic16_store_release(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a release, align 2
  ret void
}
; CHECK: atomic16_store_release
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 100508)

define void @atomic16_store_seq_cst(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a seq_cst, align 2
  ret void
}
; CHECK: atomic16_store_seq_cst
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 100532)

define i32 @atomic32_load_unordered(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a unordered, align 4
  ret i32 %0
}
; CHECK: atomic32_load_unordered
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 100501)

define i32 @atomic32_load_monotonic(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a monotonic, align 4
  ret i32 %0
}
; CHECK: atomic32_load_monotonic
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 100501)

define i32 @atomic32_load_acquire(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a acquire, align 4
  ret i32 %0
}
; CHECK: atomic32_load_acquire
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 100504)

define i32 @atomic32_load_seq_cst(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a seq_cst, align 4
  ret i32 %0
}
; CHECK: atomic32_load_seq_cst
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 100532)

define void @atomic32_store_unordered(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a unordered, align 4
  ret void
}
; CHECK: atomic32_store_unordered
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 100501)

define void @atomic32_store_monotonic(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a monotonic, align 4
  ret void
}
; CHECK: atomic32_store_monotonic
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 100501)

define void @atomic32_store_release(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a release, align 4
  ret void
}
; CHECK: atomic32_store_release
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 100508)

define void @atomic32_store_seq_cst(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a seq_cst, align 4
  ret void
}
; CHECK: atomic32_store_seq_cst
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 100532)

define i64 @atomic64_load_unordered(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a unordered, align 8
  ret i64 %0
}
; CHECK: atomic64_load_unordered
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 100501)

define i64 @atomic64_load_monotonic(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a monotonic, align 8
  ret i64 %0
}
; CHECK: atomic64_load_monotonic
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 100501)

define i64 @atomic64_load_acquire(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a acquire, align 8
  ret i64 %0
}
; CHECK: atomic64_load_acquire
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 100504)

define i64 @atomic64_load_seq_cst(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a seq_cst, align 8
  ret i64 %0
}
; CHECK: atomic64_load_seq_cst
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 100532)

define void @atomic64_store_unordered(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a unordered, align 8
  ret void
}
; CHECK: atomic64_store_unordered
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 100501)

define void @atomic64_store_monotonic(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a monotonic, align 8
  ret void
}
; CHECK: atomic64_store_monotonic
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 100501)

define void @atomic64_store_release(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a release, align 8
  ret void
}
; CHECK: atomic64_store_release
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 100508)

define void @atomic64_store_seq_cst(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a seq_cst, align 8
  ret void
}
; CHECK: atomic64_store_seq_cst
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 100532)

define i128 @atomic128_load_unordered(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a unordered, align 16
  ret i128 %0
}
; CHECK: atomic128_load_unordered
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 100501)

define i128 @atomic128_load_monotonic(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a monotonic, align 16
  ret i128 %0
}
; CHECK: atomic128_load_monotonic
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 100501)

define i128 @atomic128_load_acquire(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a acquire, align 16
  ret i128 %0
}
; CHECK: atomic128_load_acquire
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 100504)

define i128 @atomic128_load_seq_cst(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a seq_cst, align 16
  ret i128 %0
}
; CHECK: atomic128_load_seq_cst
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 100532)

define void @atomic128_store_unordered(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a unordered, align 16
  ret void
}
; CHECK: atomic128_store_unordered
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 100501)

define void @atomic128_store_monotonic(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a monotonic, align 16
  ret void
}
; CHECK: atomic128_store_monotonic
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 100501)

define void @atomic128_store_release(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a release, align 16
  ret void
}
; CHECK: atomic128_store_release
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 100508)

define void @atomic128_store_seq_cst(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a seq_cst, align 16
  ret void
}
; CHECK: atomic128_store_seq_cst
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 100532)
