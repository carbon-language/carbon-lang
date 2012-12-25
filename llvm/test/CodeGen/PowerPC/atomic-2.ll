; RUN: llc < %s -march=ppc64 | FileCheck %s

define i64 @exchange_and_add(i64* %mem, i64 %val) nounwind {
; CHECK: exchange_and_add:
; CHECK: ldarx
  %tmp = atomicrmw add i64* %mem, i64 %val monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define i64 @exchange_and_cmp(i64* %mem) nounwind {
; CHECK: exchange_and_cmp:
; CHECK: ldarx
  %tmp = cmpxchg i64* %mem, i64 0, i64 1 monotonic
; CHECK: stdcx.
; CHECK: stdcx.
  ret i64 %tmp
}

define i64 @exchange(i64* %mem, i64 %val) nounwind {
; CHECK: exchange:
; CHECK: ldarx
  %tmp = atomicrmw xchg i64* %mem, i64 1 monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define void @atomic_store(i64* %mem, i64 %val) nounwind {
entry:
; CHECK: @atomic_store
  store atomic i64 %val, i64* %mem release, align 64
; CHECK: ldarx
; CHECK: stdcx.
  ret void
}

define i64 @atomic_load(i64* %mem) nounwind {
entry:
; CHECK: @atomic_load
  %tmp = load atomic i64* %mem acquire, align 64
; CHECK: ldarx
; CHECK: stdcx.
; CHECK: stdcx.
  ret i64 %tmp
}

