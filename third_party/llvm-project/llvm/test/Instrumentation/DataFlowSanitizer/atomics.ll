; RUN: opt < %s -dfsan -S | FileCheck %s
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-instrument-with-call-threshold=0 -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i32 @AtomicRmwXchg(i32* %p, i32 %x) {
entry:
  ; COMM: atomicrmw xchg: store clean shadow/origin, return clean shadow/origin

  ; CHECK-LABEL:       @AtomicRmwXchg.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK-NEXT:        atomicrmw xchg i32* %p, i32 %x seq_cst
  ; CHECK-NEXT:        store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

  %0 = atomicrmw xchg i32* %p, i32 %x seq_cst
  ret i32 %0
}

define i32 @AtomicRmwMax(i32* %p, i32 %x) {
  ; COMM: atomicrmw max: exactly the same as above

  ; CHECK-LABEL:       @AtomicRmwMax.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK-NEXT:        atomicrmw max i32* %p, i32 %x seq_cst
  ; CHECK-NEXT:        store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

entry:
  %0 = atomicrmw max i32* %p, i32 %x seq_cst
  ret i32 %0
}


define i32 @Cmpxchg(i32* %p, i32 %a, i32 %b) {
  ; COMM: cmpxchg: store clean shadow/origin, return clean shadow/origin

  ; CHECK-LABEL:       @Cmpxchg.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK-NEXT:        %pair = cmpxchg i32* %p, i32 %a, i32 %b seq_cst seq_cst
  ; CHECK:             store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

entry:
  %pair = cmpxchg i32* %p, i32 %a, i32 %b seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}


define i32 @CmpxchgMonotonic(i32* %p, i32 %a, i32 %b) {
  ; COMM: relaxed cmpxchg: bump up to "release monotonic"

  ; CHECK-LABEL:       @CmpxchgMonotonic.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK-NEXT:        %pair = cmpxchg i32* %p, i32 %a, i32 %b release monotonic
  ; CHECK:             store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

entry:
  %pair = cmpxchg i32* %p, i32 %a, i32 %b monotonic monotonic
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}



define i32 @AtomicLoad(i32* %p) {
  ; COMM: atomic load: load shadow value after app value

  ; CHECK-LABEL:  @AtomicLoad.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK:        %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK:        %a = load atomic i32, i32* %p seq_cst, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to i[[#SBITS]]*
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to i32*
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, i32* %[[#ORIGIN_PTR]], align 16
  ; CHECK:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK:        load i[[#NUM_BITS]], i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:        %[[#AP_S:]] = or i[[#SBITS]] {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i[[#SBITS]] %[[#AP_S]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, i32* %p seq_cst, align 16
  ret i32 %a
}


define i32 @AtomicLoadAcquire(i32* %p) {
  ; COMM: atomic load: load shadow value after app value

  ; CHECK-LABEL:  @AtomicLoadAcquire.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK:        %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK:        %a = load atomic i32, i32* %p acquire, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to i[[#SBITS]]*
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to i32*
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, i32* %[[#ORIGIN_PTR]], align 16
  ; CHECK:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK:        load i[[#NUM_BITS]], i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:        %[[#AP_S:]] = or i[[#SBITS]] {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i[[#SBITS]] %[[#AP_S]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, i32* %p acquire, align 16
  ret i32 %a
}


define i32 @AtomicLoadMonotonic(i32* %p) {
  ; COMM: atomic load monotonic: bump up to load acquire

  ; CHECK-LABEL:  @AtomicLoadMonotonic.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK:        %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK:        %a = load atomic i32, i32* %p acquire, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to i[[#SBITS]]*
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to i32*
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, i32* %[[#ORIGIN_PTR]], align 16
  ; CHECK:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK:        load i[[#NUM_BITS]], i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:        %[[#AP_S:]] = or i[[#SBITS]] {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i[[#SBITS]] %[[#AP_S]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, i32* %p monotonic, align 16
  ret i32 %a
}

define i32 @AtomicLoadUnordered(i32* %p) {
  ; COMM: atomic load unordered: bump up to load acquire

  ; CHECK-LABEL:  @AtomicLoadUnordered.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK:        %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK:        %a = load atomic i32, i32* %p acquire, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to i[[#SBITS]]*
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to i32*
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, i32* %[[#ORIGIN_PTR]], align 16
  ; CHECK:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK:        load i[[#NUM_BITS]], i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:        %[[#AP_S:]] = or i[[#SBITS]] {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i[[#SBITS]] %[[#AP_S]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, i32* %p unordered, align 16
  ret i32 %a
}

define void @AtomicStore(i32* %p, i32 %x) {
  ; COMM: atomic store: store clean shadow value before app value

  ; CHECK-LABEL:       @AtomicStore.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:             store atomic i32 %x, i32* %p seq_cst, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, i32* %p seq_cst, align 16
  ret void
}

define void @AtomicStoreRelease(i32* %p, i32 %x) {
  ; COMM: atomic store: store clean shadow value before app value

  ; CHECK-LABEL:       @AtomicStoreRelease.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:             store atomic i32 %x, i32* %p release, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, i32* %p release, align 16
  ret void
}

define void @AtomicStoreMonotonic(i32* %p, i32 %x) {
  ; COMM: atomic store monotonic: bumped up to store release

  ; CHECK-LABEL:       @AtomicStoreMonotonic.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:             store atomic i32 %x, i32* %p release, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, i32* %p monotonic, align 16
  ret void
}

define void @AtomicStoreUnordered(i32* %p, i32 %x) {
  ; COMM: atomic store unordered: bumped up to store release

  ; CHECK-LABEL: @AtomicStoreUnordered.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint i32* %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to i[[#SBITS]]*
  ; CHECK-NEXT:        %[[#SHADOW_PTR64:]] = bitcast i[[#SBITS]]* %[[#SHADOW_PTR]] to i[[#NUM_BITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SHADOW_PTR64]], align [[#SBYTES]]
  ; CHECK:             store atomic i32 %x, i32* %p release, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, i32* %p unordered, align 16
  ret void
}
