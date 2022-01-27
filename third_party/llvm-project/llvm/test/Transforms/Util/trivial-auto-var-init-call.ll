; RUN: opt -annotation-remarks -o /dev/null -S -pass-remarks-output=%t.opt.yaml %s -pass-remarks-missed=annotation-remarks 2>&1 | FileCheck %s
; RUN: cat %t.opt.yaml | FileCheck -check-prefix=YAML %s

; Emit remarks for memcpy, memmove, memset, bzero.
define void @known_call(i8* %src, i8* %dst, i64 %size) {
; CHECK: Call to memset inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memset
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 %size, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memcpy
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memmove
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitCall
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          bzero
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT: ...
  call void @bzero(i8* %dst, i64 %size), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitCall
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memset
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '32'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT: ...
  call i8* @memset(i8* %dst, i32 0, i64 32), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes.
define void @known_call_with_size(i8* %src, i8* %dst) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 32 bytes.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memset
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '32'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 32, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 32 bytes.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memcpy
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '32'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 32, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 32 bytes.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memmove
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '32'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 32, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 32 bytes.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitCall
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          bzero
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '32'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT: ...
  call void @bzero(i8* %dst, i64 32), !annotation !0, !dbg !DILocation(scope: !4)

  ret void
}

; Emit remarks for memcpy, memmove, memset marked volatile.
define void @known_call_volatile(i8* %src, i8* %dst, i64 %size) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Volatile: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_volatile
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memset
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 %size, i1 true), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Volatile: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_volatile
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memcpy
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 true), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Volatile: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_volatile
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memmove
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 true), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset marked atomic.
define void @known_call_atomic(i8* %src, i8* %dst, i64 %size) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Atomic: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_atomic
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memset
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %dst, i8 0, i64 %size, i32 1), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Atomic: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_atomic
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memcpy
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 %size, i32 1), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Atomic: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_atomic
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memmove
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 %size, i32 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; an alloca.
define void @known_call_with_size_alloca(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size_alloca
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memset
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '1'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          "\n Written Variables: "
; YAML-NEXT:   - WVarName:        dst
; YAML-NEXT:   - String:          ' ('
; YAML-NEXT:   - WVarSize:        '1'
; YAML-NEXT:   - String:          ' bytes)'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  %dst = alloca i8
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size_alloca
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memcpy
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '1'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          "\n Written Variables: "
; YAML-NEXT:   - WVarName:        dst
; YAML-NEXT:   - String:          ' ('
; YAML-NEXT:   - WVarSize:        '1'
; YAML-NEXT:   - String:          ' bytes)'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitIntrinsic
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size_alloca
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          memmove
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '1'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          "\n Written Variables: "
; YAML-NEXT:   - WVarName:        dst
; YAML-NEXT:   - String:          ' ('
; YAML-NEXT:   - WVarSize:        '1'
; YAML-NEXT:   - String:          ' bytes)'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Inlined: '
; YAML-NEXT:   - StoreInlined:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:   'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitCall
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call_with_size_alloca
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Call to '
; YAML-NEXT:   - Callee:          bzero
; YAML-NEXT:   - String:          ' inserted by -ftrivial-auto-var-init.'
; YAML-NEXT:   - String:          ' Memory operation size: '
; YAML-NEXT:   - StoreSize:       '1'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          "\n Written Variables: "
; YAML-NEXT:   - WVarName:        dst
; YAML-NEXT:   - String:          ' ('
; YAML-NEXT:   - WVarSize:        '1'
; YAML-NEXT:   - String:          ' bytes)'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  call void @bzero(i8* %dst, i64 1), !annotation !0, !dbg !DILocation(scope: !4)

  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; an alloca through a GEP.
define void @known_call_with_size_alloca_gep(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
  %dst = alloca i8
  %gep = getelementptr i8, i8* %dst, i32 0
  call void @llvm.memset.p0i8.i64(i8* %gep, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %gep, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %gep, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (1 bytes).
  call void @bzero(i8* %gep, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; an alloca through a GEP in an array.
define void @known_call_with_size_alloca_gep_array(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  %dst = alloca [2 x i8]
  %gep = getelementptr [2 x i8], [2 x i8]* %dst, i64 0, i64 0
  call void @llvm.memset.p0i8.i64(i8* %gep, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %gep, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %gep, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  call void @bzero(i8* %gep, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; an alloca through a bitcast.
define void @known_call_with_size_alloca_bitcast(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  %dst = alloca [2 x i8]
  %bc = bitcast [2 x i8]* %dst to i8*
  call void @llvm.memset.p0i8.i64(i8* %bc, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bc, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %bc, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst (2 bytes).
  call void @bzero(i8* %bc, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to an alloca that has a DILocalVariable attached.
define void @known_call_with_size_alloca_di(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  %dst = alloca i8
  call void @llvm.dbg.declare(metadata i8* %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  call void @bzero(i8* %dst, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; an alloca that has more than one DILocalVariable attached.
define void @known_call_with_size_alloca_di_multiple(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  %dst = alloca i8
  call void @llvm.dbg.declare(metadata i8* %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: destination (1 bytes).
  call void @bzero(i8* %dst, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; a PHI node that can be two different allocas.
define void @known_call_with_size_alloca_phi(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), dst (1 bytes).
entry:
  %dst = alloca i8
  %dst2 = alloca i8
  %cmp = icmp eq i32 undef, undef
  br i1 %cmp, label %l0, label %l1
l0:
  br label %l2
l1:
  br label %l2
l2:
  %phidst = phi i8* [ %dst, %l0 ], [ %dst2, %l1 ]
  call void @llvm.memset.p0i8.i64(i8* %phidst, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), dst (1 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %phidst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), dst (1 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %phidst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), dst (1 bytes).
  call void @bzero(i8* %phidst, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit remarks for memcpy, memmove, memset, bzero with known constant sizes to
; a PHI node that can be two different allocas, where one of it has multiple
; DILocalVariable.
define void @known_call_with_size_alloca_phi_di_multiple(i8* %src) {
; CHECK-NEXT: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), destination2 (1 bytes), destination (1 bytes).
entry:
  %dst = alloca i8
  %dst2 = alloca i8
  call void @llvm.dbg.declare(metadata i8* %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata i8* %dst, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  %cmp = icmp eq i32 undef, undef
  br i1 %cmp, label %l0, label %l1
l0:
  br label %l2
l1:
  br label %l2
l2:
  %phidst = phi i8* [ %dst, %l0 ], [ %dst2, %l1 ]
  call void @llvm.memset.p0i8.i64(i8* %phidst, i8 0, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memcpy inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), destination2 (1 bytes), destination (1 bytes).
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %phidst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to memmove inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), destination2 (1 bytes), destination (1 bytes).
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %phidst, i8* %src, i64 1, i1 false), !annotation !0, !dbg !DILocation(scope: !4)
; CHECK-NEXT: Call to bzero inserted by -ftrivial-auto-var-init. Memory operation size: 1 bytes.
; CHECK-NEXT: Variables: dst2 (1 bytes), destination2 (1 bytes), destination (1 bytes).
  call void @bzero(i8* %phidst, i64 1), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone speculatable willreturn
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) argmemonly nounwind willreturn writeonly
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1 immarg) argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) argmemonly nounwind willreturn

declare void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* nocapture writeonly, i8, i64, i32 immarg) argmemonly nounwind willreturn writeonly
declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32 immarg) argmemonly nounwind willreturn
declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32 immarg) argmemonly nounwind willreturn

declare void @bzero(i8* nocapture, i64) nofree nounwind
declare i8* @memset(i8*, i32, i64)

!llvm.module.flags = !{!1}
!0 = !{ !"auto-init" }
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
!3 = !DIFile(filename: "file", directory: "")
!4 = distinct !DISubprogram(name: "function", scope: !3, file: !3, unit: !2)
!5 = !DIBasicType(name: "byte", size: 8)
!6 = !DILocalVariable(name: "destination", scope: !4, file: !3, type: !5)
!7 = !DILocalVariable(name: "destination2", scope: !4, file: !3, type: !5)
