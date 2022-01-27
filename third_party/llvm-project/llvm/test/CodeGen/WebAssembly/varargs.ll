; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=EMSCRIPTEN
; RUN: llc < %s -mtriple wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -verify-machineinstrs | FileCheck %s  --check-prefix=CHECK --check-prefix=UNKNOWN

; Test varargs constructs.

target triple = "wasm32-unknown-emscripten"

; Test va_start.

; TODO: Test va_start.
; CHECK-LABEL: start:
; CHECK-NEXT: .functype start (i32, i32) -> ()
; CHECK-NOT: __stack_pointer
define void @start(i8** %ap, ...) {
entry:
  %0 = bitcast i8** %ap to i8*
; Store the second argument (the hidden vararg buffer pointer) into ap
; CHECK: i32.store 0($0), $1
  call void @llvm.va_start(i8* %0)
  ret void
}

; Test va_end.

; CHECK-LABEL: end:
; CHECK-NEXT: .functype end (i32) -> (){{$}}
; CHECK-NEXT: return{{$}}
define void @end(i8** %ap) {
entry:
  %0 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %0)
  ret void
}

; Test va_copy.

; CHECK-LABEL: copy:
; CHECK-NEXT: .functype copy (i32, i32) -> (){{$}}
; CHECK-NEXT: i32.load  $push0=, 0($1){{$}}
; CHECK-NEXT: i32.store 0($0), $pop0{{$}}
; CHECK-NEXT: return{{$}}
define void @copy(i8** %ap, i8** %bp) {
entry:
  %0 = bitcast i8** %ap to i8*
  %1 = bitcast i8** %bp to i8*
  call void @llvm.va_copy(i8* %0, i8* %1)
  ret void
}

; Test va_arg with an i8 argument.

; CHECK-LABEL: arg_i8:
; CHECK-NEXT: .functype arg_i8 (i32) -> (i32){{$}}
; CHECK-NEXT: i32.load   $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: local.tee  $push[[NUM1:[0-9]+]]=, $1=, $pop[[NUM0]]{{$}}
; CHECK-NEXT: i32.const  $push[[NUM2:[0-9]+]]=, 4{{$}}
; CHECK-NEXT: i32.add    $push[[NUM3:[0-9]+]]=, $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.store  0($0), $pop[[NUM3]]{{$}}
; CHECK-NEXT: i32.load   $push[[NUM4:[0-9]+]]=, 0($1){{$}}
; CHECK-NEXT: return     $pop[[NUM4]]{{$}}
define i8 @arg_i8(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i8
  ret i8 %t
}

; Test va_arg with an i32 argument.

; CHECK-LABEL: arg_i32:
; CHECK-NEXT: .functype arg_i32 (i32) -> (i32){{$}}
; CHECK-NEXT: i32.load   $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i32.const  $push[[NUM1:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: i32.add    $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: i32.const  $push[[NUM3:[0-9]+]]=, -4{{$}}
; CHECK-NEXT: i32.and    $push[[NUM4:[0-9]+]]=, $pop[[NUM2]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: local.tee  $push[[NUM5:[0-9]+]]=, $1=, $pop[[NUM4]]{{$}}
; CHECK-NEXT: i32.const  $push[[NUM6:[0-9]+]]=, 4{{$}}
; CHECK-NEXT: i32.add    $push[[NUM7:[0-9]+]]=, $pop[[NUM5]], $pop[[NUM6]]{{$}}
; CHECK-NEXT: i32.store  0($0), $pop[[NUM7]]{{$}}
; CHECK-NEXT: i32.load   $push[[NUM8:[0-9]+]]=, 0($1){{$}}
; CHECK-NEXT: return     $pop[[NUM8]]{{$}}
define i32 @arg_i32(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i32
  ret i32 %t
}

; Test va_arg with an i128 argument.

; CHECK-LABEL: arg_i128:
; CHECK-NEXT: .functype arg_i128 (i32, i32) -> (){{$}}
; CHECK: i32.and
; CHECK: i64.load
; CHECK: i64.load
; CHECK: return{{$}}
define i128 @arg_i128(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i128
  ret i128 %t
}

; Test a varargs call with no actual arguments.

declare void @callee(...)

; CHECK-LABEL: caller_none:
; CHECK:      i32.const $push0=, 0
; CHECK-NEXT: call callee, $pop0
; CHECK-NEXT: return{{$}}
define void @caller_none() {
  call void (...) @callee()
  ret void
}

; Test a varargs call with some actual arguments.
; Note that the store of 2.0 is converted to an i64 store; this optimization
; is not needed on WebAssembly, but there isn't currently a convenient hook for
; disabling it.

; CHECK-LABEL: caller_some
; CHECK-DAG: i32.store
; CHECK-DAG: i64.store
define void @caller_some() {
  call void (...) @callee(i32 0, double 2.0)
  ret void
}

; Test a va_start call in a non-entry block
; CHECK-LABEL: startbb:
; CHECK: .functype startbb (i32, i32, i32) -> ()
define void @startbb(i1 %cond, i8** %ap, ...) {
entry:
  br i1 %cond, label %bb0, label %bb1
bb0:
  ret void
bb1:
  %0 = bitcast i8** %ap to i8*
; Store the second argument (the hidden vararg buffer pointer) into ap
; CHECK: i32.store 0($1), $2
  call void @llvm.va_start(i8* %0)
  ret void
}

; Test a call to a varargs function with a non-legal fixed argument.

declare void @callee_with_nonlegal_fixed(fp128, ...) nounwind

; CHECK-LABEL: call_nonlegal_fixed:
; CHECK: i64.const       $push[[L0:[0-9]+]]=, 0
; CHECK: i64.const       $push[[L1:[0-9]+]]=, 0
; CHECK: i32.const       $push[[L2:[0-9]+]]=, 0
; CHECK: call            callee_with_nonlegal_fixed, $pop[[L0]], $pop[[L1]], $pop[[L2]]{{$}}
define void @call_nonlegal_fixed() nounwind {
  call void (fp128, ...) @callee_with_nonlegal_fixed(fp128 0xL00000000000000000000000000000000)
  ret void
}

; Test a definition a varargs function with a non-legal fixed argument.

; CHECK-LABEL: nonlegal_fixed:
; CHECK-NEXT: .functype nonlegal_fixed (i64, i64, i32) -> (){{$}}
define void @nonlegal_fixed(fp128 %x, ...) nounwind {
  ret void
}

; Test that an fp128 argument is properly aligned and allocated
; within a vararg buffer.

; EMSCRIPTEN-LABEL: call_fp128_alignment:
; EMSCRIPTEN:      global.get      $push5=, __stack_pointer
; EMSCRIPTEN-NEXT: i32.const       $push6=, 32
; EMSCRIPTEN-NEXT: i32.sub         $push10=, $pop5, $pop6
; EMSCRIPTEN-NEXT: local.tee       $push9=, $1=, $pop10
; EMSCRIPTEN-NEXT: global.set      __stack_pointer, $pop9
; EMSCRIPTEN-NEXT: i32.const       $push0=, 16
; EMSCRIPTEN-NEXT: i32.add         $push1=, $1, $pop0
; EMSCRIPTEN-NEXT: i64.const       $push2=, -9223372036854775808
; EMSCRIPTEN-NEXT: i64.store       0($pop1), $pop2
; EMSCRIPTEN-NEXT: i64.const       $push3=, 1
; EMSCRIPTEN-NEXT: i64.store       8($1), $pop3
; EMSCRIPTEN-NEXT: i32.const       $push4=, 7
; EMSCRIPTEN-NEXT: i32.store       0($1), $pop4
; EMSCRIPTEN-NEXT: call            callee, $1

; Alignment of fp128 is a current disagreement between emscripten and others.
; UNKNOWN-LABEL: call_fp128_alignment:
; UNKNOWN:      global.get      $push7=, __stack_pointer
; UNKNOWN-NEXT: i32.const       $push8=, 32
; UNKNOWN-NEXT: i32.sub         $push12=, $pop7, $pop8
; UNKNOWN-NEXT: local.tee       $push11=, $1=, $pop12
; UNKNOWN-NEXT: global.set      __stack_pointer, $pop11
; UNKNOWN-NEXT: i32.const       $push0=, 24
; UNKNOWN-NEXT: i32.add         $push1=, $1, $pop0
; UNKNOWN-NEXT: i64.const       $push2=, -9223372036854775808
; UNKNOWN-NEXT: i64.store       0($pop1), $pop2
; UNKNOWN-NEXT: i32.const       $push3=, 16
; UNKNOWN-NEXT: i32.add         $push4=, $1, $pop3
; UNKNOWN-NEXT: i64.const       $push5=, 1
; UNKNOWN-NEXT: i64.store       0($pop4), $pop5
; UNKNOWN-NEXT: i32.const       $push6=, 7
; UNKNOWN-NEXT: i32.store       0($1), $pop6
; UNKNOWN-NEXT: call            callee, $1
define void @call_fp128_alignment(i8* %p) {
entry:
  call void (...) @callee(i8 7, fp128 0xL00000000000000018000000000000000)
  ret void
}

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.va_copy(i8*, i8*)
