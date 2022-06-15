; REQUIRES: aarch64

; RUN: rm -rf %t; mkdir %t

; RUN: llc -filetype=obj %s -O3 -o %t/icf-obj.o -enable-machine-outliner=never -mtriple arm64-apple-macos -addrsig
; RUN: %lld -arch arm64 -lSystem --icf=safe -dylib -o %t/icf-safe.dylib %t/icf-obj.o
; RUN: %lld -arch arm64 -lSystem --icf=all  -dylib -o %t/icf-all.dylib  %t/icf-obj.o
; RUN: llvm-objdump %t/icf-safe.dylib -d --macho | FileCheck %s --check-prefix=ICFSAFE
; RUN: llvm-objdump %t/icf-all.dylib  -d --macho | FileCheck %s --check-prefix=ICFALL

; RUN: llvm-as %s -o %t/icf-bitcode.o
; RUN: %lld -arch arm64 -lSystem --icf=safe -dylib -o %t/icf-safe-bitcode.dylib %t/icf-bitcode.o
; RUN: %lld -arch arm64 -lSystem --icf=all  -dylib -o %t/icf-all-bitcode.dylib %t/icf-bitcode.o
; RUN: llvm-objdump %t/icf-safe-bitcode.dylib -d --macho | FileCheck %s --check-prefix=ICFSAFE
; RUN: llvm-objdump %t/icf-all-bitcode.dylib  -d --macho | FileCheck %s --check-prefix=ICFALL

; ICFSAFE-LABEL:  _callAllFunctions
; ICFSAFE:        bl _func02
; ICFSAFE-NEXT:   bl _func02
; ICFSAFE-NEXT:   bl _func03_takeaddr

; ICFALL-LABEL:   _callAllFunctions
; ICFALL:         bl _func03_takeaddr
; ICFALL-NEXT:    bl _func03_takeaddr
; ICFALL-NEXT:    bl _func03_takeaddr

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macos11.0"

@result = global i32 0, align 4

define void @func01() local_unnamed_addr noinline {
entry:
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, 1
  store volatile i32 %add, ptr @result, align 4
  ret void
}

define void @func02() local_unnamed_addr noinline {
entry:
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, 1
  store volatile i32 %add, ptr @result, align 4
  ret void
}

define void @func03_takeaddr() noinline {
entry:
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, 1
  store volatile i32 %add, ptr @result, align 4
  ret void
}

define void @callAllFunctions() local_unnamed_addr {
entry:
  tail call void @func01()
  tail call void @func02()
  tail call void @func03_takeaddr()
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, ptrtoint (ptr @func03_takeaddr to i32)
  store volatile i32 %add, ptr @result, align 4
  ret void
}
