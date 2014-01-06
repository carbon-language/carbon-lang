; RUN: not llc < %s -O0 2> %t1
; RUN: FileCheck %s < %t1
; XFAIL: ppc64,mips,s390x
; PR18363

; CHECK: argument to '__builtin_return_address' must be a constant integer

define i32* @foo() {
entry:
  %t1 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 0, i32 0)
  %t2 = extractvalue { i32, i1 } %t1, 0
  %t3 = extractvalue { i32, i1 } %t1, 1
  br i1 %t3, label %cont, label %trap

trap:
  call void @llvm.trap()
  unreachable

cont:
  %t5 = call i8* @llvm.returnaddress(i32 %t2)
  %t6 = bitcast i8* %t5 to i32*
  ret i32* %t6
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

declare void @llvm.trap()

declare i8* @llvm.returnaddress(i32)
