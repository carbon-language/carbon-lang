; RUN: llvm-as < %s | llc -march=x86 | grep bsr
; RUN: llvm-as < %s | llc -march=x86 | grep bsf

define i32 @t1(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.ctlz.i32( i32 %x )
	ret i32 %tmp
}

declare i32 @llvm.ctlz.i32(i32) nounwind readnone 

define i32 @t2(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.cttz.i32( i32 %x )
	ret i32 %tmp
}

declare i32 @llvm.cttz.i32(i32) nounwind readnone 
