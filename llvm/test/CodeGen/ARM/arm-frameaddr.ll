; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin  | grep mov | grep r7
; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi | grep mov | grep r11
; PR4344
; PR4416

define arm_aapcscc i8* @t() nounwind {
entry:
	%0 = call i8* @llvm.frameaddress(i32 0)
        ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
