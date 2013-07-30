; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define i32 @main() {
; CHECK-LABEL: @main(
; CHECK: call i32 bitcast
entry:
	%tmp = call i32 bitcast (i7* (i999*)* @ctime to i32 (i99*)*)( i99* null )
	ret i32 %tmp
}

define i7* @ctime(i999*) {
; CHECK-LABEL: @ctime(
; CHECK: call i7* bitcast
entry:
	%tmp = call i7* bitcast (i32 ()* @main to i7* ()*)( )
	ret i7* %tmp
}
