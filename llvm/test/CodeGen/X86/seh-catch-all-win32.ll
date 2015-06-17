; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s

; 32-bit catch-all has to use a filter function because that's how it saves the
; exception code.

@str = linkonce_odr unnamed_addr constant [27 x i8] c"GetExceptionCode(): 0x%lx\0A\00", align 1

declare i32 @_except_handler3(...)
declare void @crash()
declare i32 @printf(i8* nocapture readonly, ...) nounwind
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.framerecover(i8*, i8*, i32)
declare void @llvm.frameescape(...)
declare i8* @llvm.x86.seh.exceptioninfo(i8*, i8*)

define i32 @main() personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  %__exceptioncode = alloca i32, align 4
  call void (...) @llvm.frameescape(i32* %__exceptioncode)
  invoke void @crash() #5
          to label %__try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i32 ()* @"filt$main" to i8*)
  %1 = extractvalue { i8*, i32 } %0, 1
  %2 = call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ()* @"filt$main" to i8*)) #4
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %__except, label %eh.resume

__except:                                         ; preds = %lpad
  %3 = load i32, i32* %__exceptioncode, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @str, i32 0, i32 0), i32 %3) #4
  br label %__try.cont

__try.cont:                                       ; preds = %entry, %__except
  ret i32 0

eh.resume:                                        ; preds = %lpad
  resume { i8*, i32 } %0
}

define internal i32 @"filt$main"() {
entry:
  %0 = tail call i8* @llvm.frameaddress(i32 1)
  %1 = tail call i8* @llvm.framerecover(i8* bitcast (i32 ()* @main to i8*), i8* %0, i32 0)
  %__exceptioncode = bitcast i8* %1 to i32*
  %2 = tail call i8* @llvm.x86.seh.exceptioninfo(i8* bitcast (i32 ()* @main to i8*), i8* %0)
  %3 = bitcast i8* %2 to i32**
  %4 = load i32*, i32** %3, align 4
  %5 = load i32, i32* %4, align 4
  store i32 %5, i32* %__exceptioncode, align 4
  ret i32 1
}

; Check that we can get the exception code from eax to the printf.

; CHECK-LABEL: _main:
; CHECK: Lmain$frame_escape_0 = [[code_offs:[-0-9]+]]
; CHECK: Lmain$frame_escape_1 = [[reg_offs:[-0-9]+]]
; CHECK: movl %esp, [[reg_offs]](%ebp)
; CHECK: movl $L__ehtable$main,
; 	EH state 0
; CHECK: movl $0, -4(%ebp)
; CHECK: calll _crash
; CHECK: retl
; CHECK: # Block address taken
; 	stackrestore
; CHECK: movl [[reg_offs]](%ebp), %esp
; 	EH state -1
; CHECK: movl [[code_offs]](%ebp), %[[code:[a-z]+]]
; CHECK: movl $-1, -4(%ebp)
; CHECK-DAG: movl %[[code]], 4(%esp)
; CHECK-DAG: movl $_str, (%esp)
; CHECK: calll _printf

; CHECK: .section .xdata,"dr"
; CHECK: L__ehtable$main
; CHECK-NEXT: .long -1
; CHECK-NEXT: .long _filt$main
; CHECK-NEXT: .long Ltmp{{[0-9]+}}

; CHECK-LABEL: _filt$main:
; CHECK: movl
