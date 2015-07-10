; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s

; 32-bit catch-all has to use a filter function because that's how it saves the
; exception code.

@str = linkonce_odr unnamed_addr constant [27 x i8] c"GetExceptionCode(): 0x%lx\0A\00", align 1

declare i32 @_except_handler3(...)
declare void @crash()
declare i32 @printf(i8* nocapture readonly, ...) nounwind
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.localrecover(i8*, i8*, i32)
declare void @llvm.localescape(...)
declare i8* @llvm.x86.seh.recoverfp(i8*, i8*)

define i32 @main() personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  %__exceptioncode = alloca i32, align 4
  call void (...) @llvm.localescape(i32* %__exceptioncode)
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
  %ebp = tail call i8* @llvm.frameaddress(i32 1)
  %parentfp = tail call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %ebp)
  %code.i8 = tail call i8* @llvm.localrecover(i8* bitcast (i32 ()* @main to i8*), i8* %parentfp, i32 0)
  %__exceptioncode = bitcast i8* %code.i8 to i32*
  %info.addr = getelementptr inbounds i8, i8* %ebp, i32 -20
  %0 = bitcast i8* %info.addr to i32***
  %1 = load i32**, i32*** %0, align 4
  %2 = load i32*, i32** %1, align 4
  %3 = load i32, i32* %2, align 4
  store i32 %3, i32* %__exceptioncode, align 4
  ret i32 1
}

; Check that we can get the exception code from eax to the printf.

; CHECK-LABEL: _main:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
;       Ensure that we push *all* the CSRs, since they are clobbered by the
;       __except block.
; CHECK: pushl %ebx
; CHECK: pushl %edi
; CHECK: pushl %esi

; CHECK: Lmain$frame_escape_0 = [[code_offs:[-0-9]+]]
; CHECK: Lmain$frame_escape_1 = [[reg_offs:[-0-9]+]]
; CHECK: movl %esp, [[reg_offs]](%ebp)
; CHECK: movl $L__ehtable$main,
; 	EH state 0
; CHECK: movl $0, -16(%ebp)
; CHECK: calll _crash
; CHECK: popl %esi
; CHECK: popl %edi
; CHECK: popl %ebx
; CHECK: retl
; CHECK: # Block address taken
; 	stackrestore
; CHECK: movl -24(%ebp), %esp
; 	EH state -1
; CHECK: movl [[code_offs]](%ebp), %[[code:[a-z]+]]
; CHECK: movl $-1, -16(%ebp)
; CHECK-DAG: movl %[[code]], 4(%esp)
; CHECK-DAG: movl $_str, (%esp)
; CHECK: calll _printf

; CHECK: .section .xdata,"dr"
; CHECK: Lmain$parent_frame_offset = Lmain$frame_escape_1
; CHECK: .align 4
; CHECK: L__ehtable$main
; CHECK-NEXT: .long -1
; CHECK-NEXT: .long _filt$main
; CHECK-NEXT: .long Ltmp{{[0-9]+}}

; CHECK-LABEL: _filt$main:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: movl (%ebp), %[[oldebp:[a-z]+]]
; CHECK: movl -20(%[[oldebp]]), %[[ehinfo:[a-z]+]]
; CHECK: movl (%[[ehinfo]]), %[[ehrec:[a-z]+]]
; CHECK: movl (%[[ehrec]]), %[[ehcode:[a-z]+]]
; CHECK: movl %[[ehcode]], {{.*}}(%{{.*}})
