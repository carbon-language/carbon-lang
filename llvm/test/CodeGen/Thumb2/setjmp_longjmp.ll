; RUN: llc %s -o - | FileCheck %s
target triple = "thumbv7-apple-ios"

declare i32 @llvm.eh.sjlj.setjmp(i8*)
declare void @llvm.eh.sjlj.longjmp(i8*)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.stacksave()
@g = external global i32

; CHECK-LABEL: double_foobar
;
; setjmp sequence:
; CHECK: mov [[PCREG:r[0-9]+]], pc
; CHECK-NEXT: adds [[PCREG]], [[PCREG]], #7
; CHECK-NEXT: str [[PCREG]], {{\[}}[[BUFREG:r[0-9]+]], #4]
; CHECK-NEXT: movs r0, #0
; CHECK-NEXT: b [[LABEL:L[a-zA-Z0-9]+]]
; CHECK-NEXT: movs r0, #1
; CHECK-NEXT: [[LABEL]]:
;
; setjmp sequence 2:
; CHECK: mov [[PCREG:r[0-9]+]], pc
; CHECK-NEXT: adds [[PCREG]], [[PCREG]], #7
; CHECK-NEXT: str [[PCREG]], {{\[}}[[BUFREG:r[0-9]+]], #4]
; CHECK-NEXT: movs r0, #0
; CHECK-NEXT: b [[LABEL:L[a-zA-Z0-9]+]]
; CHECK-NEXT: movs r0, #1
; CHECK-NEXT: [[LABEL]]:

; longjmp sequence:
; CHECK: ldr [[TEMPREG:r[0-9]+]], [{{\s*}}[[BUFREG:r[0-9]+]], #8]
; CHECK-NEXT: mov sp, [[TEMPREG]]
; CHECK-NEXT: ldr [[DESTREG:r[0-9]+]], {{\[}}[[BUFREG]], #4]
; CHECK-NEXT: ldr r7, {{\[}}[[BUFREG]]{{\]}}
; CHECK-NEXT: bx [[DESTREG]]
;
; longjmp sequence2:
; CHECK: ldr [[TEMPREG:r[0-9]+]], [{{\s*}}[[BUFREG:r[0-9]+]], #8]
; CHECK-NEXT: mov sp, [[TEMPREG]]
; CHECK-NEXT: ldr [[DESTREG:r[0-9]+]], {{\[}}[[BUFREG]], #4]
; CHECK-NEXT: ldr r7, {{\[}}[[BUFREG]]{{\]}}
; CHECK-NEXT: bx [[DESTREG]]
define void @double_foobar() {
entry:
  %buf = alloca [5 x i8*], align 4
  %bufptr = bitcast [5 x i8*]* %buf to i8*
  %arraydecay = getelementptr inbounds [5 x i8*], [5 x i8*]* %buf, i32 0, i32 0

  %fa = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %fa, i8** %arraydecay, align 4
  %ss = tail call i8* @llvm.stacksave()
  %ssgep = getelementptr [5 x i8*], [5 x i8*]* %buf, i32 0, i32 2
  store i8* %ss, i8** %ssgep, align 4

  %setjmpres = call i32 @llvm.eh.sjlj.setjmp(i8* %bufptr)
  %tobool = icmp ne i32 %setjmpres, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  store volatile i32 1, i32* @g, align 4
  br label %if.end

if.else:
  store volatile i32 0, i32* @g, align 4
  call void @llvm.eh.sjlj.longjmp(i8* %bufptr)
  unreachable

if.end:
  %fa2 = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %fa2, i8** %arraydecay, align 4
  %ss2 = tail call i8* @llvm.stacksave()
  store i8* %ss2, i8** %ssgep, align 4

  %setjmpres2 = call i32 @llvm.eh.sjlj.setjmp(i8* %bufptr)
  %tobool2 = icmp ne i32 %setjmpres2, 0
  br i1 %tobool2, label %if2.then, label %if2.else

if2.then:
  store volatile i32 3, i32* @g, align 4
  br label %if2.end

if2.else:
  store volatile i32 2, i32* @g, align 4
  call void @llvm.eh.sjlj.longjmp(i8* %bufptr)
  unreachable

if2.end:
  ret void
}
