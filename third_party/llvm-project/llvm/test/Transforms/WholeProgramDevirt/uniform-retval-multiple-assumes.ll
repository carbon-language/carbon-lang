; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x i8*] [i8* bitcast (i32 (i8*)* @vf1 to i8*)], !type !0
@vt2 = constant [1 x i8*] [i8* bitcast (i32 (i8*)* @vf2 to i8*)], !type !0

define i32 @vf1(i8* %this) readnone {
  ret i32 123
}

define i32 @vf2(i8* %this) readnone {
  ret i32 123
}

; CHECK: define i32 @call
define i32 @call(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %p2 = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p2)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*)*
  %result = call i32 %fptr_casted(i8* %obj)
  ; CHECK-NOT: call i32 %
  ; CHECK: ret i32 123
  ret i32 %result
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
