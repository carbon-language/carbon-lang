; RUN: llc -march=hexagon -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
; CHECK: .cfi_def_cfa r30
; CHECK: .cfi_offset r31
; CHECK: .cfi_offset r30

@g0 = global i32 0, align 4
@g1 = external constant i8*

define i32 @f0() personality i8* bitcast (i32 (...)* @f3 to i8*) {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i8*
  %v2 = alloca i32
  %v3 = alloca i32, align 4
  store i32 0, i32* %v0
  %v4 = call i8* @f1(i32 4) #1
  %v5 = bitcast i8* %v4 to i32*
  store i32 20, i32* %v5
  invoke void @f2(i8* %v4, i8* bitcast (i8** @g1 to i8*), i8* null) #2
          to label %b6 unwind label %b1

b1:                                               ; preds = %b0
  %v6 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @g1 to i8*)
  %v7 = extractvalue { i8*, i32 } %v6, 0
  store i8* %v7, i8** %v1
  %v8 = extractvalue { i8*, i32 } %v6, 1
  store i32 %v8, i32* %v2
  br label %b2

b2:                                               ; preds = %b1
  %v9 = load i32, i32* %v2
  %v10 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @g1 to i8*)) #1
  %v11 = icmp eq i32 %v9, %v10
  br i1 %v11, label %b3, label %b5

b3:                                               ; preds = %b2
  %v12 = load i8*, i8** %v1
  %v13 = call i8* @f4(i8* %v12) #1
  %v14 = bitcast i8* %v13 to i32*
  %v15 = load i32, i32* %v14, align 4
  store i32 %v15, i32* %v3, align 4
  %v16 = load i32, i32* %v3, align 4
  store i32 %v16, i32* @g0, align 4
  call void @f5() #1
  br label %b4

b4:                                               ; preds = %b3
  %v17 = load i32, i32* @g0, align 4
  ret i32 %v17

b5:                                               ; preds = %b2
  %v18 = load i8*, i8** %v1
  %v19 = load i32, i32* %v2
  %v20 = insertvalue { i8*, i32 } undef, i8* %v18, 0
  %v21 = insertvalue { i8*, i32 } %v20, i32 %v19, 1
  resume { i8*, i32 } %v21

b6:                                               ; preds = %b0
  unreachable
}

declare i8* @f1(i32)

declare void @f2(i8*, i8*, i8*)

declare i32 @f3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #0

declare i8* @f4(i8*)

declare void @f5()

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { noreturn }
