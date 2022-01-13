; RUN: opt < %s -S -passes=globalopt | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i8, i8 }

@c = internal global i8** bitcast (i8* getelementptr (i8, i8* bitcast ([8 x i8*]* @b to i8*), i64 48) to i8**), align 8
@b = internal global [8 x i8*] [i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* getelementptr inbounds (%struct.S, %struct.S* @a, i32 0, i32 0), i8* getelementptr (i8, i8* getelementptr inbounds (%struct.S, %struct.S* @a, i32 0, i32 0), i64 1)], align 16
@a = internal global %struct.S zeroinitializer, align 1

; Function Attrs: nounwind uwtable
define signext i8 @foo() #0 {
entry:
  %0 = load i8**, i8*** @c, align 8
  %1 = load i8*, i8** %0, align 8
  %2 = load i8, i8* %1, align 1
  ret i8 %2

; CHECK-LABEL: @foo
; CHECK: ret i8 0
}

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0
}

attributes #0 = { nounwind uwtable }

