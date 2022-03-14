; RUN: opt  -passes=instcombine -S %s | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define void @foo(i32) {
; CHECK-LABEL: @foo(
; CHECK: alloca
; CHECK: align 16
	%2 = alloca [3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>], align 16		; <[3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>]*> [#uses=1]
	%3 = getelementptr [3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>], [3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>]* %2, i32 0, i32 0		; <<{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>*> [#uses=1]
	%4 = getelementptr <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>, <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>* %3, i32 0, i32 0		; <{ { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } }*> [#uses=1]
	%5 = getelementptr { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } }, { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } }* %4, i32 0, i32 0		; <{ [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 }*> [#uses=1]
	%6 = bitcast { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 }* %5 to { [8 x i16] }*		; <{ [8 x i16] }*> [#uses=1]
	%7 = getelementptr { [8 x i16] }, { [8 x i16] }* %6, i32 0, i32 0		; <[8 x i16]*> [#uses=1]
	%8 = getelementptr [8 x i16], [8 x i16]* %7, i32 0, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %8, align 16
    call void @bar(i16* %8)
	ret void
}

declare void @bar(i16*)

define void @foo_as1(i32 %a, [3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>] addrspace(1)* %b) {
; CHECK-LABEL: @foo_as1(
; CHECK: align 16
  %1 = getelementptr [3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>], [3 x <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>] addrspace(1)* %b, i32 0, i32 0        ; <<{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>*> [#uses=1]
  %2 = getelementptr <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }>, <{ { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } }> addrspace(1)* %1, i32 0, i32 0      ; <{ { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } }*> [#uses=1]
  %3 = getelementptr { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } }, { { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } } addrspace(1)* %2, i32 0, i32 0        ; <{ [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 }*> [#uses=1]
  %4 = bitcast { [2 x { { i32 } }], [2 x i8], { i16 }, [2 x i8], i8, i8 } addrspace(1)* %3 to { [8 x i16] } addrspace(1)*     ; <{ [8 x i16] }*> [#uses=1]
  %5 = getelementptr { [8 x i16] }, { [8 x i16] } addrspace(1)* %4, i32 0, i32 0     ; <[8 x i16]*> [#uses=1]
  %6 = getelementptr [8 x i16], [8 x i16] addrspace(1)* %5, i32 0, i32 0     ; <i16*> [#uses=1]
  store i16 0, i16 addrspace(1)* %6, align 16
  call void @bar_as1(i16 addrspace(1)* %6)
  ret void
}

declare void @bar_as1(i16 addrspace(1)*)
