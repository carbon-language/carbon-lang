; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep load
; PR4748
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.B = type { double }
	%struct.A = type { %struct.B, i32, i32 }

define i32 @_Z4funcv(%struct.A* %a) {
entry:
  %g3 = getelementptr %struct.A* %a, i32 0, i32 1
  store i32 10, i32* %g3, align 4

  %g4 = getelementptr %struct.A* %a, i32 0, i32 0
  
  %new_a = bitcast %struct.B* %g4 to %struct.A*

  %g5 = getelementptr %struct.A* %new_a, i32 0, i32 1	
  %a_a = load i32* %g5, align 4	
  ret i32 %a_a
}

