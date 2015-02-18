; Check the miscellaneous logical vector operations added in P8
; 
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s
; Test x eqv y
define <4 x i32> @test_veqv(<4 x i32> %x, <4 x i32> %y) nounwind {
       %tmp = xor <4 x i32> %x, %y
       %ret_val = xor <4 x i32> %tmp, < i32 -1, i32 -1, i32 -1, i32 -1>
       ret <4 x i32> %ret_val
; CHECK: veqv 2, 2, 3
}

; Test x vnand y
define <4 x i32> @test_vnand(<4 x i32> %x, <4 x i32> %y) nounwind {
       %tmp = and <4 x i32> %x, %y
       %ret_val = xor <4 x i32> %tmp, <i32 -1, i32 -1, i32 -1, i32 -1>
       ret <4 x i32> %ret_val
; CHECK: vnand 2, 2, 3
}

; Test x vorc y
define <4 x i32> @test_vorc(<4 x i32> %x, <4 x i32> %y) nounwind {
       %tmp = xor <4 x i32> %y, <i32 -1, i32 -1, i32 -1, i32 -1>
       %ret_val = or <4 x i32> %x, %tmp
       ret <4 x i32> %ret_val
; CHECK: vorc 2, 2, 3      
}
