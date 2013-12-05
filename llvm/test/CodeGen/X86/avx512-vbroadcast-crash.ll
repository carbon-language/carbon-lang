; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl 

define <16 x i32> @test_vbroadcast() {
entry:
  %0 = sext <16 x i1> zeroinitializer to <16 x i32>
  %1 = fcmp uno <16 x float> undef, zeroinitializer
  %2 = sext <16 x i1> %1 to <16 x i32>
  %3 = select <16 x i1> %1, <16 x i32> %0, <16 x i32> %2
  ret <16 x i32> %3
}
