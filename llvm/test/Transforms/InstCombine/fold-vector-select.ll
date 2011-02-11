; RUN: opt < %s -instcombine -S | not grep select

define void @foo(<4 x i32> *%A, <4 x i32> *%B, <4 x i32> *%C, <4 x i32> *%D) {
 %r = select <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> zeroinitializer
 %g = select <4 x i1> <i1 false, i1 false, i1 false, i1 false>,  <4 x i32> zeroinitializer, <4 x i32> <i32 3, i32 6, i32 9, i32 1>
 %b = select <4 x i1> <i1 false, i1 true, i1 false, i1 true>,  <4 x i32> zeroinitializer, <4 x i32> <i32 7, i32 1, i32 4, i32 9>
 %a = select <4 x i1> zeroinitializer,  <4 x i32> zeroinitializer, <4 x i32> <i32 3, i32 2, i32 8, i32 5>
 store <4 x i32> %r, <4 x i32>* %A
 store <4 x i32> %g, <4 x i32>* %B
 store <4 x i32> %b, <4 x i32>* %C
 store <4 x i32> %a, <4 x i32>* %D
 ret void
}
