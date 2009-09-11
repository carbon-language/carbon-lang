; RUN: opt < %s -instcombine -S | not grep zext
; PR4548

define i8 @udiv_i8(i8 %a, i8 %b) nounwind {
  %conv = zext i8 %a to i32       
  %conv2 = zext i8 %b to i32      
  %div = udiv i32 %conv, %conv2   
  %conv3 = trunc i32 %div to i8   
  ret i8 %conv3
}

define i8 @urem_i8(i8 %a, i8 %b) nounwind {
  %conv = zext i8 %a to i32       
  %conv2 = zext i8 %b to i32      
  %div = urem i32 %conv, %conv2   
  %conv3 = trunc i32 %div to i8   
  ret i8 %conv3
}

