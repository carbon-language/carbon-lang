; RUN: opt < %s -instcombine -S | grep bitcast | count 1

; InstCombine can not 'load (cast P)' -> cast (load P)' if the cast changes
; the address space.


define void @test2(i8 addrspace(1)* %source, <2 x i8> addrspace(1)* %dest) {                                                                                        
entry: 
  %arrayidx1 = bitcast <2 x i8> addrspace(1)* %dest to <2 x i8> addrspace(1)*
  %conv = bitcast i8 addrspace(1)* %source to <16 x i8>*
  %arrayidx22 = bitcast <16 x i8>* %conv to <16 x i8>*
  %tmp3 = load <16 x i8>* %arrayidx22
  %arrayidx223 = bitcast i8 addrspace(1)* %source to i8*
  %tmp4 = load i8* %arrayidx223
  %tmp5 = insertelement <2 x i8> undef, i8 %tmp4, i32 0
  %splat = shufflevector <2 x i8> %tmp5, <2 x i8> undef, <2 x i32> zeroinitializer
  store <2 x i8> %splat, <2 x i8> addrspace(1)* %arrayidx1
  ret void                                                                                                                                                             
} 