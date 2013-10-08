; XFAIL: *
; RUN: llc < %s -march=r600 -mcpu=redwood -o %t

define void @var_insert(<4 x i32> addrspace(1)* %out, <4 x i32> %x, i32 %val, i32 %idx) nounwind  {
entry:
  %tmp3 = insertelement <4 x i32> %x, i32 %val, i32 %idx		; <<4 x i32>> [#uses=1]
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out
  ret void
}

define void @var_extract(i32 addrspace(1)* %out, <4 x i32> %x, i32 %idx) nounwind  {
entry:
  %tmp3 = extractelement <4 x i32> %x, i32 %idx		; <<i32>> [#uses=1]
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}
