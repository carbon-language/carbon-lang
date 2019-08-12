! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! Test character length conversions in constructors

module m
type :: t(k)
  integer, kind :: k = 1
  character(kind=k,len=1) :: a
  character(kind=k,len=3) :: b
end type t
type(t), parameter :: p = t(k=1)(a='xx',b='xx')
character(len=2), parameter :: c2(3) = [character(len=2) :: 'x', 'xx', 'xxx']
end module m

!Expect: m.mod
!module m
!type::t(k)
!integer(4),kind::k=1_4
!character(1_4,int(k,kind=8))::a
!character(3_4,int(k,kind=8))::b
!end type
!type(t(k=1_4)),parameter::p=t(k=1_4)(a=1_"x",b=1_"xx ")
!character(2_4,1),parameter::c2(1_8:3_8)=[CHARACTER(KIND=1,LEN=2)::1_"x ",1_"xx",1_"xx"]
!end
