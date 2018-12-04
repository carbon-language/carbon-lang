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

type ipdt(k)
 integer, kind :: k
 integer(kind=k) :: x
end type ipdt

type rpdt(k)
 integer, kind :: k
 real(kind=k) :: x
end type rpdt

type zpdt(k)
 integer, kind :: k
 complex(kind=k) :: x
end type zpdt

type lpdt(k)
 integer, kind :: k
 logical(kind=k) :: x
end type lpdt

type(ipdt(1)) i1
type(ipdt(2)) i2
type(ipdt(4)) i4
type(ipdt(8)) i8
type(ipdt(16)) i16
type(rpdt(2)) a2
type(rpdt(4)) a4
type(rpdt(8)) a8
type(rpdt(10)) a10
type(rpdt(16)) a16
type(zpdt(2)) z2
type(zpdt(4)) z4
type(zpdt(8)) z8
type(zpdt(10)) z10
type(zpdt(16)) z16
type(lpdt(1)) l1
type(lpdt(2)) l2
type(lpdt(4)) l4
type(lpdt(8)) l8

end program
