! Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

module m
  type :: pdt(k1,k2,k4,k8,k16)
    integer(1), kind :: k1
    integer(2), kind :: k2
    integer(4), kind :: k4
    integer(8), kind :: k8
    integer(16), kind :: k16
  end type pdt
 contains
  subroutine s1
    type(pdt(k16=116,k8=18,k4=14,k2=12,k1=11)) :: x1
    print *, x1%k1
    print *, x1%k2
    print *, x1%k4
    print *, x1%k8
    print *, x1%k16
  end subroutine s1
  subroutine s2
    integer(1) :: j1, ja1(1)
    integer(2) :: j2, ja2(1)
    integer(4) :: j4, ja4(1)
    integer(8) :: j8, ja8(1)
    integer(16) :: j16, ja16(1)
    real(2) :: a2, aa2(1)
    real(4) :: a4, aa4(1)
    real(8) :: a8, aa8(1)
    real(10) :: a10, aa10(1)
    real(16) :: a16, aa16(1)
    complex(2) :: z2, za2(1)
    complex(4) :: z4, za4(1)
    complex(8) :: z8, za8(1)
    complex(10) :: z10, za10(1)
    complex(16) :: z16, za16(1)
    character(1,11) :: ch1, cha1(1)
    character(2,12) :: ch2, cha2(1)
    character(4,14) :: ch4, cha4(1)
    logical(1) :: p1, pa1(1)
    logical(2) :: p2, pa2(1)
    logical(4) :: p4, pa4(1)
    logical(8) :: p8, pa8(1)
    print *, j1%kind, ja1%kind, ja1(1)%kind
    print *, j2%kind, ja2%kind, ja2(1)%kind
    print *, j4%kind, ja4%kind, ja4(1)%kind
    print *, j8%kind, ja8%kind, ja8(1)%kind
    print *, j16%kind, ja16%kind, ja16(1)%kind
    print *, a2%kind, aa2%kind, aa2(1)%kind
    print *, a4%kind, aa4%kind, aa4(1)%kind
    print *, a8%kind, aa8%kind, aa8(1)%kind
    print *, a10%kind, aa10%kind, aa10(1)%kind
    print *, a16%kind, aa16%kind, aa16(1)%kind
    print *, z2%kind, za2%kind, za2(1)%kind
    print *, z4%kind, za4%kind, za4(1)%kind
    print *, z8%kind, za8%kind, za8(1)%kind
    print *, z10%kind, za10%kind, za10(1)%kind
    print *, z16%kind, za16%kind, za16(1)%kind
    print *, ch1%kind, cha1%kind, cha1(1)%kind
    print *, ch2%kind, cha2%kind, cha2(1)%kind
    print *, ch4%kind, cha4%kind, cha4(1)%kind
    print *, ch1%len, cha1%len, cha1(1)%len
    print *, ch2%len, cha2%len, cha2(1)%len
    print *, ch4%len, cha4%len, cha4(1)%len
    print *, p1%kind, pa1%kind, pa1(1)%kind
    print *, p2%kind, pa2%kind, pa2(1)%kind
    print *, p4%kind, pa4%kind, pa4(1)%kind
    print *, p8%kind, pa8%kind, pa8(1)%kind
  end subroutine s2
end module
