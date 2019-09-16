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

! Test 15.4.2.2 constraints and restrictions for calls to implicit
! interfaces

subroutine s(assumedRank, coarray, class, classStar, typeStar)
  type :: t
  end type

  real :: assumedRank(..), coarray[*]
  class(t) :: class
  class(*) :: classStar
  type(*) :: typeStar

  type :: pdt(len)
    integer, len :: len
  end type
  type(pdt(1)) :: pdtx

  !ERROR: Invalid specification expression: reference to impure function 'implicit01'
  real :: array(implicit01())  ! 15.4.2.2(2)
  !ERROR: Keyword 'keyword=' cannot appear in a reference to a procedure with an implicit interface
  call implicit10(1, 2, keyword=3)  ! 15.4.2.2(1)
  !ERROR: Assumed rank argument requires an explicit interface
  call implicit11(assumedRank)  ! 15.4.2.2(3)(c)
  !ERROR: Coarray argument requires an explicit interface
  call implicit12(coarray)  ! 15.4.2.2(3)(d)
  !ERROR: Parameterized derived type argument requires an explicit interface
  call implicit13(pdtx)  ! 15.4.2.2(3)(e)
  !ERROR: Polymorphic argument requires an explicit interface
  call implicit14(class)  ! 15.4.2.2(3)(f)
  !ERROR: Polymorphic argument requires an explicit interface
  call implicit15(classStar)  ! 15.4.2.2(3)(f)
  !ERROR: Assumed type argument requires an explicit interface
  call implicit16(typeStar)  ! 15.4.2.2(3)(f)
end subroutine

