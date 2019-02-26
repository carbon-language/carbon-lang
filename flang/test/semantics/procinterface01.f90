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

! Tests for "proc-interface" semantics.
! These cases are all valid.

!DEF: /module1 Module
module module1
 abstract interface
  !DEF: /module1/abstract1/abstract1 ObjectEntity REAL(4)
  !DEF: /module1/abstract1/x INTENT(IN) ObjectEntity REAL(4)
  real elemental function abstract1(x)
   !REF: /module1/abstract1/x
   real, intent(in) :: x
  end function abstract1
 end interface

 interface
  !DEF: /module1/explicit1/explicit1 ObjectEntity REAL(4)
  !DEF: /module1/explicit1/x INTENT(IN) ObjectEntity REAL(4)
  real elemental function explicit1(x)
   !REF: /module1/explicit1/x
   real, intent(in) :: x
  end function explicit1
  !DEF: /module1/logical/logical ObjectEntity INTEGER(4)
  !DEF: /module1/logical/x INTENT(IN) ObjectEntity REAL(4)
  integer function logical(x)
   !REF: /module1/logical/x
   real, intent(in) :: x
  end function logical
  !DEF: /module1/tan/tan ObjectEntity CHARACTER(1_4,1)
  !DEF: /module1/tan/x INTENT(IN) ObjectEntity REAL(4)
  character(len=1) function tan(x)
   !REF: /module1/tan/x
   real, intent(in) :: x
  end function tan
 end interface

 !DEF: /module1/derived1 PUBLIC DerivedType
 type :: derived1
  !DEF: /module1/abstract1 ELEMENTAL, PUBLIC Subprogram
  !DEF: /module1/derived1/p1 NOPASS, POINTER ProcEntity
  !DEF: /module1/nested1 ELEMENTAL, PUBLIC Subprogram
  procedure(abstract1), pointer, nopass :: p1 => nested1
  !DEF: /module1/explicit1 ELEMENTAL, EXTERNAL, PUBLIC Subprogram
  !DEF: /module1/derived1/p2 NOPASS, POINTER ProcEntity
  !REF: /module1/nested1
  procedure(explicit1), pointer, nopass :: p2 => nested1
  !DEF: /module1/logical EXTERNAL, PUBLIC Subprogram
  !DEF: /module1/derived1/p3 NOPASS, POINTER ProcEntity
  !DEF: /module1/nested2 PUBLIC Subprogram
  procedure(logical), pointer, nopass :: p3 => nested2
  !DEF: /module1/derived1/p4 NOPASS, POINTER ProcEntity LOGICAL(4)
  !DEF: /module1/nested3 PUBLIC Subprogram
  procedure(logical(kind=4)), pointer, nopass :: p4 => nested3
  !DEF: /module1/derived1/p5 NOPASS, POINTER ProcEntity COMPLEX(4)
  !DEF: /module1/nested4 PUBLIC Subprogram
  procedure(complex), pointer, nopass :: p5 => nested4
  !DEF: /module1/derived1/p6 NOPASS, POINTER ProcEntity
  !REF: /module1/nested1
  ! NOTE: sin is not dumped as a DEF here because specific
  ! intrinsic functions are represented with MiscDetails
  ! and those are omitted from dumping.
  procedure(sin), pointer, nopass :: p6 => nested1
  !DEF: /module1/derived1/p7 NOPASS, POINTER ProcEntity
  procedure(sin), pointer, nopass :: p7 => cos
  !DEF: /module1/tan EXTERNAL, PUBLIC Subprogram
  !DEF: /module1/derived1/p8 NOPASS, POINTER ProcEntity
  !DEF: /module1/nested5 PUBLIC Subprogram
  procedure(tan), pointer, nopass :: p8 => nested5
 end type derived1

contains

 !DEF: /module1/nested1/nested1 ObjectEntity REAL(4)
 !DEF: /module1/nested1/x INTENT(IN) ObjectEntity REAL(4)
 real elemental function nested1(x)
  !REF: /module1/nested1/x
  real, intent(in) :: x
  !REF: /module1/nested1/nested1
  !REF: /module1/nested1/x
  nested1 = x+1.
 end function nested1

 !DEF: /module1/nested2/nested2 ObjectEntity INTEGER(4)
 !DEF: /module1/nested2/x INTENT(IN) ObjectEntity REAL(4)
 integer function nested2(x)
  !REF: /module1/nested2/x
  real, intent(in) :: x
  !REF: /module1/nested2/nested2
  !REF: /module1/nested2/x
  nested2 = x+2.
 end function nested2

 !DEF: /module1/nested3/nested3 ObjectEntity LOGICAL(4)
 !DEF: /module1/nested3/x INTENT(IN) ObjectEntity REAL(4)
 logical function nested3(x)
  !REF: /module1/nested3/x
  real, intent(in) :: x
  !REF: /module1/nested3/nested3
  !REF: /module1/nested3/x
  nested3 = x>0
 end function nested3

 !DEF: /module1/nested4/nested4 ObjectEntity COMPLEX(4)
 !DEF: /module1/nested4/x INTENT(IN) ObjectEntity REAL(4)
 complex function nested4(x)
  !REF: /module1/nested4/x
  real, intent(in) :: x
  !REF: /module1/nested4/nested4
  !DEF: /cmplx EXTERNAL (implicit) ProcEntity REAL(4)
  !REF: /module1/nested4/x
  nested4 = cmplx(x+4., 6.)
 end function nested4

 !DEF: /module1/nested5/nested5 ObjectEntity CHARACTER(1_8,1)
 !DEF: /module1/nested5/x INTENT(IN) ObjectEntity REAL(4)
 character function nested5(x)
  !REF: /module1/nested5/x
  real, intent(in) :: x
  !REF: /module1/nested5/nested5
  nested5 = "a"
 end function nested5
end module module1

!DEF: /explicit1/explicit1 ObjectEntity REAL(4)
!DEF: /explicit1/x INTENT(IN) ObjectEntity REAL(4)
real elemental function explicit1(x)
 !REF: /explicit1/x
 real, intent(in) :: x
 !REF: /explicit1/explicit1
 !REF: /explicit1/x
 explicit1 = -x
end function explicit1

!DEF: /logical/logical ObjectEntity INTEGER(4)
!DEF: /logical/x INTENT(IN) ObjectEntity REAL(4)
integer function logical(x)
 !REF: /logical/x
 real, intent(in) :: x
 !REF: /logical/logical
 !REF: /logical/x
 logical = x+3.
end function logical

!DEF: /tan/tan ObjectEntity REAL(4)
!DEF: /tan/x INTENT(IN) ObjectEntity REAL(4)
real function tan(x)
 !REF: /tan/x
 real, intent(in) :: x
 !REF: /tan/tan
 !REF: /tan/x
 tan = x+5.
end function tan

!DEF: /main MainProgram
program main
 !REF: /module1
 use :: module1
 !DEF: /main/derived1 Use
 !DEF: /main/instance ObjectEntity TYPE(derived1)
 type(derived1) :: instance
 !REF: /main/instance
 !REF: /module1/derived1/p1
 if (instance%p1(1.)/=2.) print *, "p1 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p2
 if (instance%p2(1.)/=2.) print *, "p2 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p3
 if (instance%p3(1.)/=3) print *, "p3 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p4
 if (.not.instance%p4(1.)) print *, "p4 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p5
 if (instance%p5(1.)/=(5.,6.)) print *, "p5 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p6
 if (instance%p6(1.)/=2.) print *, "p6 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p7
 if (instance%p7(0.)/=1.) print *, "p7 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p8
 if (instance%p8(1.)/="a") print *, "p8 failed"
end program main
