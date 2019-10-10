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

! Test 15.5.2.4 constraints and restrictions for non-POINTER non-ALLOCATABLE
! dummy arguments.

module m01
  type :: t
  end type
  type :: pdt(n)
    integer, len :: n
  end type
  type :: tbp
   contains
    procedure :: binding => subr01
  end type
  type :: final
   contains
    final :: subr02
  end type
  type :: alloc
    real, allocatable :: a(:)
  end type
  type :: ultimateCoarray
    real, allocatable :: a[*]
  end type

 contains

  subroutine subr01(this)
    class(tbp), intent(in) :: this
  end subroutine
  subroutine subr02(this)
    class(final), intent(in) :: this
  end subroutine

  subroutine poly(x)
    class(t), intent(in) :: x
  end subroutine
  subroutine polyassumedsize(x)
    class(t), intent(in) :: x(*)
  end subroutine
  subroutine assumedsize(x)
    real :: x(*)
  end subroutine
  subroutine assumedrank(x)
    real :: x(..)
  end subroutine
  subroutine assumedtypeandsize(x)
    type(*) :: x(*)
  end subroutine
  subroutine assumedshape(x)
    real :: x(:)
  end subroutine
  subroutine contiguous(x)
    real, contiguous :: x(:)
  end subroutine
  subroutine intentout(x)
    real, intent(out) :: x
  end subroutine
  subroutine intentinout(x)
    real, intent(in out) :: x
  end subroutine
  subroutine asynchronous(x)
    real, asynchronous :: x
  end subroutine
  subroutine asynchronousValue(x)
    real, asynchronous, value :: x
  end subroutine
  subroutine volatile(x)
    real, volatile :: x
  end subroutine
  subroutine pointer(x)
    real, pointer :: x(:)
  end subroutine
  subroutine valueassumedsize(x)
    real, value :: x(*)
  end subroutine

  subroutine test01(x) ! 15.5.2.4(2)
    class(t), intent(in) :: x[*]
    !ERROR: Coindexed polymorphic object may not be associated with a polymorphic dummy argument
    call poly(x[1])
  end subroutine

  subroutine mono(x)
    type(t), intent(in) :: x
  end subroutine
  subroutine test02(x) ! 15.5.2.4(2)
    class(t), intent(in) :: x(*)
    !ERROR: Assumed-size polymorphic array may not be associated with a monomorphic dummy argument
    call mono(x)
  end subroutine

  subroutine typestar(x)
    type(*), intent(in) :: x
  end subroutine
  subroutine test03 ! 15.5.2.4(2)
    type(pdt(0)) :: x
    !ERROR: Actual argument associated with TYPE(*) dummy argument may not have a parameterized derived type
    call typestar(x)
  end subroutine

  subroutine test04 ! 15.5.2.4(2)
    type(tbp) :: x
    !ERROR: Actual argument associated with TYPE(*) dummy argument may not have type-bound procedures
    call typestar(x)
  end subroutine

  subroutine test05 ! 15.5.2.4(2)
    type(final) :: x
    !ERROR: Actual argument associated with TYPE(*) dummy argument may not have FINAL procedures
    call typestar(x)
  end subroutine

  subroutine ch2(x)
    character(2), intent(in) :: x
  end subroutine
  subroutine test06 ! 15.5.2.4(4)
    character :: ch1
    !ERROR: Actual length '1' is less than expected length '2'
    call ch2(ch1)
    !ERROR: Actual length '1' is less than expected length '2'
    call ch2(' ')
  end subroutine

  subroutine out01(x)
    type(alloc) :: x
  end subroutine
  subroutine test07(x) ! 15.5.2.4(6)
    type(alloc) :: x[*]
    !ERROR: Coindexed actual argument with ALLOCATABLE ultimate component must be associated with a dummy argument with VALUE or INTENT(IN) attributes
    call out01(x[1])
  end subroutine

  subroutine test08(x) ! 15.5.2.4(13)
    real :: x(1)[*]
    !ERROR: Coindexed scalar actual argument must be associated with a scalar dummy argument
    call assumedsize(x(1)[1])
  end subroutine

  subroutine charray(x)
    character :: x(10)
  end subroutine
  subroutine test09(ashape, polyarray, c) ! 15.5.2.4(14), 15.5.2.11
    real :: x, arr(10)
    real, pointer :: p(:)
    real :: ashape(:)
    class(t) :: polyarray(*)
    character(10) :: c(:)
    !ERROR: Whole scalar actual argument may not be associated with a dummy argument array
    call assumedsize(x)
    !ERROR: Element of pointer array may not be associated with a dummy argument array
    call assumedsize(p(1))
    !ERROR: Element of assumed-shape array may not be associated with a dummy argument array
    call assumedsize(ashape(1))
    !ERROR: Element of polymorphic array may not be associated with a dummy argument array
    call polyassumedsize(polyarray(1))
    call charray(c(1:1))  ! not an error if character
    call assumedsize(arr(1))  ! not an error if element in sequence
    call assumedrank(x)  ! not an error
    call assumedtypeandsize(x)  ! not an error
  end subroutine

  subroutine test10(a) ! 15.5.2.4(16)
    real :: scalar, matrix(2,3)
    real :: a(*)
    !ERROR: Rank of actual argument (0) differs from assumed-shape dummy argument (1)
    call assumedshape(scalar)
    !ERROR: Rank of actual argument (2) differs from assumed-shape dummy argument (1)
    call assumedshape(matrix)
    !ERROR: Assumed-size array cannot be associated with assumed-shape dummy argument
    call assumedshape(a)
  end subroutine

  subroutine test11(in) ! C15.5.2.4(20)
    real, intent(in) :: in
    real :: x
    x = 0.
    !ERROR: Actual argument associated with INTENT(OUT) dummy must be definable
    call intentout(in)
    !ERROR: Actual argument associated with INTENT(OUT) dummy must be definable
    call intentout(3.14159)
    !ERROR: Actual argument associated with INTENT(OUT) dummy must be definable
    call intentout(in + 1.)
    call intentout(x) ! ok
    !ERROR: Actual argument associated with INTENT(OUT) dummy must be definable
    call intentout((x))
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy must be definable
    call intentinout(in)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy must be definable
    call intentinout(3.14159)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy must be definable
    call intentinout(in + 1.)
    call intentinout(x) ! ok
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy must be definable
    call intentinout((x))
  end subroutine

  subroutine test12 ! 15.5.2.4(21)
    real :: a(1)
    integer :: j(1)
    j(1) = 1
    !ERROR: Actual argument associated with INTENT(OUT) dummy must be definable
    call intentout(a(j))
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy must be definable
    call intentinout(a(j))
    !ERROR: Actual argument associated with ASYNCHRONOUS dummy must be definable
    call asynchronous(a(j))
    !ERROR: Actual argument associated with VOLATILE dummy must be definable
    call volatile(a(j))
  end subroutine

  subroutine coarr(x)
    type(ultimateCoarray):: x
  end subroutine
  subroutine volcoarr(x)
    type(ultimateCoarray), volatile :: x
  end subroutine
  subroutine test13(a, b) ! 15.5.2.4(22)
    type(ultimateCoarray) :: a
    type(ultimateCoarray), volatile :: b
    call coarr(a)  ! ok
    call volcoarr(b)  ! ok
    !ERROR: VOLATILE attributes must match when argument has a coarray ultimate component
    call coarr(b)
    !ERROR: VOLATILE attributes must match when argument has a coarray ultimate component
    call volcoarr(a)
  end subroutine

  subroutine test14(a,b,c,d) ! C1538
    real :: a[*]
    real, asynchronous :: b[*]
    real, volatile :: c[*]
    real, asynchronous, volatile :: d[*]
    call asynchronous(a[1])  ! ok
    call volatile(a[1])  ! ok
    call asynchronousValue(b[1])  ! ok
    call asynchronousValue(c[1])  ! ok
    call asynchronousValue(d[1])  ! ok
    !ERROR: coindexed ASYNCHRONOUS or VOLATILE effective argument must not be associated with dummy argument with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call asynchronous(b[1])
    call volatile(b[1])
    !ERROR: coindexed ASYNCHRONOUS or VOLATILE effective argument must not be associated with dummy argument with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call asynchronous(c[1])
    call volatile(c[1])
    !ERROR: coindexed ASYNCHRONOUS or VOLATILE effective argument must not be associated with dummy argument with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call asynchronous(d[1])
    call volatile(d[1])
  end subroutine

  subroutine test15() ! C1539
    real, pointer :: a(:)
    real, asynchronous :: b(10)
    real, volatile :: c(10)
    real, asynchronous, volatile :: d(10)
    call assumedsize(a(::2)) ! ok
    call contiguous(a(::2)) ! ok
    call valueassumedsize(a(::2)) ! ok
    call valueassumedsize(b(::2)) ! ok
    call valueassumedsize(c(::2)) ! ok
    call valueassumedsize(d(::2)) ! ok
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call assumedsize(b(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call contiguous(b(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call assumedsize(c(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call contiguous(c(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call assumedsize(d(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call contiguous(d(::2))
  end subroutine

  subroutine test16() ! C1540
    real, pointer :: a(:)
    real, asynchronous, pointer :: b(:)
    real, volatile, pointer :: c(:)
    real, asynchronous, volatile, pointer :: d(:)
    call assumedsize(a) ! ok
    call contiguous(a) ! ok
    call pointer(a) ! ok
    call pointer(b) ! ok
    call pointer(c) ! ok
    call pointer(d) ! ok
    call valueassumedsize(a) ! ok
    call valueassumedsize(b) ! ok
    call valueassumedsize(c) ! ok
    call valueassumedsize(d) ! ok
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call assumedsize(b)
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call contiguous(b)
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call assumedsize(c)
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call contiguous(c)
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call assumedsize(d)
    !ERROR: ASYNCHRONOUS or VOLATILE effective argument that is not simply contiguous cannot be associated with a contiguous dummy argument
    call contiguous(d)
  end subroutine

end module
