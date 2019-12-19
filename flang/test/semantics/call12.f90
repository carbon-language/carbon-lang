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

! Test 15.7 C1594 - prohibited assignments in PURE subprograms

module used
  real :: useassociated
end module

module m
  type :: t
    sequence
    real :: a
  end type
  type(t), target :: x
  type :: hasPtr
    real, pointer :: p
  end type
  type :: hasCoarray
    real :: co[*]
  end type
 contains
  pure function test(ptr, in, hpd)
    use used
    type(t), pointer :: ptr, ptr2
    type(t), target, intent(in) :: in
    type(t), target :: y, z
    type(hasPtr) :: hp
    type(hasPtr), intent(in) :: hpd
    type(hasPtr), allocatable :: alloc
    type(hasCoarray), pointer :: hcp
    integer :: n
    common /block/ y
    !ERROR: PURE subprogram 'test' may not define 'x' because it is host-associated
    x%a = 0.
    !ERROR: PURE subprogram 'test' may not define 'y' because it is in a COMMON block
    y%a = 0. ! C1594(1)
    !ERROR: PURE subprogram 'test' may not define 'useassociated' because it is USE-associated
    useassociated = 0.  ! C1594(1)
    !ERROR: PURE subprogram 'test' may not define 'ptr' because it is a POINTER dummy argument of a PURE function
    ptr%a = 0. ! C1594(1)
    !ERROR: PURE subprogram 'test' may not define 'in' because it is an INTENT(IN) dummy argument
    in%a = 0. ! C1594(1)
    !ERROR: A PURE subprogram may not define a coindexed object
    hcp%co[1] = 0. ! C1594(1)
    !ERROR: PURE subprogram 'test' may not define 'ptr' because it is a POINTER dummy argument of a PURE function
    ptr => z ! C1594(2)
    !ERROR: PURE subprogram 'test' may not define 'ptr' because it is a POINTER dummy argument of a PURE function
    nullify(ptr) ! C1594(2), 19.6.8
    !ERROR: A PURE subprogram may not use 'ptr' as the target of pointer assignment because it is a POINTER dummy argument of a PURE function
    ptr2 => ptr ! C1594(3)
    !ERROR: A PURE subprogram may not use 'in' as the target of pointer assignment because it is an INTENT(IN) dummy argument
    ptr2 => in ! C1594(3)
    !ERROR: A PURE subprogram may not use 'y' as the target of pointer assignment because it is in a COMMON block
    ptr2 => y ! C1594(2)
    !ERROR: Externally visible object 'block' may not be associated with pointer component 'p' in a PURE procedure
    n = size([hasPtr(y%a)]) ! C1594(4)
    !ERROR: Externally visible object 'x' may not be associated with pointer component 'p' in a PURE procedure
    n = size([hasPtr(x%a)]) ! C1594(4)
    !ERROR: Externally visible object 'ptr' may not be associated with pointer component 'p' in a PURE procedure
    n = size([hasPtr(ptr%a)]) ! C1594(4)
    !ERROR: Externally visible object 'in' may not be associated with pointer component 'p' in a PURE procedure
    n = size([hasPtr(in%a)]) ! C1594(4)
    !ERROR: A PURE subprogram may not copy the value of 'hpd' because it is an INTENT(IN) dummy argument and has the POINTER component '%p'
    hp = hpd ! C1594(5)
    !ERROR: A PURE subprogram may not copy the value of 'hpd' because it is an INTENT(IN) dummy argument and has the POINTER component '%p'
    allocate(alloc, source=hpd)
   contains
    pure subroutine internal
      type(hasPtr) :: localhp
      !ERROR: PURE subprogram 'internal' may not define 'z' because it is host-associated
      z%a = 0.
      !ERROR: Externally visible object 'z' may not be associated with pointer component 'p' in a PURE procedure
      localhp = hasPtr(z%a)
    end subroutine
  end function
end module
