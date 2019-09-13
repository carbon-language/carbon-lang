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

module m
  type :: t
    sequence
    real :: a
  end type
  type(t), target :: x
  type :: hasPtr
    real, pointer :: p
  end type
 contains
  pure subroutine msubr
    !ERROR: A PURE subprogram must not define a host-associated object
    x%a = 0.
  end subroutine
end module

pure function test(ptr, in)
  use m
  type(t), pointer :: ptr, ptr2
  type(t), target, intent(in) :: in
  type(t), save :: co[*]
  type(t), target :: y, z
  type(hasPtr) :: hp
  type(hasPtr), allocatable :: alloc
  integer :: n
  common /block/ y
  !ERROR: A PURE subprogram must not define an object in COMMON
  y%a = 0. ! C1594(1)
  !ERROR: A PURE subprogram must not define a USE-associated object
  x%a = 0.  ! C1594(1)
  !ERROR: A PURE function must not define a pointer dummy argument
  ptr%a = 0. ! C1594(1)
  !ERROR: A PURE subprogram must not define an INTENT(IN) dummy argument
  in%a = 0. ! C1594(1)
  !ERROR: A PURE subprogram must not define a coindexed object
  co[1]%a = 0. ! C1594(1)
  !ERROR: A PURE function must not define a pointer dummy argument
  ptr => y ! C1594(2)
  !ERROR: A PURE subprogram must not define a pointer dummy argument
  nullify(ptr) ! C1594(2), 19.6.8
  !ERROR: A PURE subprogram must not use a pointer dummy argument as the target of pointer assignment
  ptr2 => ptr ! C1594(3)
  !ERROR: A PURE subprogram must not use an INTENT(IN) dummy argument as the target of pointer assignment
  ptr2 => in ! C1594(3)
  !ERROR: Externally visible object 'block' must not be associated with pointer component 'p' in a PURE procedure
  n = size([hasPtr(y%a)]) ! C1594(4)
  !ERROR: Externally visible object 'x' must not be associated with pointer component 'p' in a PURE procedure
  n = size([hasPtr(x%a)]) ! C1594(4)
  !ERROR: Externally visible object 'ptr' must not be associated with pointer component 'p' in a PURE procedure
  n = size([hasPtr(ptr%a)]) ! C1594(4)
  !ERROR: A PURE subprogram must not use an INTENT(IN) dummy argument as the target of pointer assignment
  n = size([hasPtr(in%a)]) ! C1594(4)
  !ERROR: A PURE subprogram must not assign to a variable with a POINTER component
  hp = hp ! C1594(5)
  !ERROR: A PURE subprogram must not use a derived type with a POINTER component as a SOURCE=
  allocate(alloc, source=hp)
 contains
  pure subroutine internal
    !ERROR: A PURE subprogram must not define a host-associated object
    z%a = 0.
    !ERROR: Externally visible object 'z' must not be associated with pointer component 'p' in a PURE procedure
    hp = hasPtr(z%a)
  end subroutine
end function
