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

! 15.5.1 procedure reference constraints and restrictions

subroutine s01(elem, subr)
  interface
    ! Merely declaring an elemental dummy procedure is not an error;
    ! if the actual argument were an elemental unrestricted specific
    ! intrinsic function, that's okay.
    elemental real function elem(x)
      real, value :: x
    end function
    subroutine subr(elem)
      procedure(sin) :: elem
    end subroutine
  end interface
  call subr(cos) ! not an error
  !ERROR: cannot pass non-intrinsic ELEMENTAL procedure as argument
  call subr(elem)
end subroutine

module m01
  procedure(sin) :: elem01
  interface
    elemental real function elem02(x)
      real, value :: x
    end function
    subroutine callme(f)
      external f
    end subroutine
  end interface
 contains
  elemental real function elem03(x)
    real, value :: x
  end function
  subroutine test
    call callme(cos) ! not an error
    !ERROR: cannot pass non-intrinsic ELEMENTAL procedure as argument
    call callme(elem01)
    !ERROR: cannot pass non-intrinsic ELEMENTAL procedure as argument
    call callme(elem02)
    !ERROR: cannot pass non-intrinsic ELEMENTAL procedure as argument
    call callme(elem03)
    !ERROR: cannot pass non-intrinsic ELEMENTAL procedure as argument
    call callme(elem04)
   contains
    elemental real function elem04(x)
      real, value :: x
    end function
  end subroutine
end module

module m02
  interface
    subroutine altreturn(*)
    end subroutine
  end interface
 contains
  subroutine test
1   continue
   contains
    subroutine internal
      !ERROR: alternate return label must be in the inclusive scope
      call altreturn(*1)
    end subroutine
  end subroutine
end module

module m03
  type :: t
    integer, pointer :: ptr
  end type
  type(t) :: coarray[*]
 contains
  subroutine callee(x)
    type(t), intent(in) :: x
  end subroutine
  subroutine test
    !ERROR: coindexed argument cannot have a POINTER ultimate component
    call callee(coarray[1])
  end subroutine
end module
