module m
  interface
    subroutine s(i)
      integer i
    end
  end interface
  type :: t
    procedure(s), pointer, nopass :: p
  end type
contains
  ! C1027
  subroutine s1
    type(t), allocatable :: a(:)
    type(t), allocatable :: b[:]
    a(1)%p => s
    !ERROR: Procedure pointer may not be a coindexed object
    b[1]%p => s
  end
end
