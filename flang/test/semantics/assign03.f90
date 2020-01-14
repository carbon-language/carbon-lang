! Pointer assignment constraints 10.2.2.2 (see also assign02.f90)

module m
  interface
    subroutine s(i)
      integer i
    end
  end interface
  type :: t
    procedure(s), pointer, nopass :: p
    real, pointer :: q
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
  ! C1028
  subroutine s2
    type(t) :: a
    a%p => s
    !ERROR: In assignment to object pointer 'q', the target 's' is a procedure designator
    a%q => s
  end
  ! C1029
  subroutine s3
    type(t) :: a
    a%p => f()  ! OK: pointer-valued function
    !ERROR: Subroutine pointer 'p' may not be associated with function designator 'f'
    a%p => f
  contains
    function f()
      procedure(s), pointer :: f
      f => s
    end
  end

  ! C1030 and 10.2.2.4 - procedure names as target of procedure pointer
  subroutine s4(s_dummy)
    procedure(s), intent(in) :: s_dummy
    procedure(s), pointer :: p, q
    procedure(), pointer :: r
    integer :: i
    external :: s_external
    p => s_dummy
    p => s_internal
    p => s_module
    q => p
    r => s_external
  contains
    subroutine s_internal(i)
      integer i
    end
  end
  subroutine s_module(i)
    integer i
  end

  ! 10.2.2.4(3)
  subroutine s5
    procedure(f_pure), pointer :: p_pure
    procedure(f_impure), pointer :: p_impure
    !ERROR: Procedure pointer 'p_elemental' may not be ELEMENTAL
    procedure(f_elemental), pointer :: p_elemental
    p_pure => f_pure
    p_impure => f_impure
    p_impure => f_pure
    !ERROR: PURE procedure pointer 'p_pure' may not be associated with non-PURE procedure designator 'f_impure'
    p_pure => f_impure
  contains
    pure integer function f_pure()
      f_pure = 1
    end
    integer function f_impure()
      f_impure = 1
    end
    elemental integer function f_elemental()
      f_elemental = 1
    end
  end

  ! 10.2.2.4(4)
  subroutine s6
    procedure(s), pointer :: p, q
    procedure(), pointer :: r
    external :: s_external
    !ERROR: Procedure pointer 'p' with explicit interface may not be associated with procedure designator 's_external' with implicit interface
    p => s_external
    !ERROR: Procedure pointer 'r' with implicit interface may not be associated with procedure designator 's_module' with explicit interface
    r => s_module
  end

  ! 10.2.2.4(5)
  subroutine s7
    procedure(real) :: f_external
    external :: s_external
    procedure(), pointer :: p_s
    procedure(real), pointer :: p_f
    p_f => f_external
    p_s => s_external
    !ERROR: Subroutine pointer 'p_s' may not be associated with function designator 'f_external'
    p_s => f_external
    !ERROR: Function pointer 'p_f' may not be associated with subroutine designator 's_external'
    p_f => s_external
  end

end
