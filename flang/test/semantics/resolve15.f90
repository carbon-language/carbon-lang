module m
  real :: var
  interface i
    !ERROR: 'var' is not a subprogram
    !ERROR: Procedure 'bad' not found
    procedure :: sub, var, bad
  end interface
contains
  subroutine sub
  end
end

subroutine s
  interface i
    module procedure :: sub
  end interface
contains
  subroutine sub
  end
end
