subroutine s
  real :: x
contains
  !ERROR: 'x' is already declared in this scoping unit
  subroutine x
  end
end
