module m
  integer :: foo
  !Note: PGI, Intel, and GNU allow this; NAG and Sun do not
  !ERROR: 'foo' is already declared in this scoping unit
  interface foo
  end interface
end module

module m2
  !Note: PGI and GNU allow this; Intel, NAG, and Sun do not
  !ERROR: 's' is already declared in this scoping unit
  interface s
  end interface
contains
  subroutine s
  end subroutine
end module
