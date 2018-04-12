subroutine s1
  integer :: x(2)
  !ERROR: The dimensions of 'x' have already been declared
  allocatable :: x(:)
end

subroutine s2
  target :: x(1)
  !ERROR: The dimensions of 'x' have already been declared
  integer :: x(2)
end

subroutine s3
  dimension :: x(4), y(8)
  !ERROR: The dimensions of 'x' have already been declared
  allocatable :: x(:)
end

subroutine s4
  integer, dimension(10) :: x(2,2), y
end
