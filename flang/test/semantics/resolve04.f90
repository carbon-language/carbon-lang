!ERROR: No explicit type declared for 'f'
function f()
  implicit none
end

!ERROR: No explicit type declared for 'y'
subroutine s(x, y)
  implicit none
  integer :: x
end
