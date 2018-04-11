program p
  !ERROR: 'p' is already declared in this scoping unit
  integer p
end
subroutine s
  !ERROR: 's' is already declared in this scoping unit
  integer :: s
end
function f() result(res)
  integer :: res
  !ERROR: 'f' is already declared in this scoping unit
  real :: f
  res = 1
end
