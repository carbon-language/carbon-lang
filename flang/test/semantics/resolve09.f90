integer :: y
call x
!ERROR: Use of 'y' as a procedure conflicts with its declaration
call y
end
