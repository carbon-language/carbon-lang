module m1
  integer :: x
  integer, private :: y
end

use m1, local_x => x
!ERROR: 'y' is PRIVATE in 'm1'
use m1, local_y => y
!ERROR: 'z' not found in module 'm1'
use m1, local_z => z

!ERROR: 'y' is PRIVATE in 'm1'
use m1, only: y
!ERROR: 'z' not found in module 'm1'
use m1, only: z
!ERROR: 'z' not found in module 'm1'
use m1, only: my_x => z

end
