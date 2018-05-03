module m1
end

subroutine sub
end

use m1
!ERROR: Module 'm2' not found
use m2
!ERROR: 'sub' is not a module
use sub
end
