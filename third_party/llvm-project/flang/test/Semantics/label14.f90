! Tests implemented for this standard
! 11.1.4 - 4 It is permissible to branch to an end-block-stmt only within its
!            Block Construct

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: Label '20' is in a construct that prevents its use as a branch target here

subroutine s1
  block
    goto (10) 1
10  end block

  block
20  end block
end

subroutine s2
  block
    goto (20) 1
10  end block

  block
20  end block
end

subroutine s3
  block
    block
      goto (10) 1
10  end block
20  end block
end

subroutine s4
  block
    block
      goto (20) 1
10  end block
20  end block
end
