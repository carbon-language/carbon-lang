      subroutine print_result(s,crossfailed,M,name)     
      implicit none
      character (len=*)::name
	   real cert
	   integer M,crossfailed
	   integer s
	   character (len=11) :: c
 	   character (len=18) :: c2
	   cert=100.0*crossfailed/M
!           print *, "cert", cert, "cross ", crossfailed
!	test1=hundred*crossfailed
	   c="% certainty"
	   c2=" ... verified with "
      if(s.eq.1) then
         write (*,"(A, A, F7.2, A)") name, c2, cert, c
      else
         write (*,"(A,A)") name," ... FAILED"
      endif
	   end

      subroutine do_test(test_func,cross_test_func,name,N,failed,
     x  num_tests,crosschecked)
      implicit none
      integer succeed
      integer crossfail
      integer failed
      integer, external::test_func
      integer, external::cross_test_func
      character (len=*)::name
      integer fail
      integer N,i
      integer num_tests,crosschecked
      num_tests=num_tests+1
      succeed=1
      crossfail=0
      fail=0
      do i=1,N
         if(test_func().eq.0) then
            succeed=0
            fail=fail+1
            exit
         end if
         if(cross_test_func().eq.0) then
!            print *, crossfail
            crossfail=crossfail+1.0 
         end if
      enddo
      
      if (fail .ne. 0) then
         failed=failed+1
      else
         if(crossfail .ne. 0) then
            crosschecked=crosschecked+1
         end if
      endif
      call print_result(succeed,crossfail,N,name)
      end
