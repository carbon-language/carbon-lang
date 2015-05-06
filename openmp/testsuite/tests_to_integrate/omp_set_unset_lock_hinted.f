!
! Modified version of omp_set_unset_lock.F, using hinted lock API
!
      program omp_set_unset_lock_hinted

      integer ret
#if defined(_OPENMP)
      include 'omp_lib.h'

      ret = 0
      call hinted_lock(kmp_lock_hint_none, ret)
      call hinted_lock(kmp_lock_hint_contended, ret)
      call hinted_lock(kmp_lock_hint_uncontended, ret)
      call hinted_lock(kmp_lock_hint_nonspeculative, ret)
      call hinted_lock(kmp_lock_hint_speculative, ret)
      call hinted_lock(kmp_lock_hint_adaptive, ret)
#else
      ret = 0
#endif

      if (ret .eq. 0) then
          print *, 'Test omp_set_unset_lock_hinted.f passed'
      else
          print *, 'Test omp_set_unset_lock_hinted.f failed'
      endif

      stop
      end

      subroutine hinted_lock(lock_hint, ret)
#if defined(_OPENMP)
      include 'omp_lib.h'
      integer(omp_lock_kind) lock
      integer(kmp_lock_hint_kind) lock_hint
#else
      integer lock
      integer lock_hint
#endif
      integer ret
      logical passed
      integer n(1000), j, id
      
      passed = .TRUE.

      call kmp_init_lock_hinted(lock, lock_hint)

!$omp  parallel
!$omp&     shared       (n,passed,lock)
!$omp&     private      (id,j)      
#if defined(_OPENMP) && !defined(_ASSURE)
      id = omp_get_thread_num()
#else
      id = 0
#endif
#if defined(_ASSURE)
      do j = 1, 10
#else
      do j = 1, 10000
#endif
          call cscall(id, n, passed, lock)
      enddo
!$omp  end parallel

      call omp_destroy_lock(lock)

      if (.not. passed) then
          ret = ret + 1
      endif

      end

      subroutine cscall(id, n, passed, lock)
#if defined(_OPENMP)
        include 'omp_lib.h'
      integer(omp_lock_kind) lock
#else
      integer lock
#endif
      integer id, i, n(1000)
      logical passed

      call omp_set_lock(lock)
      do i = 1,1000
          n(i) = id
      enddo
      do i = 1,1000
          if (n(i) .ne. id) passed = .FALSE.
      enddo
      call omp_unset_lock(lock)
      end
