! Utility functions to have a sleep function with better resolution and
! which only stops one thread.

      subroutine my_sleep(sleeptime)
        implicit none
        double precision :: sleeptime
        integer :: u
        integer :: t(8)
        integer :: ms1, ms2
        integer :: cnt

        u = sleeptime * 1000

        call date_and_time(values=t)

        ! calculate start time in ms
        ms1 = t(8) + t(7)*1000 + t(6)*60000 + t(5)*3600000

        ms2 = ms1
        cnt = 0
        do while ( (ms2 - ms1) < u)
            call date_and_time(values=t)
            ms2 = t(8) + t(7)*1000 + t(6)*60000 + t(5)*3600000
            cnt = cnt+1
        end do
      end subroutine
