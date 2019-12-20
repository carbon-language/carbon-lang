integer :: a1(10), a2(10)
logical :: m1(10), m2(5,5)
m1 = .true.
m2 = .false.
a1 = [((i),i=1,10)]
where (m1)
  a2 = 1
!ERROR: mask of ELSEWHERE statement is not conformable with the prior mask(s) in its WHERE construct
elsewhere (m2)
  a2 = 2
elsewhere
  a2 = 3
end where
end
