# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: brc	0, foo                  # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	0, foo

#CHECK: brc	1, foo                  # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jo	foo                     # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	1, foo
	jo	foo

#CHECK: brc	2, foo                  # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jh	foo                     # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	2, foo
	jh	foo

#CHECK: brc	3, foo                  # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnle	foo                     # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	3, foo
	jnle	foo

#CHECK: brc	4, foo                  # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jl	foo                     # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	4, foo
	jl	foo

#CHECK: brc	5, foo                  # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnhe	foo                     # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	5, foo
	jnhe	foo

#CHECK: brc	6, foo                  # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jlh	foo                     # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	6, foo
	jlh	foo

#CHECK: brc	7, foo                  # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jne	foo                     # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	7, foo
	jne	foo

#CHECK: brc	8, foo                  # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: je	foo                     # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	8, foo
	je	foo

#CHECK: brc	9, foo                  # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnlh	foo                     # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	9, foo
	jnlh	foo

#CHECK: brc	10, foo                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jhe	foo                     # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	10, foo
	jhe	foo

#CHECK: brc	11, foo                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnl	foo                     # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	11, foo
	jnl	foo

#CHECK: brc	12, foo                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jle	foo                     # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	12, foo
	jle	foo

#CHECK: brc	13, foo                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnh	foo                     # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	13, foo
	jnh	foo

#CHECK: brc	14, foo                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jno	foo                     # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	14, foo
	jno	foo

#CHECK: brc	15, foo                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: j	foo                     # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	15, foo
	j	foo

#CHECK: brc	0, bar+100              # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	brc	0, bar+100

#CHECK: jo	bar+100                 # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jo	bar+100

#CHECK: jh	bar+100                 # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jh	bar+100

#CHECK: jnle	bar+100                 # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnle	bar+100

#CHECK: jl	bar+100                 # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jl	bar+100

#CHECK: jnhe	bar+100                 # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnhe	bar+100

#CHECK: jlh	bar+100                 # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jlh	bar+100

#CHECK: jne	bar+100                 # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jne	bar+100

#CHECK: je	bar+100                 # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	je	bar+100

#CHECK: jnlh	bar+100                 # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnlh	bar+100

#CHECK: jhe	bar+100                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jhe	bar+100

#CHECK: jnl	bar+100                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnl	bar+100

#CHECK: jle	bar+100                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jle	bar+100

#CHECK: jnh	bar+100                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnh	bar+100

#CHECK: jno	bar+100                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jno	bar+100

#CHECK: j	bar+100                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	j	bar+100

#CHECK: brc	0, bar@PLT              # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	brc	0, bar@PLT

#CHECK: jo	bar@PLT                 # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jo	bar@PLT

#CHECK: jh	bar@PLT                 # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jh	bar@PLT

#CHECK: jnle	bar@PLT                 # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnle	bar@PLT

#CHECK: jl	bar@PLT                 # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jl	bar@PLT

#CHECK: jnhe	bar@PLT                 # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnhe	bar@PLT

#CHECK: jlh	bar@PLT                 # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jlh	bar@PLT

#CHECK: jne	bar@PLT                 # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jne	bar@PLT

#CHECK: je	bar@PLT                 # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	je	bar@PLT

#CHECK: jnlh	bar@PLT                 # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnlh	bar@PLT

#CHECK: jhe	bar@PLT                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jhe	bar@PLT

#CHECK: jnl	bar@PLT                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnl	bar@PLT

#CHECK: jle	bar@PLT                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jle	bar@PLT

#CHECK: jnh	bar@PLT                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnh	bar@PLT

#CHECK: jno	bar@PLT                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jno	bar@PLT

#CHECK: j	bar@PLT                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	j	bar@PLT
