# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -timeline -timeline-max-iterations=2 < %s | FileCheck %s

# VALU0/VALU1
vpmulld     %xmm0, %xmm1, %xmm2
vpand       %xmm0, %xmm1, %xmm2

# VIMUL/STC
vcvttps2dq  %xmm0, %xmm1
vpclmulqdq  $0, %xmm0, %xmm1, %xmm2

# FPA/FPM
vaddps      %xmm0, %xmm1, %xmm2
vsqrtps     %xmm0, %xmm1

# FPA/FPM YMM
vaddps      %ymm0, %ymm1, %ymm2
vsqrtps     %ymm0, %ymm1


# CHECK:      Iterations:     70
# CHECK-NEXT: Instructions:   560
# CHECK-NEXT: Total Cycles:   3155
# CHECK-NEXT: Dispatch Width: 2
# CHECK-NEXT: IPC:            0.18


# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  1      2     1.00                    	vpmulld	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  1      1     0.50                    	vpand	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  1      3     1.00                    	vcvttps2dq	%xmm0, %xmm1
# CHECK-NEXT:  1      2     1.00                    	vpclmulqdq	$0, %xmm0, %xmm1, %xmm2
# CHECK-NEXT:  1      3     1.00                    	vaddps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  1      21    21.00                   	vsqrtps	%xmm0, %xmm1
# CHECK-NEXT:  1      3     2.00                    	vaddps	%ymm0, %ymm1, %ymm2
# CHECK-NEXT:  1      42    42.00                   	vsqrtps	%ymm0, %ymm1


# CHECK:      Resources:
# CHECK-NEXT: [0] - JALU0
# CHECK-NEXT: [1] - JALU1
# CHECK-NEXT: [2] - JDiv
# CHECK-NEXT: [3] - JFPA
# CHECK-NEXT: [4] - JFPM
# CHECK-NEXT: [5] - JFPU0
# CHECK-NEXT: [6] - JFPU1
# CHECK-NEXT: [7] - JLAGU
# CHECK-NEXT: [8] - JMul
# CHECK-NEXT: [9] - JSAGU
# CHECK-NEXT: [10] - JSTC
# CHECK-NEXT: [11] - JVALU0
# CHECK-NEXT: [12] - JVALU1
# CHECK-NEXT: [13] - JVIMUL


# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]
# CHECK-NEXT:  -      -      -      -     21.00  5.00   45.00   -      -      -      -      -      -     1.00

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   	Instructions:
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -      -     	vpmulld	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -      -      -      -      -     	vpand	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -      -      -      -      -     	vcvttps2dq	%xmm0, %xmm1
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -     1.00   	vpclmulqdq	$0, %xmm0, %xmm1, %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -      -     	vaddps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT:  -      -      -      -     21.00   -     1.00    -      -      -      -      -      -      -     	vsqrtps	%xmm0, %xmm1
# CHECK-NEXT:  -      -      -      -      -     2.00    -      -      -      -      -      -      -      -     	vaddps	%ymm0, %ymm1, %ymm2
# CHECK-NEXT:  -      -      -      -      -      -     42.00   -      -      -      -      -      -      -     	vsqrtps	%ymm0, %ymm1


# CHECK:      Timeline view:
# CHECK-NEXT:      	          0123456789          0123456789          0123456789          0
# CHECK-NEXT: Index	0123456789          0123456789          0123456789          0123456789

# CHECK:      [0,0]	DeeER.    .    .    .    .    .    .    .    .    .    .    .    .    .	vpmulld	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [0,1]	DeE-R.    .    .    .    .    .    .    .    .    .    .    .    .    .	vpand	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [0,2]	.DeeeER   .    .    .    .    .    .    .    .    .    .    .    .    .	vcvttps2dq	%xmm0, %xmm1
# CHECK-NEXT: [0,3]	.D===eeER .    .    .    .    .    .    .    .    .    .    .    .    .	vpclmulqdq	$0, %xmm0, %xmm1, %xmm2
# CHECK-NEXT: [0,4]	. D===eeeER    .    .    .    .    .    .    .    .    .    .    .    .	vaddps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [0,5]	. DeeeeeeeeeeeeeeeeeeeeeER    .    .    .    .    .    .    .    .    .	vsqrtps	%xmm0, %xmm1
# CHECK-NEXT: [0,6]	.  D====================eeeER .    .    .    .    .    .    .    .    .	vaddps	%ymm0, %ymm1, %ymm2
# CHECK-NEXT: [0,7]	.  DeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER  .    .    .    .    .	vsqrtps	%ymm0, %ymm1

# CHECK:      [1,0]	.   D=========================================eeER.    .    .    .    .	vpmulld	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [1,1]	.   D=========================================eE-R.    .    .    .    .	vpand	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [1,2]	.    D=========================================eeeER   .    .    .    .	vcvttps2dq	%xmm0, %xmm1
# CHECK-NEXT: [1,3]	.    D============================================eeER .    .    .    .	vpclmulqdq	$0, %xmm0, %xmm1, %xmm2
# CHECK-NEXT: [1,4]	.    .D============================================eeeER    .    .    .	vaddps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: [1,5]	.    .D=========================================eeeeeeeeeeeeeeeeeeeeeER	vsqrtps	%xmm0, %xmm1


# CHECK:      Average Wait times (based on the timeline view):
# CHECK-NEXT: [0]: Executions
# CHECK-NEXT: [1]: Average time spent waiting in a scheduler's queue
# CHECK-NEXT: [2]: Average time spent waiting in a scheduler's queue while ready
# CHECK-NEXT: [3]: Average time elapsed from WB until retire stage

# CHECK:            [0]    [1]    [2]    [3]
# CHECK-NEXT: 0.     2     21.5   0.5    0.0  	vpmulld	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: 1.     2     21.5   0.5    1.0  	vpand	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: 2.     2     21.5   21.5   0.0  	vcvttps2dq	%xmm0, %xmm1
# CHECK-NEXT: 3.     2     24.5   0.0    0.0  	vpclmulqdq	$0, %xmm0, %xmm1, %xmm2
# CHECK-NEXT: 4.     2     24.5   1.0    0.0  	vaddps	%xmm0, %xmm1, %xmm2
# CHECK-NEXT: 5.     2     21.5   21.5   0.0  	vsqrtps	%xmm0, %xmm1
# CHECK-NEXT: 6.     1     21.0   0.0    0.0  	vaddps	%ymm0, %ymm1, %ymm2
# CHECK-NEXT: 7.     1     1.0    1.0    0.0  	vsqrtps	%ymm0, %ymm1
