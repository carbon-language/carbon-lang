# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips64r6 -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  addiur1sp $7, 260        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiur1sp $7, 241        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: misaligned immediate operand value
  addiur1sp $8, 240        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $9, $7, -1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $6, $7, 10       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addius5 $7, 9            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiusp 1032             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  ddiv $32, $4, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $3, $34, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $3, $4, $35         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $32, $4, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $3, $34, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $3, $4, $35         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $32, $4, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $3, $34, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $3, $4, $35        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $32, $4, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $3, $34, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $3, $4, $35        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
