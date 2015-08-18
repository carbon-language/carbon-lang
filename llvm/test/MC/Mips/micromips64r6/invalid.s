# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips64r6 -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

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
