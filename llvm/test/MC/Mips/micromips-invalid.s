# RUN: not llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  addiur1sp $7, 260 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiur1sp $7, 241 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: misaligned immediate operand value
  addiur1sp $8, 240 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addius5 $7, 9 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiusp 1032   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addu16  $6, $14, $4 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  subu16  $5, $16, $9 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  and16   $16, $8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  not16   $18, $9   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  or16    $16, $10  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  xor16   $15, $5   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sll16   $1, $16, 5 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  srl16   $4, $9, 6  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sll16   $3, $16, 9 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  srl16   $4, $5, 15 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  li16  $8, -1 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  li16  $4, -2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiur2 $9, $7, -1 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $6, $7, 10 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
