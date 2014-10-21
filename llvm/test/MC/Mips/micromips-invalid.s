# RUN: not llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  addius5 $7, 9 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiusp 1032   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  and16   $16, $8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  not16   $18, $9   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  or16    $16, $10  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  xor16   $15, $5   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
