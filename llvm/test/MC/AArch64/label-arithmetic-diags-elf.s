// RUN: not llvm-mc -triple aarch64-elf -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

  .data
b:
  .fill 300
e:
  .byte e - b
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: value evaluated as 300 is out of range.
  // CHECK-NEXT: .byte e - b
  // CHECK-NEXT:       ^

  .section sec_x
start:
  .space 5000
end:
  add w0, w1, #(end - start)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: fixup value out of range
  // CHECK-NEXT: add w0, w1, #(end - start)
  // CHECK-NEXT: ^

  cmp w0, #(end - start)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: fixup value out of range
  // CHECK-NEXT: cmp w0, #(end - start)
  // CHECK-NEXT: ^

negative:
  add w0, w1, #(end - negative)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: fixup value out of range
  // CHECK-NEXT: add w0, w1, #(end - negative)
  // CHECK-NEXT: ^

  cmp w0, #(end - negative)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: fixup value out of range
  // CHECK-NEXT: cmp w0, #(end - negative)
  // CHECK-NEXT: ^

  add w0, w1, #(end - external)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: symbol 'external' can not be undefined in a subtraction expression
  // CHECK-NEXT: add w0, w1, #(end - external)
  // CHECK-NEXT: ^

  cmp w0, #(end - external)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: symbol 'external' can not be undefined in a subtraction expression
  // CHECK-NEXT: cmp w0, #(end - external)
  // CHECK-NEXT: ^

  add w0, w1, #:lo12:external - end
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Unsupported pc-relative fixup kind
  // CHECK-NEXT: add w0, w1, #:lo12:external - end
  // CHECK-NEXT: ^

  cmp w0, #:lo12:external - end
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Unsupported pc-relative fixup kind
  // CHECK-NEXT: cmp w0, #:lo12:external - end
  // CHECK-NEXT: ^

  add w0, w1, #:got_lo12:external - end
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Unsupported pc-relative fixup kind
  // CHECK-NEXT: add w0, w1, #:got_lo12:external - end
  // CHECK-NEXT: ^

  cmp w0, #:got_lo12:external - end
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Unsupported pc-relative fixup kind
  // CHECK-NEXT: cmp w0, #:got_lo12:external - end
  // CHECK-NEXT: ^

  .section sec_y
end_across_sec:
  add w0, w1, #(end_across_sec - start)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  // CHECK-NEXT: add w0, w1, #(end_across_sec - start)
  // CHECK-NEXT: ^

  cmp w0, #(end_across_sec - start)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  // CHECK-NEXT: cmp w0, #(end_across_sec - start)
  // CHECK-NEXT: ^

  add w0, w1, #(sec_y - sec_x)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  // CHECK-NEXT: add w0, w1, #(sec_y - sec_x)
  // CHECK-NEXT: ^

  cmp w0, #(sec_y - sec_x)
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  // CHECK-NEXT: cmp w0, #(sec_y - sec_x)
  // CHECK-NEXT: ^
