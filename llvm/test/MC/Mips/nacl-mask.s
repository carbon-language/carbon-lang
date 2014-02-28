# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-nacl %s \
# RUN:  | llvm-objdump -triple mipsel -disassemble -no-show-raw-insn - \
# RUN:  | FileCheck %s

# This test tests that address-masking sandboxing is added when given assembly
# input.

test1:
	.set	noreorder

        jr	$a0
        nop
        jr	$ra
        nop

# CHECK-LABEL:   test1:

# CHECK:         and     $4, $4, $14
# CHECK-NEXT:    jr      $4

# Check that additional nop is inserted, to align mask and jr to the next
# bundle.

# CHECK-NEXT:    nop
# CHECK-NEXT:    nop

# CHECK:         and     $ra, $ra, $14
# CHECK-NEXT:    jr      $ra
