# RUN: llvm-mc -triple=arm64-apple-ios7.0.0 -code-model=small -relocation-model=pic -filetype=obj -o %t.o %s
# RUN: llvm-rtdyld -triple=arm64-apple-ios7.0.0 -verify -check=%s %t.o
# RUN: rm %t.o
# XFAIL: mips

# FIXME: Add GOT relocation tests once GOT testing is supported.

    .section  __TEXT,__text,regular,pure_instructions
    .ios_version_min 7, 0
    .globl  foo
    .align  2
foo:
    movz  w0, #0
    ret

    .globl  _test_branch_reloc
    .align  2


# Test ARM64_RELOC_BRANCH26 relocation. The branch instruction only encodes 26
# bits of the 28-bit possible branch range. The lower two bits are always zero
# and therefore ignored.
# rtdyld-check:  decode_operand(br1, 0)[25:0] = (foo-br1)[27:2]
_test_branch_reloc:
br1:
    b foo
    ret


# Test ARM64_RELOC_UNSIGNED relocation. The absolute 64-bit address of the
# function should be stored at the 8-byte memory location.
# rtdyld-check: *{8}ptr = foo
    .section  __DATA,__data
    .globl  ptr
    .align  3
    .fill 8192, 1, 0
ptr:
    .quad foo


# Test ARM64_RELOC_PAGE21 and ARM64_RELOC_PAGEOFF12 relocation. adrp encodes
# the PC-relative page (4 KiB) difference between the adrp instruction and the
# variable ptr. ldr encodes the offset of the variable within the page. The ldr
# instruction perfroms an implicit shift on the encoded immediate (imm<<3).
# rtdyld-check:  decode_operand(adrp1, 1) = (ptr[32:12]-adrp1[32:12])
# rtdyld-check:  decode_operand(ldr1, 2) = (ptr[11:3])
    .globl  _test_adrp_ldr
    .align  2
_test_adrp_ldr:
adrp1:
    adrp x0, ptr@PAGE
ldr1:
    ldr  x0, [x0, ptr@PAGEOFF]
    ret
    .fill 8192, 1, 0
