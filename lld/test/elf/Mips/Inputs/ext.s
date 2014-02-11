# Assembly code defines set of externally visible labels.

# Shared library generation:
#   llvm-mc -triple=mipsel -filetype=obj -relocation-model=pic \
#           -o=%t1 %p/Inputs/ext.s
#   lld -flavor gnu -target mipsel -shared -o %t2 %t1

# Executable generation:
#   llvm-mc -triple=mipsel -filetype=obj -o=%t1 %p/Inputs/ext.s
#   lld -flavor gnu -target mipsel -e ext1 -o %t2 %t1

    .global ext1
    .type   ext1,@function
    .ent    ext1
ext1:
    nop
    .end    ext1

    .global ext2
    .type   ext2,@function
    .ent    ext2
ext2:
    nop
    .end    ext2

    .global ext3
    .type   ext3,@function
    .ent    ext3
ext3:
    nop
    .end    ext3

    .type   data1,@object
    .comm   data1,4,4
    .type   data2,@object
    .comm   data2,4,4
    .type   data3,@object
    .comm   data3,4,4
