# Assembly code defines set of externally visible labels.

# Shared library generation:
#   llvm-mc -triple=mipsel -filetype=obj -relocation-model=pic \
#           -o=%t1 %p/Inputs/ext.s
#   lld -flavor gnu -target mipsel -shared -o %t2 %t1

# Executable generation:
#   llvm-mc -triple=mipsel -filetype=obj -o=%t1 %p/Inputs/ext.s
#   lld -flavor gnu -target mipsel -e ext1 -o %t2 %t1

    .global ext1
    .ent    ext1
ext1:
    nop
    .end    ext1

    .global ext2
    .ent    ext2
ext2:
    nop
    .end    ext2

    .global ext3
    .ent    ext3
ext3:
    nop
    .end    ext3

    .global ext4
    .ent    ext4
ext4:
    nop
    .end    ext4

    .global ext5
    .ent    ext5
ext5:
    nop
    .end    ext5

    .global ext6
    .ent    ext6
ext6:
    nop
    .end    ext6
