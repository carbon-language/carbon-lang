# Assembly code defines set of externally visible labels.

# Shared library generation:
#   llvm-mc -triple=mipsel -filetype=obj -relocation-model=pic \
#           -o=%t1 %p/Inputs/ext1.s
#   lld -flavor gnu -target mipsel -shared -o %t2 %t1

# Executable generation:
#   llvm-mc -triple=mipsel -filetype=obj -o=%t1 %p/Inputs/ext1.s
#   lld -flavor gnu -target mipsel -e ext4 -o %t2 %t1

    .abicalls
    .global ext4
    .type   ext4,@function
    .ent    ext4
ext4:
    nop
    .end    ext4

    .global ext5
    .type   ext5,@function
    .ent    ext5
ext5:
    nop
    .end    ext5

    .global ext6
    .type   ext6,@function
    .ent    ext6
ext6:
    nop
    .end    ext6

    .type   data4,@object
    .comm   data4,4,4
    .type   data5,@object
    .comm   data5,4,4
    .type   data6,@object
    .comm   data6,4,4
