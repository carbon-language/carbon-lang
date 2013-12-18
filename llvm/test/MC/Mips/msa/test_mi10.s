# RUN: llvm-mc %s -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        ld.b $w0, -512($1)              # encoding: [0x7a,0x00,0x08,0x20]
# CHECK:        ld.b $w1, 0($2)                 # encoding: [0x78,0x00,0x10,0x60]
# CHECK:        ld.b $w2, 511($3)               # encoding: [0x79,0xff,0x18,0xa0]

# CHECK:        ld.h $w3, -1024($4)             # encoding: [0x7a,0x00,0x20,0xe1]
# CHECK:        ld.h $w4, -512($5)              # encoding: [0x7b,0x00,0x29,0x21]
# CHECK:        ld.h $w5, 0($6)                 # encoding: [0x78,0x00,0x31,0x61]
# CHECK:        ld.h $w6, 512($7)               # encoding: [0x79,0x00,0x39,0xa1]
# CHECK:        ld.h $w7, 1022($8)              # encoding: [0x79,0xff,0x41,0xe1]

# CHECK:        ld.w $w8, -2048($9)             # encoding: [0x7a,0x00,0x4a,0x22]
# CHECK:        ld.w $w9, -1024($10)            # encoding: [0x7b,0x00,0x52,0x62]
# CHECK:        ld.w $w10, -512($11)            # encoding: [0x7b,0x80,0x5a,0xa2]
# CHECK:        ld.w $w11, 512($12)             # encoding: [0x78,0x80,0x62,0xe2]
# CHECK:        ld.w $w12, 1024($13)            # encoding: [0x79,0x00,0x6b,0x22]
# CHECK:        ld.w $w13, 2044($14)            # encoding: [0x79,0xff,0x73,0x62]

# CHECK:        ld.d $w14, -4096($15)           # encoding: [0x7a,0x00,0x7b,0xa3]
# CHECK:        ld.d $w15, -2048($16)           # encoding: [0x7b,0x00,0x83,0xe3]
# CHECK:        ld.d $w16, -1024($17)           # encoding: [0x7b,0x80,0x8c,0x23]
# CHECK:        ld.d $w17, -512($18)            # encoding: [0x7b,0xc0,0x94,0x63]
# CHECK:        ld.d $w18, 0($19)               # encoding: [0x78,0x00,0x9c,0xa3]
# CHECK:        ld.d $w19, 512($20)             # encoding: [0x78,0x40,0xa4,0xe3]
# CHECK:        ld.d $w20, 1024($21)            # encoding: [0x78,0x80,0xad,0x23]
# CHECK:        ld.d $w21, 2048($22)            # encoding: [0x79,0x00,0xb5,0x63]
# CHECK:        ld.d $w22, 4088($23)            # encoding: [0x79,0xff,0xbd,0xa3]

# CHECKOBJDUMP:        ld.b $w0, -512($1)
# CHECKOBJDUMP:        ld.b $w1, 0($2)
# CHECKOBJDUMP:        ld.b $w2, 511($3)

# CHECKOBJDUMP:        ld.h $w3, -1024($4)
# CHECKOBJDUMP:        ld.h $w4, -512($5)
# CHECKOBJDUMP:        ld.h $w5, 0($6)
# CHECKOBJDUMP:        ld.h $w6, 512($7)
# CHECKOBJDUMP:        ld.h $w7, 1022($8)

# CHECKOBJDUMP:        ld.w $w8, -2048($9)
# CHECKOBJDUMP:        ld.w $w9, -1024($10)
# CHECKOBJDUMP:        ld.w $w10, -512($11)
# CHECKOBJDUMP:        ld.w $w11, 512($12)
# CHECKOBJDUMP:        ld.w $w12, 1024($13)
# CHECKOBJDUMP:        ld.w $w13, 2044($14)

# CHECKOBJDUMP:        ld.d $w14, -4096($15)
# CHECKOBJDUMP:        ld.d $w15, -2048($16)
# CHECKOBJDUMP:        ld.d $w16, -1024($17)
# CHECKOBJDUMP:        ld.d $w17, -512($18)
# CHECKOBJDUMP:        ld.d $w18, 0($19)
# CHECKOBJDUMP:        ld.d $w19, 512($20)
# CHECKOBJDUMP:        ld.d $w20, 1024($21)
# CHECKOBJDUMP:        ld.d $w21, 2048($22)
# CHECKOBJDUMP:        ld.d $w22, 4088($23)

        ld.b $w0, -512($1)
        ld.b $w1, 0($2)
        ld.b $w2, 511($3)

        ld.h $w3, -1024($4)
        ld.h $w4, -512($5)
        ld.h $w5, 0($6)
        ld.h $w6, 512($7)
        ld.h $w7, 1022($8)

        ld.w $w8, -2048($9)
        ld.w $w9, -1024($10)
        ld.w $w10, -512($11)
        ld.w $w11, 512($12)
        ld.w $w12, 1024($13)
        ld.w $w13, 2044($14)

        ld.d $w14, -4096($15)
        ld.d $w15, -2048($16)
        ld.d $w16, -1024($17)
        ld.d $w17, -512($18)
        ld.d $w18, 0($19)
        ld.d $w19, 512($20)
        ld.d $w20, 1024($21)
        ld.d $w21, 2048($22)
        ld.d $w22, 4088($23)
