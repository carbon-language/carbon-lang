# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -triple=mipsel-unknown-linux -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        ld.b    $w2, 1($7)              # encoding: [0x78,0x01,0x38,0xa0]
# CHECK:        ld.h    $w16, -9($zero)         # encoding: [0x7b,0xf7,0x04,0x21]
# CHECK:        ld.w    $w13, -6($4)            # encoding: [0x7b,0xfa,0x23,0x62]
# CHECK:        ld.d    $w1, -5($16)            # encoding: [0x7b,0xfb,0x80,0x63]
# CHECK:        st.b    $w29, 1($14)            # encoding: [0x78,0x01,0x77,0x64]
# CHECK:        st.h    $w6, -1($8)             # encoding: [0x7b,0xff,0x41,0xa5]
# CHECK:        st.w    $w18, 8($15)            # encoding: [0x78,0x08,0x7c,0xa6]
# CHECK:        st.d    $w3, -14($18)           # encoding: [0x7b,0xf2,0x90,0xe7]

# CHECKOBJDUMP:        ld.b    $w2, 1($7)
# CHECKOBJDUMP:        ld.h    $w16, -9($zero)
# CHECKOBJDUMP:        ld.w    $w13, -6($4)
# CHECKOBJDUMP:        ld.d    $w1, -5($16)
# CHECKOBJDUMP:        st.b    $w29, 1($14)
# CHECKOBJDUMP:        st.h    $w6, -1($8)
# CHECKOBJDUMP:        st.w    $w18, 8($15)
# CHECKOBJDUMP:        st.d    $w3, -14($18)

                ld.b    $w2, 1($7)
                ld.h    $w16, -9($zero)
                ld.w    $w13, -6($4)
                ld.d    $w1, -5($16)
                st.b    $w29, 1($14)
                st.h    $w6, -1($8)
                st.w    $w18, 8($15)
                st.d    $w3, -14($18)
