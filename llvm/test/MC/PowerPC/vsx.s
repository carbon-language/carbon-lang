# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: lxsdx 7, 5, 31                     # encoding: [0x7c,0xe5,0xfc,0x98]
# CHECK-LE: lxsdx 7, 5, 31                     # encoding: [0x98,0xfc,0xe5,0x7c]
            lxsdx 7, 5, 31
# CHECK-BE: lxvd2x 7, 5, 31                    # encoding: [0x7c,0xe5,0xfe,0x98]
# CHECK-LE: lxvd2x 7, 5, 31                    # encoding: [0x98,0xfe,0xe5,0x7c]
            lxvd2x 7, 5, 31
# CHECK-BE: lxvdsx 7, 5, 31                    # encoding: [0x7c,0xe5,0xfa,0x98]
# CHECK-LE: lxvdsx 7, 5, 31                    # encoding: [0x98,0xfa,0xe5,0x7c]
            lxvdsx 7, 5, 31
# CHECK-BE: lxvw4x 7, 5, 31                    # encoding: [0x7c,0xe5,0xfe,0x18]
# CHECK-LE: lxvw4x 7, 5, 31                    # encoding: [0x18,0xfe,0xe5,0x7c]
            lxvw4x 7, 5, 31
# CHECK-BE: stxsdx 8, 5, 31                    # encoding: [0x7d,0x05,0xfd,0x98]
# CHECK-LE: stxsdx 8, 5, 31                    # encoding: [0x98,0xfd,0x05,0x7d]
            stxsdx 8, 5, 31
# CHECK-BE: stxvd2x 8, 5, 31                   # encoding: [0x7d,0x05,0xff,0x98]
# CHECK-LE: stxvd2x 8, 5, 31                   # encoding: [0x98,0xff,0x05,0x7d]
            stxvd2x 8, 5, 31
# CHECK-BE: stxvw4x 8, 5, 31                   # encoding: [0x7d,0x05,0xff,0x18]
# CHECK-LE: stxvw4x 8, 5, 31                   # encoding: [0x18,0xff,0x05,0x7d]
            stxvw4x 8, 5, 31
# CHECK-BE: xsabsdp 7, 27                      # encoding: [0xf0,0xe0,0xdd,0x64]
# CHECK-LE: xsabsdp 7, 27                      # encoding: [0x64,0xdd,0xe0,0xf0]
            xsabsdp 7, 27
# CHECK-BE: xsadddp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x04]
# CHECK-LE: xsadddp 7, 63, 27                  # encoding: [0x04,0xd9,0xff,0xf0]
            xsadddp 7, 63, 27
# CHECK-BE: xscmpodp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0x5c]
# CHECK-LE: xscmpodp 6, 63, 27                 # encoding: [0x5c,0xd9,0x1f,0xf3]
            xscmpodp 6, 63, 27
# CHECK-BE: xscmpudp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0x1c]
# CHECK-LE: xscmpudp 6, 63, 27                 # encoding: [0x1c,0xd9,0x1f,0xf3]
            xscmpudp 6, 63, 27
# CHECK-BE: xscpsgndp 7, 63, 27                # encoding: [0xf0,0xff,0xdd,0x84]
# CHECK-LE: xscpsgndp 7, 63, 27                # encoding: [0x84,0xdd,0xff,0xf0]
            xscpsgndp 7, 63, 27
# CHECK-BE: xscvdpsp 7, 27                     # encoding: [0xf0,0xe0,0xdc,0x24]
# CHECK-LE: xscvdpsp 7, 27                     # encoding: [0x24,0xdc,0xe0,0xf0]
            xscvdpsp 7, 27
# CHECK-BE: xscvdpsxds 7, 27                   # encoding: [0xf0,0xe0,0xdd,0x60]
# CHECK-LE: xscvdpsxds 7, 27                   # encoding: [0x60,0xdd,0xe0,0xf0]
            xscvdpsxds 7, 27
# CHECK-BE: xscvdpsxws 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x60]
# CHECK-LE: xscvdpsxws 7, 27                   # encoding: [0x60,0xd9,0xe0,0xf0]
            xscvdpsxws 7, 27
# CHECK-BE: xscvdpuxds 7, 27                   # encoding: [0xf0,0xe0,0xdd,0x20]
# CHECK-LE: xscvdpuxds 7, 27                   # encoding: [0x20,0xdd,0xe0,0xf0]
            xscvdpuxds 7, 27
# CHECK-BE: xscvdpuxws 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x20]
# CHECK-LE: xscvdpuxws 7, 27                   # encoding: [0x20,0xd9,0xe0,0xf0]
            xscvdpuxws 7, 27
# CHECK-BE: xscvspdp 7, 27                     # encoding: [0xf0,0xe0,0xdd,0x24]
# CHECK-LE: xscvspdp 7, 27                     # encoding: [0x24,0xdd,0xe0,0xf0]
            xscvspdp 7, 27
# CHECK-BE: xscvsxddp 7, 27                    # encoding: [0xf0,0xe0,0xdd,0xe0]
# CHECK-LE: xscvsxddp 7, 27                    # encoding: [0xe0,0xdd,0xe0,0xf0]
            xscvsxddp 7, 27
# CHECK-BE: xscvuxddp 7, 27                    # encoding: [0xf0,0xe0,0xdd,0xa0]
# CHECK-LE: xscvuxddp 7, 27                    # encoding: [0xa0,0xdd,0xe0,0xf0]
            xscvuxddp 7, 27
# CHECK-BE: xsdivdp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0xc4]
# CHECK-LE: xsdivdp 7, 63, 27                  # encoding: [0xc4,0xd9,0xff,0xf0]
            xsdivdp 7, 63, 27
# CHECK-BE: xsmaddadp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x0c]
# CHECK-LE: xsmaddadp 7, 63, 27                # encoding: [0x0c,0xd9,0xff,0xf0]
            xsmaddadp 7, 63, 27
# CHECK-BE: xsmaddmdp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x4c]
# CHECK-LE: xsmaddmdp 7, 63, 27                # encoding: [0x4c,0xd9,0xff,0xf0]
            xsmaddmdp 7, 63, 27
# CHECK-BE: xsmaxdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x04]
# CHECK-LE: xsmaxdp 7, 63, 27                  # encoding: [0x04,0xdd,0xff,0xf0]
            xsmaxdp 7, 63, 27
# CHECK-BE: xsmindp 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x44]
# CHECK-LE: xsmindp 7, 63, 27                  # encoding: [0x44,0xdd,0xff,0xf0]
            xsmindp 7, 63, 27
# CHECK-BE: xsmsubadp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x8c]
# CHECK-LE: xsmsubadp 7, 63, 27                # encoding: [0x8c,0xd9,0xff,0xf0]
            xsmsubadp 7, 63, 27
# CHECK-BE: xsmsubmdp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0xcc]
# CHECK-LE: xsmsubmdp 7, 63, 27                # encoding: [0xcc,0xd9,0xff,0xf0]
            xsmsubmdp 7, 63, 27
# CHECK-BE: xsmuldp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x84]
# CHECK-LE: xsmuldp 7, 63, 27                  # encoding: [0x84,0xd9,0xff,0xf0]
            xsmuldp 7, 63, 27
# CHECK-BE: xsnabsdp 7, 27                     # encoding: [0xf0,0xe0,0xdd,0xa4]
# CHECK-LE: xsnabsdp 7, 27                     # encoding: [0xa4,0xdd,0xe0,0xf0]
            xsnabsdp 7, 27
# CHECK-BE: xsnegdp 7, 27                      # encoding: [0xf0,0xe0,0xdd,0xe4]
# CHECK-LE: xsnegdp 7, 27                      # encoding: [0xe4,0xdd,0xe0,0xf0]
            xsnegdp 7, 27
# CHECK-BE: xsnmaddadp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x0c]
# CHECK-LE: xsnmaddadp 7, 63, 27               # encoding: [0x0c,0xdd,0xff,0xf0]
            xsnmaddadp 7, 63, 27
# CHECK-BE: xsnmaddmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x4c]
# CHECK-LE: xsnmaddmdp 7, 63, 27               # encoding: [0x4c,0xdd,0xff,0xf0]
            xsnmaddmdp 7, 63, 27
# CHECK-BE: xsnmsubadp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x8c]
# CHECK-LE: xsnmsubadp 7, 63, 27               # encoding: [0x8c,0xdd,0xff,0xf0]
            xsnmsubadp 7, 63, 27
# CHECK-BE: xsnmsubmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0xcc]
# CHECK-LE: xsnmsubmdp 7, 63, 27               # encoding: [0xcc,0xdd,0xff,0xf0]
            xsnmsubmdp 7, 63, 27
# CHECK-BE: xsrdpi 7, 27                       # encoding: [0xf0,0xe0,0xd9,0x24]
# CHECK-LE: xsrdpi 7, 27                       # encoding: [0x24,0xd9,0xe0,0xf0]
            xsrdpi 7, 27
# CHECK-BE: xsrdpic 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xac]
# CHECK-LE: xsrdpic 7, 27                      # encoding: [0xac,0xd9,0xe0,0xf0]
            xsrdpic 7, 27
# CHECK-BE: xsrdpim 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xe4]
# CHECK-LE: xsrdpim 7, 27                      # encoding: [0xe4,0xd9,0xe0,0xf0]
            xsrdpim 7, 27
# CHECK-BE: xsrdpip 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xa4]
# CHECK-LE: xsrdpip 7, 27                      # encoding: [0xa4,0xd9,0xe0,0xf0]
            xsrdpip 7, 27
# CHECK-BE: xsrdpiz 7, 27                      # encoding: [0xf0,0xe0,0xd9,0x64]
# CHECK-LE: xsrdpiz 7, 27                      # encoding: [0x64,0xd9,0xe0,0xf0]
            xsrdpiz 7, 27
# CHECK-BE: xsredp 7, 27                       # encoding: [0xf0,0xe0,0xd9,0x68]
# CHECK-LE: xsredp 7, 27                       # encoding: [0x68,0xd9,0xe0,0xf0]
            xsredp 7, 27
# CHECK-BE: xsrsqrtedp 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x28]
# CHECK-LE: xsrsqrtedp 7, 27                   # encoding: [0x28,0xd9,0xe0,0xf0]
            xsrsqrtedp 7, 27
# CHECK-BE: xssqrtdp 7, 27                     # encoding: [0xf0,0xe0,0xd9,0x2c]
# CHECK-LE: xssqrtdp 7, 27                     # encoding: [0x2c,0xd9,0xe0,0xf0]
            xssqrtdp 7, 27
# CHECK-BE: xssubdp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x44]
# CHECK-LE: xssubdp 7, 63, 27                  # encoding: [0x44,0xd9,0xff,0xf0]
            xssubdp 7, 63, 27
# CHECK-BE: xstdivdp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0xec]
# CHECK-LE: xstdivdp 6, 63, 27                 # encoding: [0xec,0xd9,0x1f,0xf3]
            xstdivdp 6, 63, 27
# CHECK-BE: xstsqrtdp 6, 27                    # encoding: [0xf3,0x00,0xd9,0xa8]
# CHECK-LE: xstsqrtdp 6, 27                    # encoding: [0xa8,0xd9,0x00,0xf3]
            xstsqrtdp 6, 27
# CHECK-BE: xvabsdp 7, 27                      # encoding: [0xf0,0xe0,0xdf,0x64]
# CHECK-LE: xvabsdp 7, 27                      # encoding: [0x64,0xdf,0xe0,0xf0]
            xvabsdp 7, 27
# CHECK-BE: xvabssp 7, 27                      # encoding: [0xf0,0xe0,0xde,0x64]
# CHECK-LE: xvabssp 7, 27                      # encoding: [0x64,0xde,0xe0,0xf0]
            xvabssp 7, 27
# CHECK-BE: xvadddp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x04]
# CHECK-LE: xvadddp 7, 63, 27                  # encoding: [0x04,0xdb,0xff,0xf0]
            xvadddp 7, 63, 27
# CHECK-BE: xvaddsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x04]
# CHECK-LE: xvaddsp 7, 63, 27                  # encoding: [0x04,0xda,0xff,0xf0]
            xvaddsp 7, 63, 27
# CHECK-BE: xvcmpeqdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x1c]
# CHECK-LE: xvcmpeqdp 7, 63, 27                # encoding: [0x1c,0xdb,0xff,0xf0]
            xvcmpeqdp 7, 63, 27
# CHECK-BE: xvcmpeqdp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x1c]
# CHECK-LE: xvcmpeqdp. 7, 63, 27               # encoding: [0x1c,0xdf,0xff,0xf0]
            xvcmpeqdp. 7, 63, 27
# CHECK-BE: xvcmpeqsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x1c]
# CHECK-LE: xvcmpeqsp 7, 63, 27                # encoding: [0x1c,0xda,0xff,0xf0]
            xvcmpeqsp 7, 63, 27
# CHECK-BE: xvcmpeqsp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x1c]
# CHECK-LE: xvcmpeqsp. 7, 63, 27               # encoding: [0x1c,0xde,0xff,0xf0]
            xvcmpeqsp. 7, 63, 27
# CHECK-BE: xvcmpgedp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x9c]
# CHECK-LE: xvcmpgedp 7, 63, 27                # encoding: [0x9c,0xdb,0xff,0xf0]
            xvcmpgedp 7, 63, 27
# CHECK-BE: xvcmpgedp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x9c]
# CHECK-LE: xvcmpgedp. 7, 63, 27               # encoding: [0x9c,0xdf,0xff,0xf0]
            xvcmpgedp. 7, 63, 27
# CHECK-BE: xvcmpgesp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x9c]
# CHECK-LE: xvcmpgesp 7, 63, 27                # encoding: [0x9c,0xda,0xff,0xf0]
            xvcmpgesp 7, 63, 27
# CHECK-BE: xvcmpgesp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x9c]
# CHECK-LE: xvcmpgesp. 7, 63, 27               # encoding: [0x9c,0xde,0xff,0xf0]
            xvcmpgesp. 7, 63, 27
# CHECK-BE: xvcmpgtdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x5c]
# CHECK-LE: xvcmpgtdp 7, 63, 27                # encoding: [0x5c,0xdb,0xff,0xf0]
            xvcmpgtdp 7, 63, 27
# CHECK-BE: xvcmpgtdp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x5c]
# CHECK-LE: xvcmpgtdp. 7, 63, 27               # encoding: [0x5c,0xdf,0xff,0xf0]
            xvcmpgtdp. 7, 63, 27
# CHECK-BE: xvcmpgtsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x5c]
# CHECK-LE: xvcmpgtsp 7, 63, 27                # encoding: [0x5c,0xda,0xff,0xf0]
            xvcmpgtsp 7, 63, 27
# CHECK-BE: xvcmpgtsp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x5c]
# CHECK-LE: xvcmpgtsp. 7, 63, 27               # encoding: [0x5c,0xde,0xff,0xf0]
            xvcmpgtsp. 7, 63, 27
# CHECK-BE: xvcpsgndp 7, 63, 27                # encoding: [0xf0,0xff,0xdf,0x84]
# CHECK-LE: xvcpsgndp 7, 63, 27                # encoding: [0x84,0xdf,0xff,0xf0]
            xvcpsgndp 7, 63, 27
# CHECK-BE: xvcpsgnsp 7, 63, 27                # encoding: [0xf0,0xff,0xde,0x84]
# CHECK-LE: xvcpsgnsp 7, 63, 27                # encoding: [0x84,0xde,0xff,0xf0]
            xvcpsgnsp 7, 63, 27
# CHECK-BE: xvcvdpsp 7, 27                     # encoding: [0xf0,0xe0,0xde,0x24]
# CHECK-LE: xvcvdpsp 7, 27                     # encoding: [0x24,0xde,0xe0,0xf0]
            xvcvdpsp 7, 27
# CHECK-BE: xvcvdpsxds 7, 27                   # encoding: [0xf0,0xe0,0xdf,0x60]
# CHECK-LE: xvcvdpsxds 7, 27                   # encoding: [0x60,0xdf,0xe0,0xf0]
            xvcvdpsxds 7, 27
# CHECK-BE: xvcvdpsxws 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x60]
# CHECK-LE: xvcvdpsxws 7, 27                   # encoding: [0x60,0xdb,0xe0,0xf0]
            xvcvdpsxws 7, 27
# CHECK-BE: xvcvdpuxds 7, 27                   # encoding: [0xf0,0xe0,0xdf,0x20]
# CHECK-LE: xvcvdpuxds 7, 27                   # encoding: [0x20,0xdf,0xe0,0xf0]
            xvcvdpuxds 7, 27
# CHECK-BE: xvcvdpuxws 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x20]
# CHECK-LE: xvcvdpuxws 7, 27                   # encoding: [0x20,0xdb,0xe0,0xf0]
            xvcvdpuxws 7, 27
# CHECK-BE: xvcvspdp 7, 27                     # encoding: [0xf0,0xe0,0xdf,0x24]
# CHECK-LE: xvcvspdp 7, 27                     # encoding: [0x24,0xdf,0xe0,0xf0]
            xvcvspdp 7, 27
# CHECK-BE: xvcvspsxds 7, 27                   # encoding: [0xf0,0xe0,0xde,0x60]
# CHECK-LE: xvcvspsxds 7, 27                   # encoding: [0x60,0xde,0xe0,0xf0]
            xvcvspsxds 7, 27
# CHECK-BE: xvcvspsxws 7, 27                   # encoding: [0xf0,0xe0,0xda,0x60]
# CHECK-LE: xvcvspsxws 7, 27                   # encoding: [0x60,0xda,0xe0,0xf0]
            xvcvspsxws 7, 27
# CHECK-BE: xvcvspuxds 7, 27                   # encoding: [0xf0,0xe0,0xde,0x20]
# CHECK-LE: xvcvspuxds 7, 27                   # encoding: [0x20,0xde,0xe0,0xf0]
            xvcvspuxds 7, 27
# CHECK-BE: xvcvspuxws 7, 27                   # encoding: [0xf0,0xe0,0xda,0x20]
# CHECK-LE: xvcvspuxws 7, 27                   # encoding: [0x20,0xda,0xe0,0xf0]
            xvcvspuxws 7, 27
# CHECK-BE: xvcvsxddp 7, 27                    # encoding: [0xf0,0xe0,0xdf,0xe0]
# CHECK-LE: xvcvsxddp 7, 27                    # encoding: [0xe0,0xdf,0xe0,0xf0]
            xvcvsxddp 7, 27
# CHECK-BE: xvcvsxdsp 7, 27                    # encoding: [0xf0,0xe0,0xde,0xe0]
# CHECK-LE: xvcvsxdsp 7, 27                    # encoding: [0xe0,0xde,0xe0,0xf0]
            xvcvsxdsp 7, 27
# CHECK-BE: xvcvsxwdp 7, 27                    # encoding: [0xf0,0xe0,0xdb,0xe0]
# CHECK-LE: xvcvsxwdp 7, 27                    # encoding: [0xe0,0xdb,0xe0,0xf0]
            xvcvsxwdp 7, 27
# CHECK-BE: xvcvsxwsp 7, 27                    # encoding: [0xf0,0xe0,0xda,0xe0]
# CHECK-LE: xvcvsxwsp 7, 27                    # encoding: [0xe0,0xda,0xe0,0xf0]
            xvcvsxwsp 7, 27
# CHECK-BE: xvcvuxddp 7, 27                    # encoding: [0xf0,0xe0,0xdf,0xa0]
# CHECK-LE: xvcvuxddp 7, 27                    # encoding: [0xa0,0xdf,0xe0,0xf0]
            xvcvuxddp 7, 27
# CHECK-BE: xvcvuxdsp 7, 27                    # encoding: [0xf0,0xe0,0xde,0xa0]
# CHECK-LE: xvcvuxdsp 7, 27                    # encoding: [0xa0,0xde,0xe0,0xf0]
            xvcvuxdsp 7, 27
# CHECK-BE: xvcvuxwdp 7, 27                    # encoding: [0xf0,0xe0,0xdb,0xa0]
# CHECK-LE: xvcvuxwdp 7, 27                    # encoding: [0xa0,0xdb,0xe0,0xf0]
            xvcvuxwdp 7, 27
# CHECK-BE: xvcvuxwsp 7, 27                    # encoding: [0xf0,0xe0,0xda,0xa0]
# CHECK-LE: xvcvuxwsp 7, 27                    # encoding: [0xa0,0xda,0xe0,0xf0]
            xvcvuxwsp 7, 27
# CHECK-BE: xvdivdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0xc4]
# CHECK-LE: xvdivdp 7, 63, 27                  # encoding: [0xc4,0xdb,0xff,0xf0]
            xvdivdp 7, 63, 27
# CHECK-BE: xvdivsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0xc4]
# CHECK-LE: xvdivsp 7, 63, 27                  # encoding: [0xc4,0xda,0xff,0xf0]
            xvdivsp 7, 63, 27
# CHECK-BE: xvmaddadp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x0c]
# CHECK-LE: xvmaddadp 7, 63, 27                # encoding: [0x0c,0xdb,0xff,0xf0]
            xvmaddadp 7, 63, 27
# CHECK-BE: xvmaddasp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x0c]
# CHECK-LE: xvmaddasp 7, 63, 27                # encoding: [0x0c,0xda,0xff,0xf0]
            xvmaddasp 7, 63, 27
# CHECK-BE: xvmaddmdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x4c]
# CHECK-LE: xvmaddmdp 7, 63, 27                # encoding: [0x4c,0xdb,0xff,0xf0]
            xvmaddmdp 7, 63, 27
# CHECK-BE: xvmaddmsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x4c]
# CHECK-LE: xvmaddmsp 7, 63, 27                # encoding: [0x4c,0xda,0xff,0xf0]
            xvmaddmsp 7, 63, 27
# CHECK-BE: xvmaxdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdf,0x04]
# CHECK-LE: xvmaxdp 7, 63, 27                  # encoding: [0x04,0xdf,0xff,0xf0]
            xvmaxdp 7, 63, 27
# CHECK-BE: xvmaxsp 7, 63, 27                  # encoding: [0xf0,0xff,0xde,0x04]
# CHECK-LE: xvmaxsp 7, 63, 27                  # encoding: [0x04,0xde,0xff,0xf0]
            xvmaxsp 7, 63, 27
# CHECK-BE: xvmindp 7, 63, 27                  # encoding: [0xf0,0xff,0xdf,0x44]
# CHECK-LE: xvmindp 7, 63, 27                  # encoding: [0x44,0xdf,0xff,0xf0]
            xvmindp 7, 63, 27
# CHECK-BE: xvminsp 7, 63, 27                  # encoding: [0xf0,0xff,0xde,0x44]
# CHECK-LE: xvminsp 7, 63, 27                  # encoding: [0x44,0xde,0xff,0xf0]
            xvminsp 7, 63, 27
# CHECK-BE: xvcpsgndp 7, 63, 63                # encoding: [0xf0,0xff,0xff,0x86]
# CHECK-LE: xvcpsgndp 7, 63, 63                # encoding: [0x86,0xff,0xff,0xf0]
            xvmovdp 7, 63
# CHECK-BE: xvcpsgnsp 7, 63, 63                # encoding: [0xf0,0xff,0xfe,0x86]
# CHECK-LE: xvcpsgnsp 7, 63, 63                # encoding: [0x86,0xfe,0xff,0xf0]
            xvmovsp 7, 63
# CHECK-BE: xvmsubadp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x8c]
# CHECK-LE: xvmsubadp 7, 63, 27                # encoding: [0x8c,0xdb,0xff,0xf0]
            xvmsubadp 7, 63, 27
# CHECK-BE: xvmsubasp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x8c]
# CHECK-LE: xvmsubasp 7, 63, 27                # encoding: [0x8c,0xda,0xff,0xf0]
            xvmsubasp 7, 63, 27
# CHECK-BE: xvmsubmdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0xcc]
# CHECK-LE: xvmsubmdp 7, 63, 27                # encoding: [0xcc,0xdb,0xff,0xf0]
            xvmsubmdp 7, 63, 27
# CHECK-BE: xvmsubmsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0xcc]
# CHECK-LE: xvmsubmsp 7, 63, 27                # encoding: [0xcc,0xda,0xff,0xf0]
            xvmsubmsp 7, 63, 27
# CHECK-BE: xvmuldp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x84]
# CHECK-LE: xvmuldp 7, 63, 27                  # encoding: [0x84,0xdb,0xff,0xf0]
            xvmuldp 7, 63, 27
# CHECK-BE: xvmulsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x84]
# CHECK-LE: xvmulsp 7, 63, 27                  # encoding: [0x84,0xda,0xff,0xf0]
            xvmulsp 7, 63, 27
# CHECK-BE: xvnabsdp 7, 27                     # encoding: [0xf0,0xe0,0xdf,0xa4]
# CHECK-LE: xvnabsdp 7, 27                     # encoding: [0xa4,0xdf,0xe0,0xf0]
            xvnabsdp 7, 27
# CHECK-BE: xvnabssp 7, 27                     # encoding: [0xf0,0xe0,0xde,0xa4]
# CHECK-LE: xvnabssp 7, 27                     # encoding: [0xa4,0xde,0xe0,0xf0]
            xvnabssp 7, 27
# CHECK-BE: xvnegdp 7, 27                      # encoding: [0xf0,0xe0,0xdf,0xe4]
# CHECK-LE: xvnegdp 7, 27                      # encoding: [0xe4,0xdf,0xe0,0xf0]
            xvnegdp 7, 27
# CHECK-BE: xvnegsp 7, 27                      # encoding: [0xf0,0xe0,0xde,0xe4]
# CHECK-LE: xvnegsp 7, 27                      # encoding: [0xe4,0xde,0xe0,0xf0]
            xvnegsp 7, 27
# CHECK-BE: xvnmaddadp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x0c]
# CHECK-LE: xvnmaddadp 7, 63, 27               # encoding: [0x0c,0xdf,0xff,0xf0]
            xvnmaddadp 7, 63, 27
# CHECK-BE: xvnmaddasp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x0c]
# CHECK-LE: xvnmaddasp 7, 63, 27               # encoding: [0x0c,0xde,0xff,0xf0]
            xvnmaddasp 7, 63, 27
# CHECK-BE: xvnmaddmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x4c]
# CHECK-LE: xvnmaddmdp 7, 63, 27               # encoding: [0x4c,0xdf,0xff,0xf0]
            xvnmaddmdp 7, 63, 27
# CHECK-BE: xvnmaddmsp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x4c]
# CHECK-LE: xvnmaddmsp 7, 63, 27               # encoding: [0x4c,0xde,0xff,0xf0]
            xvnmaddmsp 7, 63, 27
# CHECK-BE: xvnmsubadp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x8c]
# CHECK-LE: xvnmsubadp 7, 63, 27               # encoding: [0x8c,0xdf,0xff,0xf0]
            xvnmsubadp 7, 63, 27
# CHECK-BE: xvnmsubasp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x8c]
# CHECK-LE: xvnmsubasp 7, 63, 27               # encoding: [0x8c,0xde,0xff,0xf0]
            xvnmsubasp 7, 63, 27
# CHECK-BE: xvnmsubmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0xcc]
# CHECK-LE: xvnmsubmdp 7, 63, 27               # encoding: [0xcc,0xdf,0xff,0xf0]
            xvnmsubmdp 7, 63, 27
# CHECK-BE: xvnmsubmsp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0xcc]
# CHECK-LE: xvnmsubmsp 7, 63, 27               # encoding: [0xcc,0xde,0xff,0xf0]
            xvnmsubmsp 7, 63, 27
# CHECK-BE: xvrdpi 7, 27                       # encoding: [0xf0,0xe0,0xdb,0x24]
# CHECK-LE: xvrdpi 7, 27                       # encoding: [0x24,0xdb,0xe0,0xf0]
            xvrdpi 7, 27
# CHECK-BE: xvrdpic 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xac]
# CHECK-LE: xvrdpic 7, 27                      # encoding: [0xac,0xdb,0xe0,0xf0]
            xvrdpic 7, 27
# CHECK-BE: xvrdpim 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xe4]
# CHECK-LE: xvrdpim 7, 27                      # encoding: [0xe4,0xdb,0xe0,0xf0]
            xvrdpim 7, 27
# CHECK-BE: xvrdpip 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xa4]
# CHECK-LE: xvrdpip 7, 27                      # encoding: [0xa4,0xdb,0xe0,0xf0]
            xvrdpip 7, 27
# CHECK-BE: xvrdpiz 7, 27                      # encoding: [0xf0,0xe0,0xdb,0x64]
# CHECK-LE: xvrdpiz 7, 27                      # encoding: [0x64,0xdb,0xe0,0xf0]
            xvrdpiz 7, 27
# CHECK-BE: xvredp 7, 27                       # encoding: [0xf0,0xe0,0xdb,0x68]
# CHECK-LE: xvredp 7, 27                       # encoding: [0x68,0xdb,0xe0,0xf0]
            xvredp 7, 27
# CHECK-BE: xvresp 7, 27                       # encoding: [0xf0,0xe0,0xda,0x68]
# CHECK-LE: xvresp 7, 27                       # encoding: [0x68,0xda,0xe0,0xf0]
            xvresp 7, 27
# CHECK-BE: xvrspi 7, 27                       # encoding: [0xf0,0xe0,0xda,0x24]
# CHECK-LE: xvrspi 7, 27                       # encoding: [0x24,0xda,0xe0,0xf0]
            xvrspi 7, 27
# CHECK-BE: xvrspic 7, 27                      # encoding: [0xf0,0xe0,0xda,0xac]
# CHECK-LE: xvrspic 7, 27                      # encoding: [0xac,0xda,0xe0,0xf0]
            xvrspic 7, 27
# CHECK-BE: xvrspim 7, 27                      # encoding: [0xf0,0xe0,0xda,0xe4]
# CHECK-LE: xvrspim 7, 27                      # encoding: [0xe4,0xda,0xe0,0xf0]
            xvrspim 7, 27
# CHECK-BE: xvrspip 7, 27                      # encoding: [0xf0,0xe0,0xda,0xa4]
# CHECK-LE: xvrspip 7, 27                      # encoding: [0xa4,0xda,0xe0,0xf0]
            xvrspip 7, 27
# CHECK-BE: xvrspiz 7, 27                      # encoding: [0xf0,0xe0,0xda,0x64]
# CHECK-LE: xvrspiz 7, 27                      # encoding: [0x64,0xda,0xe0,0xf0]
            xvrspiz 7, 27
# CHECK-BE: xvrsqrtedp 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x28]
# CHECK-LE: xvrsqrtedp 7, 27                   # encoding: [0x28,0xdb,0xe0,0xf0]
            xvrsqrtedp 7, 27
# CHECK-BE: xvrsqrtesp 7, 27                   # encoding: [0xf0,0xe0,0xda,0x28]
# CHECK-LE: xvrsqrtesp 7, 27                   # encoding: [0x28,0xda,0xe0,0xf0]
            xvrsqrtesp 7, 27
# CHECK-BE: xvsqrtdp 7, 27                     # encoding: [0xf0,0xe0,0xdb,0x2c]
# CHECK-LE: xvsqrtdp 7, 27                     # encoding: [0x2c,0xdb,0xe0,0xf0]
            xvsqrtdp 7, 27
# CHECK-BE: xvsqrtsp 7, 27                     # encoding: [0xf0,0xe0,0xda,0x2c]
# CHECK-LE: xvsqrtsp 7, 27                     # encoding: [0x2c,0xda,0xe0,0xf0]
            xvsqrtsp 7, 27
# CHECK-BE: xvsubdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x44]
# CHECK-LE: xvsubdp 7, 63, 27                  # encoding: [0x44,0xdb,0xff,0xf0]
            xvsubdp 7, 63, 27
# CHECK-BE: xvsubsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x44]
# CHECK-LE: xvsubsp 7, 63, 27                  # encoding: [0x44,0xda,0xff,0xf0]
            xvsubsp 7, 63, 27
# CHECK-BE: xvtdivdp 6, 63, 27                 # encoding: [0xf3,0x1f,0xdb,0xec]
# CHECK-LE: xvtdivdp 6, 63, 27                 # encoding: [0xec,0xdb,0x1f,0xf3]
            xvtdivdp 6, 63, 27
# CHECK-BE: xvtdivsp 6, 63, 27                 # encoding: [0xf3,0x1f,0xda,0xec]
# CHECK-LE: xvtdivsp 6, 63, 27                 # encoding: [0xec,0xda,0x1f,0xf3]
            xvtdivsp 6, 63, 27
# CHECK-BE: xvtsqrtdp 6, 27                    # encoding: [0xf3,0x00,0xdb,0xa8]
# CHECK-LE: xvtsqrtdp 6, 27                    # encoding: [0xa8,0xdb,0x00,0xf3]
            xvtsqrtdp 6, 27
# CHECK-BE: xvtsqrtsp 6, 27                    # encoding: [0xf3,0x00,0xda,0xa8]
# CHECK-LE: xvtsqrtsp 6, 27                    # encoding: [0xa8,0xda,0x00,0xf3]
            xvtsqrtsp 6, 27
# CHECK-BE: xxland 7, 63, 27                   # encoding: [0xf0,0xff,0xdc,0x14]
# CHECK-LE: xxland 7, 63, 27                   # encoding: [0x14,0xdc,0xff,0xf0]
            xxland 7, 63, 27
# CHECK-BE: xxlandc 7, 63, 27                  # encoding: [0xf0,0xff,0xdc,0x54]
# CHECK-LE: xxlandc 7, 63, 27                  # encoding: [0x54,0xdc,0xff,0xf0]
            xxlandc 7, 63, 27
# CHECK-BE: xxlnor 7, 63, 27                   # encoding: [0xf0,0xff,0xdd,0x14]
# CHECK-LE: xxlnor 7, 63, 27                   # encoding: [0x14,0xdd,0xff,0xf0]
            xxlnor 7, 63, 27
# CHECK-BE: xxlor 7, 63, 27                    # encoding: [0xf0,0xff,0xdc,0x94]
# CHECK-LE: xxlor 7, 63, 27                    # encoding: [0x94,0xdc,0xff,0xf0]
            xxlor 7, 63, 27
# CHECK-BE: xxlxor 7, 63, 27                   # encoding: [0xf0,0xff,0xdc,0xd4]
# CHECK-LE: xxlxor 7, 63, 27                   # encoding: [0xd4,0xdc,0xff,0xf0]
            xxlxor 7, 63, 27
# CHECK-BE: xxpermdi 7, 63, 27, 0              # encoding: [0xf0,0xff,0xd8,0x54]
# CHECK-LE: xxpermdi 7, 63, 27, 0              # encoding: [0x54,0xd8,0xff,0xf0]
            xxmrghd 7, 63, 27
# CHECK-BE: xxmrghw 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x94]
# CHECK-LE: xxmrghw 7, 63, 27                  # encoding: [0x94,0xd8,0xff,0xf0]
            xxmrghw 7, 63, 27
# CHECK-BE: xxpermdi 7, 63, 27, 3              # encoding: [0xf0,0xff,0xdb,0x54]
# CHECK-LE: xxpermdi 7, 63, 27, 3              # encoding: [0x54,0xdb,0xff,0xf0]
            xxmrgld 7, 63, 27
# CHECK-BE: xxmrglw 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x94]
# CHECK-LE: xxmrglw 7, 63, 27                  # encoding: [0x94,0xd9,0xff,0xf0]
            xxmrglw 7, 63, 27
# CHECK-BE: xxpermdi 7, 63, 27, 2              # encoding: [0xf0,0xff,0xda,0x54]
# CHECK-LE: xxpermdi 7, 63, 27, 2              # encoding: [0x54,0xda,0xff,0xf0]
            xxpermdi 7, 63, 27, 2
# CHECK-BE: xxsel 7, 63, 27, 14                # encoding: [0xf0,0xff,0xdb,0xb4]
# CHECK-LE: xxsel 7, 63, 27, 14                # encoding: [0xb4,0xdb,0xff,0xf0]
            xxsel 7, 63, 27, 14
# CHECK-BE: xxsldwi 7, 63, 27, 1               # encoding: [0xf0,0xff,0xd9,0x14]
# CHECK-LE: xxsldwi 7, 63, 27, 1               # encoding: [0x14,0xd9,0xff,0xf0]
            xxsldwi 7, 63, 27, 1
# CHECK-BE: xxpermdi 7, 63, 63, 3              # encoding: [0xf0,0xff,0xfb,0x56]
# CHECK-LE: xxpermdi 7, 63, 63, 3              # encoding: [0x56,0xfb,0xff,0xf0]
            xxspltd 7, 63, 1
# CHECK-BE: xxspltw 7, 27, 3                   # encoding: [0xf0,0xe3,0xda,0x90]
# CHECK-LE: xxspltw 7, 27, 3                   # encoding: [0x90,0xda,0xe3,0xf0]
            xxspltw 7, 27, 3
# CHECK-BE: xxpermdi 7, 63, 63, 2              # encoding: [0xf0,0xff,0xfa,0x56]
# CHECK-LE: xxpermdi 7, 63, 63, 2              # encoding: [0x56,0xfa,0xff,0xf0]
            xxswapd 7, 63
