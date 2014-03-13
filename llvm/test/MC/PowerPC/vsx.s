# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | FileCheck %s

# CHECK: lxsdx 7, 5, 31                     # encoding: [0x7c,0xe5,0xfc,0x98]
         lxsdx 7, 5, 31
# CHECK: lxvd2x 7, 5, 31                    # encoding: [0x7c,0xe5,0xfe,0x98]
         lxvd2x 7, 5, 31
# CHECK: lxvdsx 7, 5, 31                    # encoding: [0x7c,0xe5,0xfa,0x98]
         lxvdsx 7, 5, 31
# CHECK: lxvw4x 7, 5, 31                    # encoding: [0x7c,0xe5,0xfe,0x18]
         lxvw4x 7, 5, 31
# CHECK: stxsdx 8, 5, 31                    # encoding: [0x7d,0x05,0xfd,0x98]
         stxsdx 8, 5, 31
# CHECK: stxvd2x 8, 5, 31                   # encoding: [0x7d,0x05,0xff,0x98]
         stxvd2x 8, 5, 31
# CHECK: stxvw4x 8, 5, 31                   # encoding: [0x7d,0x05,0xff,0x18]
         stxvw4x 8, 5, 31
# CHECK: xsabsdp 7, 27                      # encoding: [0xf0,0xe0,0xdd,0x64]
         xsabsdp 7, 27
# CHECK: xsadddp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x04]
         xsadddp 7, 63, 27
# CHECK: xscmpodp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0x5c]
         xscmpodp 6, 63, 27
# CHECK: xscmpudp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0x1c]
         xscmpudp 6, 63, 27
# CHECK: xscpsgndp 7, 63, 27                # encoding: [0xf0,0xff,0xdd,0x84]
         xscpsgndp 7, 63, 27
# CHECK: xscvdpsp 7, 27                     # encoding: [0xf0,0xe0,0xdc,0x24]
         xscvdpsp 7, 27
# CHECK: xscvdpsxds 7, 27                   # encoding: [0xf0,0xe0,0xdd,0x60]
         xscvdpsxds 7, 27
# CHECK: xscvdpsxws 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x60]
         xscvdpsxws 7, 27
# CHECK: xscvdpuxds 7, 27                   # encoding: [0xf0,0xe0,0xdd,0x20]
         xscvdpuxds 7, 27
# CHECK: xscvdpuxws 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x20]
         xscvdpuxws 7, 27
# CHECK: xscvspdp 7, 27                     # encoding: [0xf0,0xe0,0xdd,0x24]
         xscvspdp 7, 27
# CHECK: xscvsxddp 7, 27                    # encoding: [0xf0,0xe0,0xdd,0xe0]
         xscvsxddp 7, 27
# CHECK: xscvuxddp 7, 27                    # encoding: [0xf0,0xe0,0xdd,0xa0]
         xscvuxddp 7, 27
# CHECK: xsdivdp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0xc4]
         xsdivdp 7, 63, 27
# CHECK: xsmaddadp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x0c]
         xsmaddadp 7, 63, 27
# CHECK: xsmaddmdp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x4c]
         xsmaddmdp 7, 63, 27
# CHECK: xsmaxdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x04]
         xsmaxdp 7, 63, 27
# CHECK: xsmindp 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x44]
         xsmindp 7, 63, 27
# CHECK: xsmsubadp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x8c]
         xsmsubadp 7, 63, 27
# CHECK: xsmsubmdp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0xcc]
         xsmsubmdp 7, 63, 27
# CHECK: xsmuldp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x84]
         xsmuldp 7, 63, 27
# CHECK: xsnabsdp 7, 27                     # encoding: [0xf0,0xe0,0xdd,0xa4]
         xsnabsdp 7, 27
# CHECK: xsnegdp 7, 27                      # encoding: [0xf0,0xe0,0xdd,0xe4]
         xsnegdp 7, 27
# CHECK: xsnmaddadp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x0c]
         xsnmaddadp 7, 63, 27
# CHECK: xsnmaddmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x4c]
         xsnmaddmdp 7, 63, 27
# CHECK: xsnmsubadp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x8c]
         xsnmsubadp 7, 63, 27
# CHECK: xsnmsubmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0xcc]
         xsnmsubmdp 7, 63, 27
# CHECK: xsrdpi 7, 27                       # encoding: [0xf0,0xe0,0xd9,0x24]
         xsrdpi 7, 27
# CHECK: xsrdpic 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xac]
         xsrdpic 7, 27
# CHECK: xsrdpim 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xe4]
         xsrdpim 7, 27
# CHECK: xsrdpip 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xa4]
         xsrdpip 7, 27
# CHECK: xsrdpiz 7, 27                      # encoding: [0xf0,0xe0,0xd9,0x64]
         xsrdpiz 7, 27
# CHECK: xsredp 7, 27                       # encoding: [0xf0,0xe0,0xd9,0x68]
         xsredp 7, 27
# CHECK: xsrsqrtedp 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x28]
         xsrsqrtedp 7, 27
# CHECK: xssqrtdp 7, 27                     # encoding: [0xf0,0xe0,0xd9,0x2c]
         xssqrtdp 7, 27
# CHECK: xssubdp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x44]
         xssubdp 7, 63, 27
# CHECK: xstdivdp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0xec]
         xstdivdp 6, 63, 27
# CHECK: xstsqrtdp 6, 27                    # encoding: [0xf3,0x00,0xd9,0xa8]
         xstsqrtdp 6, 27
# CHECK: xvabsdp 7, 27                      # encoding: [0xf0,0xe0,0xdf,0x64]
         xvabsdp 7, 27
# CHECK: xvabssp 7, 27                      # encoding: [0xf0,0xe0,0xde,0x64]
         xvabssp 7, 27
# CHECK: xvadddp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x04]
         xvadddp 7, 63, 27
# CHECK: xvaddsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x04]
         xvaddsp 7, 63, 27
# CHECK: xvcmpeqdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x1c]
         xvcmpeqdp 7, 63, 27
# CHECK: xvcmpeqdp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x1c]
         xvcmpeqdp. 7, 63, 27
# CHECK: xvcmpeqsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x1c]
         xvcmpeqsp 7, 63, 27
# CHECK: xvcmpeqsp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x1c]
         xvcmpeqsp. 7, 63, 27
# CHECK: xvcmpgedp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x9c]
         xvcmpgedp 7, 63, 27
# CHECK: xvcmpgedp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x9c]
         xvcmpgedp. 7, 63, 27
# CHECK: xvcmpgesp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x9c]
         xvcmpgesp 7, 63, 27
# CHECK: xvcmpgesp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x9c]
         xvcmpgesp. 7, 63, 27
# CHECK: xvcmpgtdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x5c]
         xvcmpgtdp 7, 63, 27
# CHECK: xvcmpgtdp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x5c]
         xvcmpgtdp. 7, 63, 27
# CHECK: xvcmpgtsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x5c]
         xvcmpgtsp 7, 63, 27
# CHECK: xvcmpgtsp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x5c]
         xvcmpgtsp. 7, 63, 27
# CHECK: xvcpsgndp 7, 63, 27                # encoding: [0xf0,0xff,0xdf,0x84]
         xvcpsgndp 7, 63, 27
# CHECK: xvcpsgnsp 7, 63, 27                # encoding: [0xf0,0xff,0xde,0x84]
         xvcpsgnsp 7, 63, 27
# CHECK: xvcvdpsp 7, 27                     # encoding: [0xf0,0xe0,0xde,0x24]
         xvcvdpsp 7, 27
# CHECK: xvcvdpsxds 7, 27                   # encoding: [0xf0,0xe0,0xdf,0x60]
         xvcvdpsxds 7, 27
# CHECK: xvcvdpsxws 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x60]
         xvcvdpsxws 7, 27
# CHECK: xvcvdpuxds 7, 27                   # encoding: [0xf0,0xe0,0xdf,0x20]
         xvcvdpuxds 7, 27
# CHECK: xvcvdpuxws 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x20]
         xvcvdpuxws 7, 27
# CHECK: xvcvspdp 7, 27                     # encoding: [0xf0,0xe0,0xdf,0x24]
         xvcvspdp 7, 27
# CHECK: xvcvspsxds 7, 27                   # encoding: [0xf0,0xe0,0xde,0x60]
         xvcvspsxds 7, 27
# CHECK: xvcvspsxws 7, 27                   # encoding: [0xf0,0xe0,0xda,0x60]
         xvcvspsxws 7, 27
# CHECK: xvcvspuxds 7, 27                   # encoding: [0xf0,0xe0,0xde,0x20]
         xvcvspuxds 7, 27
# CHECK: xvcvspuxws 7, 27                   # encoding: [0xf0,0xe0,0xda,0x20]
         xvcvspuxws 7, 27
# CHECK: xvcvsxddp 7, 27                    # encoding: [0xf0,0xe0,0xdf,0xe0]
         xvcvsxddp 7, 27
# CHECK: xvcvsxdsp 7, 27                    # encoding: [0xf0,0xe0,0xde,0xe0]
         xvcvsxdsp 7, 27
# CHECK: xvcvsxwdp 7, 27                    # encoding: [0xf0,0xe0,0xdb,0xe0]
         xvcvsxwdp 7, 27
# CHECK: xvcvsxwsp 7, 27                    # encoding: [0xf0,0xe0,0xda,0xe0]
         xvcvsxwsp 7, 27
# CHECK: xvcvuxddp 7, 27                    # encoding: [0xf0,0xe0,0xdf,0xa0]
         xvcvuxddp 7, 27
# CHECK: xvcvuxdsp 7, 27                    # encoding: [0xf0,0xe0,0xde,0xa0]
         xvcvuxdsp 7, 27
# CHECK: xvcvuxwdp 7, 27                    # encoding: [0xf0,0xe0,0xdb,0xa0]
         xvcvuxwdp 7, 27
# CHECK: xvcvuxwsp 7, 27                    # encoding: [0xf0,0xe0,0xda,0xa0]
         xvcvuxwsp 7, 27
# CHECK: xvdivdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0xc4]
         xvdivdp 7, 63, 27
# CHECK: xvdivsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0xc4]
         xvdivsp 7, 63, 27
# CHECK: xvmaddadp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x0c]
         xvmaddadp 7, 63, 27
# CHECK: xvmaddasp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x0c]
         xvmaddasp 7, 63, 27
# CHECK: xvmaddmdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x4c]
         xvmaddmdp 7, 63, 27
# CHECK: xvmaddmsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x4c]
         xvmaddmsp 7, 63, 27
# CHECK: xvmaxdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdf,0x04]
         xvmaxdp 7, 63, 27
# CHECK: xvmaxsp 7, 63, 27                  # encoding: [0xf0,0xff,0xde,0x04]
         xvmaxsp 7, 63, 27
# CHECK: xvmindp 7, 63, 27                  # encoding: [0xf0,0xff,0xdf,0x44]
         xvmindp 7, 63, 27
# CHECK: xvminsp 7, 63, 27                  # encoding: [0xf0,0xff,0xde,0x44]
         xvminsp 7, 63, 27
# CHECK: xvcpsgndp 7, 63, 63                # encoding: [0xf0,0xff,0xff,0x86]
         xvmovdp 7, 63
# CHECK: xvcpsgnsp 7, 63, 63                # encoding: [0xf0,0xff,0xfe,0x86]
         xvmovsp 7, 63
# CHECK: xvmsubadp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x8c]
         xvmsubadp 7, 63, 27
# CHECK: xvmsubasp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x8c]
         xvmsubasp 7, 63, 27
# CHECK: xvmsubmdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0xcc]
         xvmsubmdp 7, 63, 27
# CHECK: xvmsubmsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0xcc]
         xvmsubmsp 7, 63, 27
# CHECK: xvmuldp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x84]
         xvmuldp 7, 63, 27
# CHECK: xvmulsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x84]
         xvmulsp 7, 63, 27
# CHECK: xvnabsdp 7, 27                     # encoding: [0xf0,0xe0,0xdf,0xa4]
         xvnabsdp 7, 27
# CHECK: xvnabssp 7, 27                     # encoding: [0xf0,0xe0,0xde,0xa4]
         xvnabssp 7, 27
# CHECK: xvnegdp 7, 27                      # encoding: [0xf0,0xe0,0xdf,0xe4]
         xvnegdp 7, 27
# CHECK: xvnegsp 7, 27                      # encoding: [0xf0,0xe0,0xde,0xe4]
         xvnegsp 7, 27
# CHECK: xvnmaddadp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x0c]
         xvnmaddadp 7, 63, 27
# CHECK: xvnmaddasp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x0c]
         xvnmaddasp 7, 63, 27
# CHECK: xvnmaddmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x4c]
         xvnmaddmdp 7, 63, 27
# CHECK: xvnmaddmsp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x4c]
         xvnmaddmsp 7, 63, 27
# CHECK: xvnmsubadp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x8c]
         xvnmsubadp 7, 63, 27
# CHECK: xvnmsubasp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x8c]
         xvnmsubasp 7, 63, 27
# CHECK: xvnmsubmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0xcc]
         xvnmsubmdp 7, 63, 27
# CHECK: xvnmsubmsp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0xcc]
         xvnmsubmsp 7, 63, 27
# CHECK: xvrdpi 7, 27                       # encoding: [0xf0,0xe0,0xdb,0x24]
         xvrdpi 7, 27
# CHECK: xvrdpic 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xac]
         xvrdpic 7, 27
# CHECK: xvrdpim 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xe4]
         xvrdpim 7, 27
# CHECK: xvrdpip 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xa4]
         xvrdpip 7, 27
# CHECK: xvrdpiz 7, 27                      # encoding: [0xf0,0xe0,0xdb,0x64]
         xvrdpiz 7, 27
# CHECK: xvredp 7, 27                       # encoding: [0xf0,0xe0,0xdb,0x68]
         xvredp 7, 27
# CHECK: xvresp 7, 27                       # encoding: [0xf0,0xe0,0xda,0x68]
         xvresp 7, 27
# CHECK: xvrspi 7, 27                       # encoding: [0xf0,0xe0,0xda,0x24]
         xvrspi 7, 27
# CHECK: xvrspic 7, 27                      # encoding: [0xf0,0xe0,0xda,0xac]
         xvrspic 7, 27
# CHECK: xvrspim 7, 27                      # encoding: [0xf0,0xe0,0xda,0xe4]
         xvrspim 7, 27
# CHECK: xvrspip 7, 27                      # encoding: [0xf0,0xe0,0xda,0xa4]
         xvrspip 7, 27
# CHECK: xvrspiz 7, 27                      # encoding: [0xf0,0xe0,0xda,0x64]
         xvrspiz 7, 27
# CHECK: xvrsqrtedp 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x28]
         xvrsqrtedp 7, 27
# CHECK: xvrsqrtesp 7, 27                   # encoding: [0xf0,0xe0,0xda,0x28]
         xvrsqrtesp 7, 27
# CHECK: xvsqrtdp 7, 27                     # encoding: [0xf0,0xe0,0xdb,0x2c]
         xvsqrtdp 7, 27
# CHECK: xvsqrtsp 7, 27                     # encoding: [0xf0,0xe0,0xda,0x2c]
         xvsqrtsp 7, 27
# CHECK: xvsubdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x44]
         xvsubdp 7, 63, 27
# CHECK: xvsubsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x44]
         xvsubsp 7, 63, 27
# CHECK: xvtdivdp 6, 63, 27                 # encoding: [0xf3,0x1f,0xdb,0xec]
         xvtdivdp 6, 63, 27
# CHECK: xvtdivsp 6, 63, 27                 # encoding: [0xf3,0x1f,0xda,0xec]
         xvtdivsp 6, 63, 27
# CHECK: xvtsqrtdp 6, 27                    # encoding: [0xf3,0x00,0xdb,0xa8]
         xvtsqrtdp 6, 27
# CHECK: xvtsqrtsp 6, 27                    # encoding: [0xf3,0x00,0xda,0xa8]
         xvtsqrtsp 6, 27
# CHECK: xxland 7, 63, 27                   # encoding: [0xf0,0xff,0xdc,0x14]
         xxland 7, 63, 27
# CHECK: xxlandc 7, 63, 27                  # encoding: [0xf0,0xff,0xdc,0x54]
         xxlandc 7, 63, 27
# CHECK: xxlnor 7, 63, 27                   # encoding: [0xf0,0xff,0xdd,0x14]
         xxlnor 7, 63, 27
# CHECK: xxlor 7, 63, 27                    # encoding: [0xf0,0xff,0xdc,0x94]
         xxlor 7, 63, 27
# CHECK: xxlxor 7, 63, 27                   # encoding: [0xf0,0xff,0xdc,0xd4]
         xxlxor 7, 63, 27
# CHECK: xxpermdi 7, 63, 27, 0              # encoding: [0xf0,0xff,0xd8,0x54]
         xxmrghd 7, 63, 27
# CHECK: xxmrghw 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x94]
         xxmrghw 7, 63, 27
# CHECK: xxpermdi 7, 63, 27, 3              # encoding: [0xf0,0xff,0xdb,0x54]
         xxmrgld 7, 63, 27
# CHECK: xxmrglw 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x94]
         xxmrglw 7, 63, 27
# CHECK: xxpermdi 7, 63, 27, 2              # encoding: [0xf0,0xff,0xda,0x54]
         xxpermdi 7, 63, 27, 2
# CHECK: xxsel 7, 63, 27, 14                # encoding: [0xf0,0xff,0xdb,0xb4]
         xxsel 7, 63, 27, 14
# CHECK: xxsldwi 7, 63, 27, 1               # encoding: [0xf0,0xff,0xd9,0x14]
         xxsldwi 7, 63, 27, 1
# CHECK: xxpermdi 7, 63, 63, 3              # encoding: [0xf0,0xff,0xfb,0x56]
         xxspltd 7, 63, 1
# CHECK: xxspltw 7, 27, 3                   # encoding: [0xf0,0xe3,0xda,0x90]
         xxspltw 7, 27, 3
# CHECK: xxpermdi 7, 63, 63, 2              # encoding: [0xf0,0xff,0xfa,0x56]
         xxswapd 7, 63
