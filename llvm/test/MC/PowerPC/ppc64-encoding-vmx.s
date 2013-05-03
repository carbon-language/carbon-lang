
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# Vector facility

# Vector storage access instructions

# CHECK: lvebx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x0e]
         lvebx 2, 3, 4
# CHECK: lvehx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x4e]
         lvehx 2, 3, 4
# CHECK: lvewx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x8e]
         lvewx 2, 3, 4
# CHECK: lvx 2, 3, 4                     # encoding: [0x7c,0x43,0x20,0xce]
         lvx 2, 3, 4
# CHECK: lvxl 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0xce]
         lvxl 2, 3, 4
# CHECK: stvebx 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x0e]
         stvebx 2, 3, 4
# CHECK: stvehx 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x4e]
         stvehx 2, 3, 4
# CHECK: stvewx 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x8e]
         stvewx 2, 3, 4
# CHECK: stvx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0xce]
         stvx 2, 3, 4
# CHECK: stvxl 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0xce]
         stvxl 2, 3, 4
# CHECK: lvsl 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x0c]
         lvsl 2, 3, 4
# CHECK: lvsr 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x4c]
         lvsr 2, 3, 4

# Vector permute and formatting instructions

# CHECK: vpkpx 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x0e]
         vpkpx 2, 3, 4
# CHECK: vpkshss 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x8e]
         vpkshss 2, 3, 4
# CHECK: vpkshus 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x0e]
         vpkshus 2, 3, 4
# CHECK: vpkswss 2, 3, 4                 # encoding: [0x10,0x43,0x21,0xce]
         vpkswss 2, 3, 4
# CHECK: vpkswus 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x4e]
         vpkswus 2, 3, 4
# CHECK: vpkuhum 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x0e]
         vpkuhum 2, 3, 4
# CHECK: vpkuhus 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x8e]
         vpkuhus 2, 3, 4
# CHECK: vpkuwum 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x4e]
         vpkuwum 2, 3, 4
# CHECK: vpkuwus 2, 3, 4                 # encoding: [0x10,0x43,0x20,0xce]
         vpkuwus 2, 3, 4

# CHECK: vupkhpx 2, 3                    # encoding: [0x10,0x40,0x1b,0x4e]
         vupkhpx 2, 3
# CHECK: vupkhsb 2, 3                    # encoding: [0x10,0x40,0x1a,0x0e]
         vupkhsb 2, 3
# CHECK: vupkhsh 2, 3                    # encoding: [0x10,0x40,0x1a,0x4e]
         vupkhsh 2, 3
# CHECK: vupklpx 2, 3                    # encoding: [0x10,0x40,0x1b,0xce]
         vupklpx 2, 3
# CHECK: vupklsb 2, 3                    # encoding: [0x10,0x40,0x1a,0x8e]
         vupklsb 2, 3
# CHECK: vupklsh 2, 3                    # encoding: [0x10,0x40,0x1a,0xce]
         vupklsh 2, 3

# CHECK: vmrghb 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x0c]
         vmrghb 2, 3, 4
# CHECK: vmrghh 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x4c]
         vmrghh 2, 3, 4
# CHECK: vmrghw 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x8c]
         vmrghw 2, 3, 4
# CHECK: vmrglb 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x0c]
         vmrglb 2, 3, 4
# CHECK: vmrglh 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x4c]
         vmrglh 2, 3, 4
# CHECK: vmrglw 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x8c]
         vmrglw 2, 3, 4

# CHECK: vspltb 2, 3, 1                  # encoding: [0x10,0x41,0x1a,0x0c]
         vspltb 2, 3, 1
# CHECK: vsplth 2, 3, 1                  # encoding: [0x10,0x41,0x1a,0x4c]
         vsplth 2, 3, 1
# CHECK: vspltw 2, 3, 1                  # encoding: [0x10,0x41,0x1a,0x8c]
         vspltw 2, 3, 1
# CHECK: vspltisb 2, 3                   # encoding: [0x10,0x43,0x03,0x0c]
         vspltisb 2, 3
# CHECK: vspltish 2, 3                   # encoding: [0x10,0x43,0x03,0x4c]
         vspltish 2, 3
# CHECK: vspltisw 2, 3                   # encoding: [0x10,0x43,0x03,0x8c]
         vspltisw 2, 3

# CHECK: vperm 2, 3, 4, 5                # encoding: [0x10,0x43,0x21,0x6b]
         vperm 2, 3, 4, 5
# CHECK: vsel 2, 3, 4, 5                 # encoding: [0x10,0x43,0x21,0x6a]
         vsel 2, 3, 4, 5

# CHECK: vsl 2, 3, 4                     # encoding: [0x10,0x43,0x21,0xc4]
         vsl 2, 3, 4
# CHECK: vsldoi 2, 3, 4, 5               # encoding: [0x10,0x43,0x21,0x6c]
         vsldoi 2, 3, 4, 5
# CHECK: vslo 2, 3, 4                    # encoding: [0x10,0x43,0x24,0x0c]
         vslo 2, 3, 4
# CHECK: vsr 2, 3, 4                     # encoding: [0x10,0x43,0x22,0xc4]
         vsr 2, 3, 4
# CHECK: vsro 2, 3, 4                    # encoding: [0x10,0x43,0x24,0x4c]
         vsro 2, 3, 4

# Vector integer arithmetic instructions

# CHECK: vaddcuw 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x80]
         vaddcuw 2, 3, 4
# CHECK: vaddsbs 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x00]
         vaddsbs 2, 3, 4
# CHECK: vaddshs 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x40]
         vaddshs 2, 3, 4
# CHECK: vaddsws 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x80]
         vaddsws 2, 3, 4
# CHECK: vaddubm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x00]
         vaddubm 2, 3, 4
# CHECK: vadduhm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x40]
         vadduhm 2, 3, 4
# CHECK: vadduwm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x80]
         vadduwm 2, 3, 4
# CHECK: vaddubs 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x00]
         vaddubs 2, 3, 4
# CHECK: vadduhs 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x40]
         vadduhs 2, 3, 4
# CHECK: vadduws 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x80]
         vadduws 2, 3, 4

# CHECK: vsubcuw 2, 3, 4                 # encoding: [0x10,0x43,0x25,0x80]
         vsubcuw 2, 3, 4
# CHECK: vsubsbs 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x00]
         vsubsbs 2, 3, 4
# CHECK: vsubshs 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x40]
         vsubshs 2, 3, 4
# CHECK: vsubsws 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x80]
         vsubsws 2, 3, 4
# CHECK: vsububm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x00]
         vsububm 2, 3, 4
# CHECK: vsubuhm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x40]
         vsubuhm 2, 3, 4
# CHECK: vsubuwm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x80]
         vsubuwm 2, 3, 4
# CHECK: vsububs 2, 3, 4                 # encoding: [0x10,0x43,0x26,0x00]
         vsububs 2, 3, 4
# CHECK: vsubuhs 2, 3, 4                 # encoding: [0x10,0x43,0x26,0x40]
         vsubuhs 2, 3, 4
# CHECK: vsubuws 2, 3, 4                 # encoding: [0x10,0x43,0x26,0x80]
         vsubuws 2, 3, 4

# CHECK: vmulesb 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x08]
         vmulesb 2, 3, 4
# CHECK: vmulesh 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x48]
         vmulesh 2, 3, 4
# CHECK: vmuleub 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x08]
         vmuleub 2, 3, 4
# CHECK: vmuleuh 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x48]
         vmuleuh 2, 3, 4
# CHECK: vmulosb 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x08]
         vmulosb 2, 3, 4
# CHECK: vmulosh 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x48]
         vmulosh 2, 3, 4
# CHECK: vmuloub 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x08]
         vmuloub 2, 3, 4
# CHECK: vmulouh 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x48]
         vmulouh 2, 3, 4

# CHECK: vmhaddshs 2, 3, 4, 5            # encoding: [0x10,0x43,0x21,0x60]
         vmhaddshs 2, 3, 4, 5
# CHECK: vmhraddshs 2, 3, 4, 5           # encoding: [0x10,0x43,0x21,0x61]
         vmhraddshs 2, 3, 4, 5
# CHECK: vmladduhm 2, 3, 4, 5            # encoding: [0x10,0x43,0x21,0x62]
         vmladduhm 2, 3, 4, 5
# CHECK: vmsumubm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x64]
         vmsumubm 2, 3, 4, 5
# CHECK: vmsummbm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x65]
         vmsummbm 2, 3, 4, 5
# CHECK: vmsumshm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x68]
         vmsumshm 2, 3, 4, 5
# CHECK: vmsumshs 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x69]
         vmsumshs 2, 3, 4, 5
# CHECK: vmsumuhm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x66]
         vmsumuhm 2, 3, 4, 5
# CHECK: vmsumuhs 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x67]
         vmsumuhs 2, 3, 4, 5

# CHECK: vsumsws 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x88]
         vsumsws 2, 3, 4
# CHECK: vsum2sws 2, 3, 4                # encoding: [0x10,0x43,0x26,0x88]
         vsum2sws 2, 3, 4
# CHECK: vsum4sbs 2, 3, 4                # encoding: [0x10,0x43,0x27,0x08]
         vsum4sbs 2, 3, 4
# CHECK: vsum4shs 2, 3, 4                # encoding: [0x10,0x43,0x26,0x48]
         vsum4shs 2, 3, 4
# CHECK: vsum4ubs 2, 3, 4                # encoding: [0x10,0x43,0x26,0x08]
         vsum4ubs 2, 3, 4

# CHECK: vavgsb 2, 3, 4                  # encoding: [0x10,0x43,0x25,0x02]
         vavgsb 2, 3, 4
# CHECK: vavgsh 2, 3, 4                  # encoding: [0x10,0x43,0x25,0x42]
         vavgsh 2, 3, 4
# CHECK: vavgsw 2, 3, 4                  # encoding: [0x10,0x43,0x25,0x82]
         vavgsw 2, 3, 4
# CHECK: vavgub 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x02]
         vavgub 2, 3, 4
# CHECK: vavguh 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x42]
         vavguh 2, 3, 4
# CHECK: vavguw 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x82]
         vavguw 2, 3, 4

# CHECK: vmaxsb 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x02]
         vmaxsb 2, 3, 4
# CHECK: vmaxsh 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x42]
         vmaxsh 2, 3, 4
# CHECK: vmaxsw 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x82]
         vmaxsw 2, 3, 4
# CHECK: vmaxub 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x02]
         vmaxub 2, 3, 4
# CHECK: vmaxuh 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x42]
         vmaxuh 2, 3, 4
# CHECK: vmaxuw 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x82]
         vmaxuw 2, 3, 4

# CHECK: vminsb 2, 3, 4                  # encoding: [0x10,0x43,0x23,0x02]
         vminsb 2, 3, 4
# CHECK: vminsh 2, 3, 4                  # encoding: [0x10,0x43,0x23,0x42]
         vminsh 2, 3, 4
# CHECK: vminsw 2, 3, 4                  # encoding: [0x10,0x43,0x23,0x82]
         vminsw 2, 3, 4
# CHECK: vminub 2, 3, 4                  # encoding: [0x10,0x43,0x22,0x02]
         vminub 2, 3, 4
# CHECK: vminuh 2, 3, 4                  # encoding: [0x10,0x43,0x22,0x42]
         vminuh 2, 3, 4
# CHECK: vminuw 2, 3, 4                  # encoding: [0x10,0x43,0x22,0x82]
         vminuw 2, 3, 4

# Vector integer compare instructions

# CHECK: vcmpequb 2, 3, 4                # encoding: [0x10,0x43,0x20,0x06]
         vcmpequb 2, 3, 4
# CHECK: vcmpequb. 2, 3, 4               # encoding: [0x10,0x43,0x24,0x06]
         vcmpequb. 2, 3, 4
# CHECK: vcmpequh 2, 3, 4                # encoding: [0x10,0x43,0x20,0x46]
         vcmpequh 2, 3, 4
# CHECK: vcmpequh. 2, 3, 4               # encoding: [0x10,0x43,0x24,0x46]
         vcmpequh. 2, 3, 4
# CHECK: vcmpequw 2, 3, 4                # encoding: [0x10,0x43,0x20,0x86]
         vcmpequw 2, 3, 4
# CHECK: vcmpequw. 2, 3, 4               # encoding: [0x10,0x43,0x24,0x86]
         vcmpequw. 2, 3, 4
# CHECK: vcmpgtsb 2, 3, 4                # encoding: [0x10,0x43,0x23,0x06]
         vcmpgtsb 2, 3, 4
# CHECK: vcmpgtsb. 2, 3, 4               # encoding: [0x10,0x43,0x27,0x06]
         vcmpgtsb. 2, 3, 4
# CHECK: vcmpgtsh 2, 3, 4                # encoding: [0x10,0x43,0x23,0x46]
         vcmpgtsh 2, 3, 4
# CHECK: vcmpgtsh. 2, 3, 4               # encoding: [0x10,0x43,0x27,0x46]
         vcmpgtsh. 2, 3, 4
# CHECK: vcmpgtsw 2, 3, 4                # encoding: [0x10,0x43,0x23,0x86]
         vcmpgtsw 2, 3, 4
# CHECK: vcmpgtsw. 2, 3, 4               # encoding: [0x10,0x43,0x27,0x86]
         vcmpgtsw. 2, 3, 4
# CHECK: vcmpgtub 2, 3, 4                # encoding: [0x10,0x43,0x22,0x06]
         vcmpgtub 2, 3, 4
# CHECK: vcmpgtub. 2, 3, 4               # encoding: [0x10,0x43,0x26,0x06]
         vcmpgtub. 2, 3, 4
# CHECK: vcmpgtuh 2, 3, 4                # encoding: [0x10,0x43,0x22,0x46]
         vcmpgtuh 2, 3, 4
# CHECK: vcmpgtuh. 2, 3, 4               # encoding: [0x10,0x43,0x26,0x46]
         vcmpgtuh. 2, 3, 4
# CHECK: vcmpgtuw 2, 3, 4                # encoding: [0x10,0x43,0x22,0x86]
         vcmpgtuw 2, 3, 4
# CHECK: vcmpgtuw. 2, 3, 4               # encoding: [0x10,0x43,0x26,0x86]
         vcmpgtuw. 2, 3, 4

# Vector integer logical instructions

# CHECK: vand 2, 3, 4                    # encoding: [0x10,0x43,0x24,0x04]
         vand 2, 3, 4
# CHECK: vandc 2, 3, 4                   # encoding: [0x10,0x43,0x24,0x44]
         vandc 2, 3, 4
# CHECK: vnor 2, 3, 4                    # encoding: [0x10,0x43,0x25,0x04]
         vnor 2, 3, 4
# CHECK: vor 2, 3, 4                     # encoding: [0x10,0x43,0x24,0x84]
         vor 2, 3, 4
# CHECK: vxor 2, 3, 4                    # encoding: [0x10,0x43,0x24,0xc4]
         vxor 2, 3, 4

# Vector integer rotate and shift instructions

# CHECK: vrlb 2, 3, 4                    # encoding: [0x10,0x43,0x20,0x04]
         vrlb 2, 3, 4
# CHECK: vrlh 2, 3, 4                    # encoding: [0x10,0x43,0x20,0x44]
         vrlh 2, 3, 4
# CHECK: vrlw 2, 3, 4                    # encoding: [0x10,0x43,0x20,0x84]
         vrlw 2, 3, 4

# CHECK: vslb 2, 3, 4                    # encoding: [0x10,0x43,0x21,0x04]
         vslb 2, 3, 4
# CHECK: vslh 2, 3, 4                    # encoding: [0x10,0x43,0x21,0x44]
         vslh 2, 3, 4
# CHECK: vslw 2, 3, 4                    # encoding: [0x10,0x43,0x21,0x84]
         vslw 2, 3, 4
# CHECK: vsrb 2, 3, 4                    # encoding: [0x10,0x43,0x22,0x04]
         vsrb 2, 3, 4
# CHECK: vsrh 2, 3, 4                    # encoding: [0x10,0x43,0x22,0x44]
         vsrh 2, 3, 4
# CHECK: vsrw 2, 3, 4                    # encoding: [0x10,0x43,0x22,0x84]
         vsrw 2, 3, 4
# CHECK: vsrab 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x04]
         vsrab 2, 3, 4
# CHECK: vsrah 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x44]
         vsrah 2, 3, 4
# CHECK: vsraw 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x84]
         vsraw 2, 3, 4

# Vector floating-point instructions

# CHECK: vaddfp 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x0a]
         vaddfp 2, 3, 4
# CHECK: vsubfp 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x4a]
         vsubfp 2, 3, 4
# CHECK: vmaddfp 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x2e]
         vmaddfp 2, 3, 4, 5
# CHECK: vnmsubfp 2, 3, 4, 5             # encoding: [0x10,0x43,0x29,0x2f]
         vnmsubfp 2, 3, 4, 5

# CHECK: vmaxfp 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x0a]
         vmaxfp 2, 3, 4
# CHECK: vminfp 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x4a]
         vminfp 2, 3, 4

# CHECK: vctsxs 2, 3, 4                  # encoding: [0x10,0x44,0x1b,0xca]
         vctsxs 2, 3, 4
# CHECK: vctuxs 2, 3, 4                  # encoding: [0x10,0x44,0x1b,0x8a]
         vctuxs 2, 3, 4
# CHECK: vcfsx 2, 3, 4                   # encoding: [0x10,0x44,0x1b,0x4a]
         vcfsx 2, 3, 4
# CHECK: vcfux 2, 3, 4                   # encoding: [0x10,0x44,0x1b,0x0a]
         vcfux 2, 3, 4
# CHECK: vrfim 2, 3                      # encoding: [0x10,0x40,0x1a,0xca]
         vrfim 2, 3
# CHECK: vrfin 2, 3                      # encoding: [0x10,0x40,0x1a,0x0a]
         vrfin 2, 3
# CHECK: vrfip 2, 3                      # encoding: [0x10,0x40,0x1a,0x8a]
         vrfip 2, 3
# CHECK: vrfiz 2, 3                      # encoding: [0x10,0x40,0x1a,0x4a]
         vrfiz 2, 3

# CHECK: vcmpbfp 2, 3, 4                 # encoding: [0x10,0x43,0x23,0xc6]
         vcmpbfp 2, 3, 4
# CHECK: vcmpbfp. 2, 3, 4                # encoding: [0x10,0x43,0x27,0xc6]
         vcmpbfp. 2, 3, 4
# CHECK: vcmpeqfp 2, 3, 4                # encoding: [0x10,0x43,0x20,0xc6]
         vcmpeqfp 2, 3, 4
# CHECK: vcmpeqfp. 2, 3, 4               # encoding: [0x10,0x43,0x24,0xc6]
         vcmpeqfp. 2, 3, 4
# CHECK: vcmpgefp 2, 3, 4                # encoding: [0x10,0x43,0x21,0xc6]
         vcmpgefp 2, 3, 4
# CHECK: vcmpgefp. 2, 3, 4               # encoding: [0x10,0x43,0x25,0xc6]
         vcmpgefp. 2, 3, 4
# CHECK: vcmpgtfp 2, 3, 4                # encoding: [0x10,0x43,0x22,0xc6]
         vcmpgtfp 2, 3, 4
# CHECK: vcmpgtfp. 2, 3, 4               # encoding: [0x10,0x43,0x26,0xc6]
         vcmpgtfp. 2, 3, 4

# CHECK: vexptefp 2, 3                   # encoding: [0x10,0x40,0x19,0x8a]
         vexptefp 2, 3
# CHECK: vlogefp 2, 3                    # encoding: [0x10,0x40,0x19,0xca]
         vlogefp 2, 3
# CHECK: vrefp 2, 3                      # encoding: [0x10,0x40,0x19,0x0a]
         vrefp 2, 3
# CHECK: vrsqrtefp 2, 3                  # encoding: [0x10,0x40,0x19,0x4a]
         vrsqrtefp 2, 3

# Vector status and control register instructions

# CHECK: mtvscr 2                        # encoding: [0x10,0x00,0x16,0x44]
         mtvscr 2
# CHECK: mfvscr 2                        # encoding: [0x10,0x40,0x06,0x04]
         mfvscr 2

