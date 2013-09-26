
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# Floating-point facility

# Floating-point load instructions

# CHECK: lfs 2, 128(4)                   # encoding: [0xc0,0x44,0x00,0x80]
         lfs 2, 128(4)
# CHECK: lfsx 2, 3, 4                    # encoding: [0x7c,0x43,0x24,0x2e]
         lfsx 2, 3, 4
# CHECK: lfsu 2, 128(4)                  # encoding: [0xc4,0x44,0x00,0x80]
         lfsu 2, 128(4)
# CHECK: lfsux 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0x6e]
         lfsux 2, 3, 4
# CHECK: lfd 2, 128(4)                   # encoding: [0xc8,0x44,0x00,0x80]
         lfd 2, 128(4)
# CHECK: lfdx 2, 3, 4                    # encoding: [0x7c,0x43,0x24,0xae]
         lfdx 2, 3, 4
# CHECK: lfdu 2, 128(4)                  # encoding: [0xcc,0x44,0x00,0x80]
         lfdu 2, 128(4)
# CHECK: lfdux 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0xee]
         lfdux 2, 3, 4
# CHECK: lfiwax 2, 3, 4                  # encoding: [0x7c,0x43,0x26,0xae]
         lfiwax 2, 3, 4
# CHECK: lfiwzx 2, 3, 4                  # encoding: [0x7c,0x43,0x26,0xee]
         lfiwzx 2, 3, 4

# Floating-point store instructions

# CHECK: stfs 2, 128(4)                  # encoding: [0xd0,0x44,0x00,0x80]
         stfs 2, 128(4)
# CHECK: stfsx 2, 3, 4                   # encoding: [0x7c,0x43,0x25,0x2e]
         stfsx 2, 3, 4
# CHECK: stfsu 2, 128(4)                 # encoding: [0xd4,0x44,0x00,0x80]
         stfsu 2, 128(4)
# CHECK: stfsux 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0x6e]
         stfsux 2, 3, 4
# CHECK: stfd 2, 128(4)                  # encoding: [0xd8,0x44,0x00,0x80]
         stfd 2, 128(4)
# CHECK: stfdx 2, 3, 4                   # encoding: [0x7c,0x43,0x25,0xae]
         stfdx 2, 3, 4
# CHECK: stfdu 2, 128(4)                 # encoding: [0xdc,0x44,0x00,0x80]
         stfdu 2, 128(4)
# CHECK: stfdux 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0xee]
         stfdux 2, 3, 4
# CHECK: stfiwx 2, 3, 4                  # encoding: [0x7c,0x43,0x27,0xae]
         stfiwx 2, 3, 4

# Floating-point move instructions

# CHECK: fmr 2, 3                        # encoding: [0xfc,0x40,0x18,0x90]
         fmr 2, 3
# CHECK: fmr. 2, 3                       # encoding: [0xfc,0x40,0x18,0x91]
         fmr. 2, 3
# CHECK: fneg 2, 3                       # encoding: [0xfc,0x40,0x18,0x50]
         fneg 2, 3
# CHECK: fneg. 2, 3                      # encoding: [0xfc,0x40,0x18,0x51]
         fneg. 2, 3
# CHECK: fabs 2, 3                       # encoding: [0xfc,0x40,0x1a,0x10]
         fabs 2, 3
# CHECK: fabs. 2, 3                      # encoding: [0xfc,0x40,0x1a,0x11]
         fabs. 2, 3
# CHECK: fnabs 2, 3                      # encoding: [0xfc,0x40,0x19,0x10]
         fnabs 2, 3
# CHECK: fnabs. 2, 3                     # encoding: [0xfc,0x40,0x19,0x11]
         fnabs. 2, 3
# CHECK: fcpsgn 2, 3, 4                  # encoding: [0xfc,0x43,0x20,0x10]
         fcpsgn 2, 3, 4
# CHECK: fcpsgn. 2, 3, 4                 # encoding: [0xfc,0x43,0x20,0x11]
         fcpsgn. 2, 3, 4

# Floating-point arithmetic instructions

# CHECK: fadd 2, 3, 4                    # encoding: [0xfc,0x43,0x20,0x2a]
         fadd 2, 3, 4
# CHECK: fadd. 2, 3, 4                   # encoding: [0xfc,0x43,0x20,0x2b]
         fadd. 2, 3, 4
# CHECK: fadds 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x2a]
         fadds 2, 3, 4
# CHECK: fadds. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x2b]
         fadds. 2, 3, 4
# CHECK: fsub 2, 3, 4                    # encoding: [0xfc,0x43,0x20,0x28]
         fsub 2, 3, 4
# CHECK: fsub. 2, 3, 4                   # encoding: [0xfc,0x43,0x20,0x29]
         fsub. 2, 3, 4
# CHECK: fsubs 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x28]
         fsubs 2, 3, 4
# CHECK: fsubs. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x29]
         fsubs. 2, 3, 4

# CHECK: fmul 2, 3, 4                    # encoding: [0xfc,0x43,0x01,0x32]
         fmul 2, 3, 4
# CHECK: fmul. 2, 3, 4                   # encoding: [0xfc,0x43,0x01,0x33]
         fmul. 2, 3, 4
# CHECK: fmuls 2, 3, 4                   # encoding: [0xec,0x43,0x01,0x32]
         fmuls 2, 3, 4
# CHECK: fmuls. 2, 3, 4                  # encoding: [0xec,0x43,0x01,0x33]
         fmuls. 2, 3, 4
# CHECK: fdiv 2, 3, 4                    # encoding: [0xfc,0x43,0x20,0x24]
         fdiv 2, 3, 4
# CHECK: fdiv. 2, 3, 4                   # encoding: [0xfc,0x43,0x20,0x25]
         fdiv. 2, 3, 4
# CHECK: fdivs 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x24]
         fdivs 2, 3, 4
# CHECK: fdivs. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x25]
         fdivs. 2, 3, 4
# CHECK: fsqrt 2, 3                      # encoding: [0xfc,0x40,0x18,0x2c]
         fsqrt 2, 3
# CHECK: fsqrt. 2, 3                     # encoding: [0xfc,0x40,0x18,0x2d]
         fsqrt. 2, 3
# CHECK: fsqrts 2, 3                     # encoding: [0xec,0x40,0x18,0x2c]
         fsqrts 2, 3
# CHECK: fsqrts. 2, 3                    # encoding: [0xec,0x40,0x18,0x2d]
         fsqrts. 2, 3

# CHECK: fre 2, 3                        # encoding: [0xfc,0x40,0x18,0x30]
         fre 2, 3
# CHECK: fre. 2, 3                       # encoding: [0xfc,0x40,0x18,0x31]
         fre. 2, 3
# CHECK: fres 2, 3                       # encoding: [0xec,0x40,0x18,0x30]
         fres 2, 3
# CHECK: fres. 2, 3                      # encoding: [0xec,0x40,0x18,0x31]
         fres. 2, 3
# CHECK: frsqrte 2, 3                    # encoding: [0xfc,0x40,0x18,0x34]
         frsqrte 2, 3
# CHECK: frsqrte. 2, 3                   # encoding: [0xfc,0x40,0x18,0x35]
         frsqrte. 2, 3
# CHECK: frsqrtes 2, 3                   # encoding: [0xec,0x40,0x18,0x34]
         frsqrtes 2, 3
# CHECK: frsqrtes. 2, 3                  # encoding: [0xec,0x40,0x18,0x35]
         frsqrtes. 2, 3
# FIXME: ftdiv 2, 3, 4
# FIXME: ftsqrt 2, 3, 4

# CHECK: fmadd 2, 3, 4, 5                # encoding: [0xfc,0x43,0x29,0x3a]
         fmadd 2, 3, 4, 5
# CHECK: fmadd. 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x3b]
         fmadd. 2, 3, 4, 5
# CHECK: fmadds 2, 3, 4, 5               # encoding: [0xec,0x43,0x29,0x3a]
         fmadds 2, 3, 4, 5
# CHECK: fmadds. 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x3b]
         fmadds. 2, 3, 4, 5
# CHECK: fmsub 2, 3, 4, 5                # encoding: [0xfc,0x43,0x29,0x38]
         fmsub 2, 3, 4, 5
# CHECK: fmsub. 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x39]
         fmsub. 2, 3, 4, 5
# CHECK: fmsubs 2, 3, 4, 5               # encoding: [0xec,0x43,0x29,0x38]
         fmsubs 2, 3, 4, 5
# CHECK: fmsubs. 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x39]
         fmsubs. 2, 3, 4, 5
# CHECK: fnmadd 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x3e]
         fnmadd 2, 3, 4, 5
# CHECK: fnmadd. 2, 3, 4, 5              # encoding: [0xfc,0x43,0x29,0x3f]
         fnmadd. 2, 3, 4, 5
# CHECK: fnmadds 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x3e]
         fnmadds 2, 3, 4, 5
# CHECK: fnmadds. 2, 3, 4, 5             # encoding: [0xec,0x43,0x29,0x3f]
         fnmadds. 2, 3, 4, 5
# CHECK: fnmsub 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x3c]
         fnmsub 2, 3, 4, 5
# CHECK: fnmsub. 2, 3, 4, 5              # encoding: [0xfc,0x43,0x29,0x3d]
         fnmsub. 2, 3, 4, 5
# CHECK: fnmsubs 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x3c]
         fnmsubs 2, 3, 4, 5
# CHECK: fnmsubs. 2, 3, 4, 5             # encoding: [0xec,0x43,0x29,0x3d]
         fnmsubs. 2, 3, 4, 5

# Floating-point rounding and conversion instructions

# CHECK: frsp 2, 3                       # encoding: [0xfc,0x40,0x18,0x18]
         frsp 2, 3
# CHECK: frsp. 2, 3                      # encoding: [0xfc,0x40,0x18,0x19]
         frsp. 2, 3

# CHECK: fctid 2, 3                      # encoding: [0xfc,0x40,0x1e,0x5c]
         fctid 2, 3
# CHECK: fctid. 2, 3                     # encoding: [0xfc,0x40,0x1e,0x5d]
         fctid. 2, 3
# CHECK: fctidz 2, 3                     # encoding: [0xfc,0x40,0x1e,0x5e]
         fctidz 2, 3
# CHECK: fctidz. 2, 3                    # encoding: [0xfc,0x40,0x1e,0x5f]
         fctidz. 2, 3
# FIXME: fctidu 2, 3
# FIXME: fctidu. 2, 3
# CHECK: fctiduz 2, 3                    # encoding: [0xfc,0x40,0x1f,0x5e]
         fctiduz 2, 3
# CHECK: fctiduz. 2, 3                   # encoding: [0xfc,0x40,0x1f,0x5f]
         fctiduz. 2, 3
# CHECK: fctiw 2, 3                      # encoding: [0xfc,0x40,0x18,0x1c]
         fctiw 2, 3
# CHECK: fctiw. 2, 3                     # encoding: [0xfc,0x40,0x18,0x1d]
         fctiw. 2, 3
# CHECK: fctiwz 2, 3                     # encoding: [0xfc,0x40,0x18,0x1e]
         fctiwz 2, 3
# CHECK: fctiwz. 2, 3                    # encoding: [0xfc,0x40,0x18,0x1f]
         fctiwz. 2, 3
# FIXME: fctiwu 2, 3
# FIXME: fctiwu. 2, 3
# CHECK: fctiwuz 2, 3                    # encoding: [0xfc,0x40,0x19,0x1e]
         fctiwuz 2, 3
# CHECK: fctiwuz. 2, 3                   # encoding: [0xfc,0x40,0x19,0x1f]
         fctiwuz. 2, 3
# CHECK: fcfid 2, 3                      # encoding: [0xfc,0x40,0x1e,0x9c]
         fcfid 2, 3
# CHECK: fcfid. 2, 3                     # encoding: [0xfc,0x40,0x1e,0x9d]
         fcfid. 2, 3
# CHECK: fcfidu 2, 3                     # encoding: [0xfc,0x40,0x1f,0x9c]
         fcfidu 2, 3
# CHECK: fcfidu. 2, 3                    # encoding: [0xfc,0x40,0x1f,0x9d]
         fcfidu. 2, 3
# CHECK: fcfids 2, 3                     # encoding: [0xec,0x40,0x1e,0x9c]
         fcfids 2, 3
# CHECK: fcfids. 2, 3                    # encoding: [0xec,0x40,0x1e,0x9d]
         fcfids. 2, 3
# CHECK: fcfidus 2, 3                    # encoding: [0xec,0x40,0x1f,0x9c]
         fcfidus 2, 3
# CHECK: fcfidus. 2, 3                   # encoding: [0xec,0x40,0x1f,0x9d]
         fcfidus. 2, 3
# CHECK: frin 2, 3                       # encoding: [0xfc,0x40,0x1b,0x10]
         frin 2, 3
# CHECK: frin. 2, 3                      # encoding: [0xfc,0x40,0x1b,0x11]
         frin. 2, 3
# CHECK: frip 2, 3                       # encoding: [0xfc,0x40,0x1b,0x90]
         frip 2, 3
# CHECK: frip. 2, 3                      # encoding: [0xfc,0x40,0x1b,0x91]
         frip. 2, 3
# CHECK: friz 2, 3                       # encoding: [0xfc,0x40,0x1b,0x50]
         friz 2, 3
# CHECK: friz. 2, 3                      # encoding: [0xfc,0x40,0x1b,0x51]
         friz. 2, 3
# CHECK: frim 2, 3                       # encoding: [0xfc,0x40,0x1b,0xd0]
         frim 2, 3
# CHECK: frim. 2, 3                      # encoding: [0xfc,0x40,0x1b,0xd1]
         frim. 2, 3

# Floating-point compare instructions

# CHECK: fcmpu 2, 3, 4                   # encoding: [0xfd,0x03,0x20,0x00]
         fcmpu 2, 3, 4
# FIXME: fcmpo 2, 3, 4

# Floating-point select instruction

# CHECK: fsel 2, 3, 4, 5                 # encoding: [0xfc,0x43,0x29,0x2e]
         fsel 2, 3, 4, 5
# CHECK: fsel. 2, 3, 4, 5                # encoding: [0xfc,0x43,0x29,0x2f]
         fsel. 2, 3, 4, 5

# Floating-point status and control register instructions

# CHECK: mffs 2                          # encoding: [0xfc,0x40,0x04,0x8e]
         mffs 2
# FIXME: mffs. 2

# FIXME: mcrfs 2, 3

# FIXME: mtfsfi 2, 3, 1
# FIXME: mtfsfi. 2, 3, 1
# FIXME: mtfsf 2, 3, 1, 1
# FIXME: mtfsf. 2, 3, 1, 1

# CHECK: mtfsb0 31                       # encoding: [0xff,0xe0,0x00,0x8c]
         mtfsb0 31
# FIXME: mtfsb0. 31
# CHECK: mtfsb1 31                       # encoding: [0xff,0xe0,0x00,0x4c]
         mtfsb1 31
# FIXME: mtfsb1. 31

