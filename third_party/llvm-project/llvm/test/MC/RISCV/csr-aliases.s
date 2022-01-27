# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=+f -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-f < %s \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-f < %s \
# RUN:     | llvm-objdump -d --mattr=-f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F-OFF %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=-f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F-OFF %s

# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=+f -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=-f < %s \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=-f < %s \
# RUN:     | llvm-objdump -d --mattr=-f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F-OFF %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d --mattr=-f - \
# RUN:     | FileCheck -check-prefix=CHECK-EXT-F-OFF %s


# CHECK-INST: csrrs t0, fcsr, zero
# CHECK-ALIAS: frcsr t0
# CHECK-EXT-F:  frcsr t0
# CHECK-EXT-F-OFF: csrr t0, fcsr
csrrs t0, 3, zero

# CHECK-INST: csrrw t1, fcsr, t2
# CHECK-ALIAS: fscsr t1, t2
# CHECK-EXT-F-ON: fscsr t1, t2
# CHECK-EXT-F-OFF: csrrw t1, fcsr, t2
csrrw t1, 3, t2

# CHECK-INST: csrrw zero, fcsr, t2
# CHECK-ALIAS: fscsr t2
# CHECK-EXT-F-ON: fscsr t2
# CHECK-EXT-F-OFF: csrw fcsr, t2
csrrw zero, 3, t2

# CHECK-INST: csrrw zero, fcsr, t2
# CHECK-ALIAS: fscsr t2
# CHECK-EXT-F-ON: fscsr t2
# CHECK-EXT-F-OFF: csrw fcsr, t2
csrrw zero, 3, t2

# CHECK-INST: csrrw t0, frm, zero
# CHECK-ALIAS: fsrm  t0, zero
# CHECK-EXT-F-ON: fsrm t0, zero
# CHECK-EXT-F-OFF: csrrw t0, frm
csrrw t0, 2, zero

# CHECK-INST: csrrw t0, frm, t1
# CHECK-ALIAS: fsrm t0, t1
# CHECK-EXT-F-ON: fsrm t0, t1
# CHECK-EXT-F-OFF: csrrw t0, frm, t1
csrrw t0, 2, t1

# CHECK-INST: csrrwi t0, frm, 31
# CHECK-ALIAS: fsrmi t0, 31
# CHECK-EXT-F-ON: fsrmi t0, 31
# CHECK-EXT-F-OFF: csrrwi t0, frm, 31
csrrwi t0, 2, 31

# CHECK-INST: csrrwi zero, frm, 31
# CHECK-ALIAS: fsrmi 31
# CHECK-EXT-F-ON: fsrmi 31
# CHECK-EXT-F-OFF:  csrwi frm, 31
csrrwi zero, 2, 31

# CHECK-INST: csrrs t0, fflags, zero
# CHECK-ALIAS: frflags t0
# CHECK-EXT-F-ON: frflags t0
# CHECK-EXT-F-OFF: csrr t0, fflags
csrrs t0, 1, zero

# CHECK-INST: csrrw t0, fflags, t2
# CHECK-ALIAS: fsflags t0, t2
# CHECK-EXT-F-ON: fsflags t0, t2
# CHECK-EXT-F-OFF: csrrw t0, fflags, t2
csrrw t0, 1, t2

# CHECK-INST: csrrw zero, fflags, t2
# CHECK-ALIAS: fsflags t2
# CHECK-EXT-F-ON: fsflags t2
# CHECK-EXT-F-OFF: csrw fflags, t2
csrrw zero, 1, t2

# CHECK-INST: csrrwi t0, fflags, 31
# CHECK-ALIAS: fsflagsi t0, 31
# CHECK-EXT-F: fsflagsi t0, 31
# CHECK-EXT-F-OFF: csrrwi t0, fflags, 31
csrrwi t0, 1, 31

# CHECK-INST: csrrwi zero, fflags, 31
# CHECK-ALIAS: fsflagsi 31
# CHECK-EXT-F: fsflagsi 31
# CHECK-EXT-F-OFF: csrwi fflags, 31
csrrwi zero, 1, 31

