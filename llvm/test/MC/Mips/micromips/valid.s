# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -show-inst -mattr=micromips,eva | FileCheck %s

.set noat
addiusp -16                 # CHECK: addiusp -16        # encoding: [0x4f,0xf9]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUSP_MM
addiusp -1028               # CHECK: addiusp -1028      # encoding: [0x4f,0xff]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUSP_MM
addiusp -1032               # CHECK: addiusp -1032      # encoding: [0x4f,0xfd]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUSP_MM
addiusp 1024                # CHECK: addiusp 1024       # encoding: [0x4c,0x01]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUSP_MM
addiusp 1028                # CHECK: addiusp 1028       # encoding: [0x4c,0x03]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUSP_MM
andi16 $16, $2, 31          # CHECK: andi16 $16, $2, 31 # encoding: [0x2c,0x29]
                            # CHECK:                    # <MCInst #{{.*}} ANDI16_MM
jraddiusp 20                # CHECK: jraddiusp 20       # encoding: [0x47,0x05]
                            # CHECK:                    # <MCInst #{{.*}} JRADDIUSP
addu16 $6, $17, $4          # CHECK: addu16 $6, $17, $4 # encoding: [0x07,0x42]
                            # CHECK:                    # <MCInst #{{.*}} ADDU16_MM
subu16 $5, $16, $3          # CHECK: subu16 $5, $16, $3 # encoding: [0x06,0xb1]
                            # CHECK:                    # <MCInst #{{.*}} SUBU16_MM
and16 $16, $2               # CHECK: and16 $16, $2      # encoding: [0x44,0x82]
                            # CHECK:                    # <MCInst #{{.*}} AND16_MM
not16 $17, $3               # CHECK: not16 $17, $3      # encoding: [0x44,0x0b]
                            # CHECK:                    # <MCInst #{{.*}} NOT16_MM
or16 $16, $4                # CHECK: or16 $16, $4       # encoding: [0x44,0xc4]
                            # CHECK:                    # <MCInst #{{.*}} OR16_MM
xor16 $17, $5               # CHECK: xor16 $17, $5      # encoding: [0x44,0x4d]
                            # CHECK:                    # <MCInst #{{.*}} XOR16_MM
sll16 $3, $16, 5            # CHECK: sll16 $3, $16, 5   # encoding: [0x25,0x8a]
                            # CHECK:                    # <MCInst #{{.*}} SLL16_MM
srl16 $4, $17, 6            # CHECK: srl16 $4, $17, 6   # encoding: [0x26,0x1d]
                            # CHECK:                    # <MCInst #{{.*}} SRL16_MM
lbu16 $3, 4($17)            # CHECK: lbu16 $3, 4($17)   # encoding: [0x09,0x94]
                            # CHECK:                    # <MCInst #{{.*}} LBU16_MM
lbu16 $3, -1($16)           # CHECK: lbu16 $3, -1($16)  # encoding: [0x09,0x8f]
                            # CHECK:                    # <MCInst #{{.*}} LBU16_MM
lhu16 $3, 4($16)            # CHECK: lhu16 $3, 4($16)   # encoding: [0x29,0x82]
                            # CHECK:                    # <MCInst #{{.*}} LHU16_MM
lw16 $4, 8($17)             # CHECK: lw16 $4, 8($17)    # encoding: [0x6a,0x12]
                            # CHECK:                    # <MCInst #{{.*}} LW16_MM
sb16 $3, 4($16)             # CHECK: sb16 $3, 4($16)    # encoding: [0x89,0x84]
                            # CHECK:                    # <MCInst #{{.*}} SB16_MM
sh16 $4, 8($17)             # CHECK: sh16 $4, 8($17)    # encoding: [0xaa,0x14]
                            # CHECK:                    # <MCInst #{{.*}} SH16_MM
sw16 $4, 4($17)             # CHECK: sw16 $4, 4($17)    # encoding: [0xea,0x11]
                            # CHECK:                    # <MCInst #{{.*}} SW16_MM
sw16 $zero, 4($17)          # CHECK: sw16 $zero, 4($17) # encoding: [0xe8,0x11]
                            # CHECK:                    # <MCInst #{{.*}} SW16_MM
mfhi16 $9                   # CHECK: mfhi16 $9          # encoding: [0x46,0x09]
                            # CHECK:                    # <MCInst #{{.*}} MFHI16_MM
mflo16 $9                   # CHECK: mflo16 $9          # encoding: [0x46,0x49]
                            # CHECK:                    # <MCInst #{{.*}} MFLO16_MM
move $25, $1                # CHECK: move $25, $1       # encoding: [0x0f,0x21]
                            # CHECK:                    # <MCInst #{{.*}} MOVE16_MM
jrc $9                      # CHECK: jrc $9             # encoding: [0x45,0xa9]
                            # CHECK:                    # <MCInst #{{.*}} JRC16_MM
jalr $9                     # CHECK: jalr $9            # encoding: [0x45,0xc9]
                            # CHECK:                    # <MCInst #{{.*}} JALR16_MM
jalrs16 $9                  # CHECK: jalrs16 $9         # encoding: [0x45,0xe9]
                            # CHECK:                    # <MCInst #{{.*}} MOVE16_MM
jr16 $9                     # CHECK: jr16 $9            # encoding: [0x45,0x89]
                            # CHECK:                    # <MCInst #{{.*}} JR16_MM
li16 $3, -1                 # CHECK: li16 $3, -1        # encoding: [0xed,0xff]
                            # CHECK:                    # <MCInst #{{.*}} LI16_MM
li16 $3, 126                # CHECK: li16 $3, 126       # encoding: [0xed,0xfe]
                            # CHECK:                    # <MCInst #{{.*}} LI16_MM
addiur1sp $7, 4             # CHECK: addiur1sp $7, 4    # encoding: [0x6f,0x83]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUR1SP_MM
addiur2 $6, $7, -1          # CHECK: addiur2 $6, $7, -1 # encoding: [0x6f,0x7e]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUR2_MM
addiur2 $6, $7, 12          # CHECK: addiur2 $6, $7, 12 # encoding: [0x6f,0x76]
                            # CHECK:                    # <MCInst #{{.*}} ADDIUR2_MM
addius5 $7, -2              # CHECK: addius5 $7, -2     # encoding: [0x4c,0xfc]
nop                         # CHECK: nop                # encoding: [0x00,0x00,0x00,0x00]
beqz16 $6, 20               # CHECK: beqz16 $6, 20      # encoding: [0x8f,0x0a]
                            # CHECK:                    # <MCInst #{{.*}} BEQZ16_MM
bnez16 $6, 20               # CHECK: bnez16 $6, 20      # encoding: [0xaf,0x0a]
                            # CHECK:                    # <MCInst #{{.*}} BNEZ16_MM
b16 132                     # CHECK: b16 132            # encoding: [0xcc,0x42]
                            # CHECK:                    # <MCInst #{{.*}} B16_MM
lwm16 $16, $17, $ra, 8($sp) # CHECK: lwm16 $16, $17, $ra, 8($sp) # encoding: [0x45,0x12]
                            # CHECK:                             # <MCInst #{{.*}} LWM16_MM
swm16 $16, $17, $ra, 8($sp) # CHECK: swm16 $16, $17, $ra, 8($sp) # encoding: [0x45,0x52]
                            # CHECK:                      # <MCInst #{{.*}} SWM16_MM
movep $5, $6, $2, $3        # CHECK: movep $5, $6, $2, $3 # encoding: [0x84,0x34]
                            # CHECK:                      # <MCInst #{{.*}} MOVEP_MM
break16 8                   # CHECK: break16 8            # encoding: [0x46,0x88]
                            # CHECK:                      # <MCInst #{{.*}} BREAK16_MM
sdbbp16 14                  # CHECK: sdbbp16 14           # encoding: [0x46,0xce]
                            # CHECK:                      # <MCInst #{{.*}} SDBBP16_MM
lw $3, 32($sp)              # CHECK: lw $3, 32($sp)       # encoding: [0x48,0x68]
                            # CHECK:                      # <MCInst #{{.*}} LWSP_MM
sw $4, 124($sp)             # CHECK: sw $4, 124($sp)      # encoding: [0xc8,0x9f]
                            # CHECK:                      # <MCInst #{{.*}} SWSP_MM
lw $3, 32($gp)              # CHECK: lw $3, 32($gp)       # encoding: [0x65,0x88]
                            # CHECK:                      # <MCInst #{{.*}} LWGP_MM
abs.s $f0, $f2              # CHECK:  abs.s $f0, $f2    # encoding: [0x54,0x02,0x03,0x7b]
                            # CHECK-NEXT:               # <MCInst #{{[0-9]+}} FABS_S_MM
abs.d $f4, $f6              # CHECK:  abs.d $f4, $f6    # encoding: [0x54,0x86,0x23,0x7b]
                            # CHECK-NEXT:               # <MCInst #{{[0-9]+}} FABS_D32_MM
sqrt.s  $f0, $f12           # CHECK:  sqrt.s  $f0, $f12 # encoding: [0x54,0x0c,0x0a,0x3b]
                            # CHECK-NEXT:               # <MCInst #{{[0-9]+}} FSQRT_S_MM
sqrt.d  $f0, $f12           # CHECK:  sqrt.d  $f0, $f12 # encoding: [0x54,0x0c,0x4a,0x3b]
                            # CHECK-NEXT:               # <MCInst #{{[0-9]+}} FSQRT_D32_MM
add $9, $6, $7              # CHECK: add $9, $6, $7         # encoding: [0x00,0xe6,0x49,0x10]
add.d $f0, $f2, $f4         # CHECK: add.d $f0, $f2, $f4    # encoding: [0x54,0x82,0x01,0x30]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FADD_D32_MM
addi $9, $6, 17767          # CHECK: addi $9, $6, 17767     # encoding: [0x11,0x26,0x45,0x67]
addiu $9, $6, -15001        # CHECK: addiu $9, $6, -15001   # encoding: [0x31,0x26,0xc5,0x67]
addi $9, $6, 17767          # CHECK: addi $9, $6, 17767     # encoding: [0x11,0x26,0x45,0x67]
addiu $9, $6, -15001        # CHECK: addiu $9, $6, -15001   # encoding: [0x31,0x26,0xc5,0x67]
addu $9, $6, $7             # CHECK: addu $9, $6, $7        # encoding: [0x00,0xe6,0x49,0x50]
sub $9, $6, $7              # CHECK: sub $9, $6, $7         # encoding: [0x00,0xe6,0x49,0x90]
subu $4, $3, $5             # CHECK: subu $4, $3, $5        # encoding: [0x00,0xa3,0x21,0xd0]
sub $6, $zero, $7           # CHECK: neg $6, $7             # encoding: [0x00,0xe0,0x31,0x90]
sub.d $f0, $f2, $f4         # CHECK: sub.d $f0, $f2, $f4    # encoding: [0x54,0x82,0x01,0x70]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FSUB_D32_MM
subu $6, $zero, $7          # CHECK: negu $6, $7            # encoding: [0x00,0xe0,0x31,0xd0]
addu $7, $8, $zero          # CHECK: addu $7, $8, $zero     # encoding: [0x00,0x08,0x39,0x50]
slt $3, $3, $5              # CHECK: slt $3, $3, $5         # encoding: [0x00,0xa3,0x1b,0x50]
slti $3, $3, 103            # CHECK: slti $3, $3, 103       # encoding: [0x90,0x63,0x00,0x67]
slti $3, $3, 103            # CHECK: slti $3, $3, 103       # encoding: [0x90,0x63,0x00,0x67]
sltiu $3, $3, 103           # CHECK: sltiu $3, $3, 103      # encoding: [0xb0,0x63,0x00,0x67]
sltu $3, $3, $5             # CHECK: sltu $3, $3, $5        # encoding: [0x00,0xa3,0x1b,0x90]
lui $9, 17767               # CHECK: lui $9, 17767          # encoding: [0x41,0xa9,0x45,0x67]
and $9, $6, $7              # CHECK: and $9, $6, $7         # encoding: [0x00,0xe6,0x4a,0x50]
andi $9, $6, 17767          # CHECK: andi $9, $6, 17767     # encoding: [0xd1,0x26,0x45,0x67]
or $3, $4, $5               # CHECK: or $3, $4, $5          # encoding: [0x00,0xa4,0x1a,0x90]
ori $9, $6, 17767           # CHECK: ori $9, $6, 17767      # encoding: [0x51,0x26,0x45,0x67]
xor $3, $3, $5              # CHECK: xor $3, $3, $5         # encoding: [0x00,0xa3,0x1b,0x10]
xori $9, $6, 17767          # CHECK: xori $9, $6, 17767     # encoding: [0x71,0x26,0x45,0x67]
nor $9, $6, $7              # CHECK: nor $9, $6, $7         # encoding: [0x00,0xe6,0x4a,0xd0]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} NOR_MM
not $7, $8                  # CHECK: not $7, $8             # encoding: [0x00,0x08,0x3a,0xd0]
not $7                      # CHECK: not $7, $7             # encoding: [0x00,0x07,0x3a,0xd0]
mul $9, $6, $7              # CHECK: mul $9, $6, $7         # encoding: [0x00,0xe6,0x4a,0x10]
mul.d $f0, $f2, $f4         # CHECK: mul.d $f0, $f2, $f4    # encoding: [0x54,0x82,0x01,0xb0]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FMUL_D32_MM
mult $9, $7                 # CHECK: mult $9, $7            # encoding: [0x00,0xe9,0x8b,0x3c]
multu $9, $7                # CHECK: multu $9, $7           # encoding: [0x00,0xe9,0x9b,0x3c]
div $zero, $9, $7           # CHECK: div $zero, $9, $7      # encoding: [0x00,0xe9,0xab,0x3c]
div.d $f0, $f2, $f4         # CHECK: div.d $f0, $f2, $f4    # encoding: [0x54,0x82,0x01,0xf0]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FDIV_D32_MM
divu $zero, $9, $7          # CHECK: divu $zero, $9, $7     # encoding: [0x00,0xe9,0xbb,0x3c]
rdhwr $5, $29, 2            # CHECK: rdhwr $5, $29, 2       # encoding: [0x00,0xbd,0x6b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} RDHWR_MM
rdhwr $5, $29, 0            # CHECK: rdhwr $5, $29          # encoding: [0x00,0xbd,0x6b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} RDHWR_MM
rdhwr $5, $29               # CHECK: rdhwr $5, $29          # encoding: [0x00,0xbd,0x6b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} RDHWR_MM
sll $4, $3, 7               # CHECK: sll $4, $3, 7          # encoding: [0x00,0x83,0x38,0x00]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLL_MM
sllv $2, $3, $5             # CHECK: sllv $2, $3, $5        # encoding: [0x00,0x65,0x10,0x10]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLLV_MM
sra $4, $3, 7               # CHECK: sra $4, $3, 7          # encoding: [0x00,0x83,0x38,0x80]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRA_MM
srav $2, $3, $5             # CHECK: srav $2, $3, $5        # encoding: [0x00,0x65,0x10,0x90]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRAV_MM
srl $4, $3, 7               # CHECK: srl $4, $3, 7          # encoding: [0x00,0x83,0x38,0x40]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRL_MM
srlv $2, $3, $5             # CHECK: srlv $2, $3, $5        # encoding: [0x00,0x65,0x10,0x50]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRLV_MM
rotr $9, $6, 7              # CHECK: rotr $9, $6, 7         # encoding: [0x01,0x26,0x38,0xc0]
rotrv $9, $6, $7            # CHECK: rotrv $9, $6, $7       # encoding: [0x00,0xc7,0x48,0xd0]
lb $5, 8($4)                # CHECK: lb $5, 8($4)           # encoding: [0x1c,0xa4,0x00,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} LB_MM
lbu $6, 8($4)               # CHECK: lbu $6, 8($4)          # encoding: [0x14,0xc4,0x00,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} LBu_MM
lh $2, 8($4)                # CHECK: lh $2, 8($4)           # encoding: [0x3c,0x44,0x00,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} LH_MM
lhu $4, 8($2)               # CHECK: lhu $4, 8($2)          # encoding: [0x34,0x82,0x00,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} LHu_MM
lw  $6, 4($5)               # CHECK: lw  $6, 4($5)          # encoding: [0xfc,0xc5,0x00,0x04]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} LW_MM
lw $6, 123($sp)             # CHECK: lw $6, 123($sp)        # encoding: [0xfc,0xdd,0x00,0x7b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} LW_MM
sb $5, 8($4)                # CHECK: sb $5, 8($4)           # encoding: [0x18,0xa4,0x00,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SB_MM
sh  $2, 8($4)               # CHECK: sh  $2, 8($4)          # encoding: [0x38,0x44,0x00,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SH_MM
sw  $5, 4($6)               # CHECK: sw  $5, 4($6)          # encoding: [0xf8,0xa6,0x00,0x04]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SW_MM
sw $5, 123($sp)             # CHECK: sw $5, 123($sp)        # encoding: [0xf8,0xbd,0x00,0x7b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SW_MM
lwu $2, 8($4)               # CHECK: lwu $2, 8($4)          # encoding: [0x60,0x44,0xe0,0x08]
lwl $4, 16($5)              # CHECK: lwl $4, 16($5)         # encoding: [0x60,0x85,0x00,0x10]
lwr $4, 16($5)              # CHECK: lwr $4, 16($5)         # encoding: [0x60,0x85,0x10,0x10]
swl $4, 16($5)              # CHECK: swl $4, 16($5)         # encoding: [0x60,0x85,0x80,0x10]
swr $4, 16($5)              # CHECK: swr $4, 16($5)         # encoding: [0x60,0x85,0x90,0x10]
mov.s $f0, $f2              # CHECK: mov.s $f0, $f2         # encoding: [0x54,0x02,0x00,0x7b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FMOV_S_MM
mov.d $f0, $f2              # CHECK: mov.d $f0, $f2         # encoding: [0x54,0x02,0x20,0x7b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FMOV_D32_MM
movz $9, $6, $7             # CHECK: movz $9, $6, $7        # encoding: [0x00,0xe6,0x48,0x58]
movn $9, $6, $7             # CHECK: movn $9, $6, $7        # encoding: [0x00,0xe6,0x48,0x18]
movt $9, $6, $fcc0          # CHECK: movt $9, $6, $fcc0     # encoding: [0x55,0x26,0x09,0x7b]
movf $9, $6, $fcc0          # CHECK: movf $9, $6, $fcc0     # encoding: [0x55,0x26,0x01,0x7b]
# FIXME: MTHI should also have its 16 bit implementation selected in micromips
mthi   $6                   # CHECK: mthi   $6              # encoding: [0x00,0x06,0x2d,0x7c]
mfhi   $6                   # CHECK: mfhi   $6              # encoding: [0x00,0x06,0x0d,0x7c]
# FIXME: MTLO should also have its 16 bit implementation selected in micromips
mtlo   $6                   # CHECK: mtlo   $6              # encoding: [0x00,0x06,0x3d,0x7c]
mflo   $6                   # CHECK: mflo   $6              # encoding: [0x00,0x06,0x1d,0x7c]
mfc1  $3, $f4               # CHECK: mfc1  $3, $f4          # encoding: [0x54,0x64,0x20,0x3b]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} MFC1_MM
mtc1  $2, $f4               # CHECK: mtc1  $2, $f4          # encoding: [0x54,0x44,0x28,0x3b]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} MTC1_MM
mfhc1 $4, $f0               # CHECK: mfhc1 $4, $f0          # encoding: [0x54,0x80,0x30,0x3b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} MFHC1_D32_MM
mthc1 $4, $f0               # CHECK: mthc1 $4, $f0          # encoding: [0x54,0x80,0x38,0x3b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} MTHC1_D32_MM
madd   $4, $5               # CHECK: madd   $4, $5          # encoding: [0x00,0xa4,0xcb,0x3c]
maddu  $4, $5               # CHECK: maddu  $4, $5          # encoding: [0x00,0xa4,0xdb,0x3c]
msub   $4, $5               # CHECK: msub   $4, $5          # encoding: [0x00,0xa4,0xeb,0x3c]
msubu  $4, $5               # CHECK: msubu  $4, $5          # encoding: [0x00,0xa4,0xfb,0x3c]
neg.d $f0, $f2              # CHECK: neg.d $f0, $f2         # encoding: [0x54,0x02,0x2b,0x7b]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} FNEG_D32_MM
clz $9, $6                  # CHECK: clz $9, $6             # encoding: [0x01,0x26,0x5b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} CLZ_MM
clo $9, $6                  # CHECK: clo $9, $6             # encoding: [0x01,0x26,0x4b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} CLO_MM
seb $9, $6                  # CHECK: seb $9, $6             # encoding: [0x01,0x26,0x2b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SEB_MM
seb $9                      # CHECK: seb $9, $9             # encoding: [0x01,0x29,0x2b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SEB_MM
seh $9, $6                  # CHECK: seh $9, $6             # encoding: [0x01,0x26,0x3b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SEH_MM
seh $9                      # CHECK: seh $9, $9             # encoding: [0x01,0x29,0x3b,0x3c]
                            # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SEH_MM
wsbh $9, $6                 # CHECK: wsbh $9, $6            # encoding: [0x01,0x26,0x7b,0x3c]
ext $9, $6, 3, 7            # CHECK: ext $9, $6, 3, 7       # encoding: [0x01,0x26,0x30,0xec]
ins $9, $6, 3, 7            # CHECK: ins $9, $6, 3, 7       # encoding: [0x01,0x26,0x48,0xcc]
j 1328                      # CHECK: j 1328                 # encoding: [0xd4,0x00,0x02,0x98]
jal 1328                    # CHECK: jal 1328               # encoding: [0xf4,0x00,0x02,0x98]
jalr $ra, $6                # CHECK: jalr $ra, $6           # encoding: [0x03,0xe6,0x0f,0x3c]
jr $7                       # CHECK: jr $7                  # encoding: [0x00,0x07,0x0f,0x3c]
beq $9, $6, 1332            # CHECK: beq $9, $6, 1332       # encoding: [0x94,0xc9,0x02,0x9a]
bgez $6, 1332               # CHECK: bgez $6, 1332          # encoding: [0x40,0x46,0x02,0x9a]
bgezal $6, 1332             # CHECK: bgezal $6, 1332        # encoding: [0x40,0x66,0x02,0x9a]
bltzal $6, 1332             # CHECK: bltzal $6, 1332        # encoding: [0x40,0x26,0x02,0x9a]
bgtz $6, 1332               # CHECK: bgtz $6, 1332          # encoding: [0x40,0xc6,0x02,0x9a]
blez $6, 1332               # CHECK: blez $6, 1332          # encoding: [0x40,0x86,0x02,0x9a]
bne $9, $6, 1332            # CHECK: bne $9, $6, 1332       # encoding: [0xb4,0xc9,0x02,0x9a]
bltz $6, 1332               # CHECK: bltz $6, 1332          # encoding: [0x40,0x06,0x02,0x9a]
teq $8, $9                  # CHECK: teq $8, $9             # encoding: [0x01,0x28,0x00,0x3c]
tge $8, $9                  # CHECK: tge $8, $9             # encoding: [0x01,0x28,0x02,0x3c]
tgeu $8, $9                 # CHECK: tgeu $8, $9            # encoding: [0x01,0x28,0x04,0x3c]
tlt $8, $9                  # CHECK: tlt $8, $9             # encoding: [0x01,0x28,0x08,0x3c]
tltu $8, $9                 # CHECK: tltu $8, $9            # encoding: [0x01,0x28,0x0a,0x3c]
tne $8, $9                  # CHECK: tne $8, $9             # encoding: [0x01,0x28,0x0c,0x3c]
teqi $9, 17767              # CHECK: teqi $9, 17767         # encoding: [0x41,0xc9,0x45,0x67]
tgei $9, 17767              # CHECK: tgei $9, 17767         # encoding: [0x41,0x29,0x45,0x67]
tgeiu $9, 17767             # CHECK: tgeiu $9, 17767        # encoding: [0x41,0x69,0x45,0x67]
tlti $9, 17767              # CHECK: tlti $9, 17767         # encoding: [0x41,0x09,0x45,0x67]
tltiu $9, 17767             # CHECK: tltiu $9, 17767        # encoding: [0x41,0x49,0x45,0x67]
tnei $9, 17767              # CHECK: tnei $9, 17767         # encoding: [0x41,0x89,0x45,0x67]
cache 1, 8($5)              # CHECK: cache 1, 8($5)         # encoding: [0x20,0x25,0x60,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} CACHE_MM
pref 1, 8($5)               # CHECK: pref 1, 8($5)          # encoding: [0x60,0x25,0x20,0x08]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} PREF_MM
ssnop                       # CHECK: ssnop                  # encoding: [0x00,0x00,0x08,0x00]
ehb                         # CHECK: ehb                    # encoding: [0x00,0x00,0x18,0x00]
pause                       # CHECK: pause                  # encoding: [0x00,0x00,0x28,0x00]
ll $2, 8($4)                # CHECK: ll $2, 8($4)           # encoding: [0x60,0x44,0x30,0x08]
sc $2, 8($4)                # CHECK: sc $2, 8($4)           # encoding: [0x60,0x44,0xb0,0x08]
lwxs $2, $3($4)             # CHECK: lwxs $2, $3($4)        # encoding: [0x00,0x64,0x11,0x18]
bgezals $6, 1332            # CHECK: bgezals $6, 1332       # encoding: [0x42,0x66,0x02,0x9a]
bltzals $6, 1332            # CHECK: bltzals $6, 1332       # encoding: [0x42,0x26,0x02,0x9a]
beqzc $9, 1332              # CHECK: beqzc $9, 1332         # encoding: [0x40,0xe9,0x02,0x9a]
bnezc $9, 1332              # CHECK: bnezc $9, 1332         # encoding: [0x40,0xa9,0x02,0x9a]
jals 1328                   # CHECK: jals 1328              # encoding: [0x74,0x00,0x02,0x98]
jalrs $ra, $6               # CHECK: jalrs $ra, $6          # encoding: [0x03,0xe6,0x4f,0x3c]
lwm32 $16, $17, 8($4)       # CHECK: lwm32 $16, $17, 8($4)  # encoding: [0x20,0x44,0x50,0x08]
lwm32 $16, $17, $18, $19, $20, $21, $22, $23, $fp, -1660($27)   # CHECK: lwm32 $16, $17, $18, $19, $20, $21, $22, $23, $fp, -1660($27)  # encoding: [0x21,0x3b,0x59,0x84]
swm32 $16, $17, 8($4)       # CHECK: swm32 $16, $17, 8($4)  # encoding: [0x20,0x44,0xd0,0x08]
swp $16, 8($4)              # CHECK: swp $16, 8($4)         # encoding: [0x22,0x04,0x90,0x08]
lwp $16, 8($4)              # CHECK: lwp $16, 8($4)         # encoding: [0x22,0x04,0x10,0x08]
nop                         # CHECK: nop                    # encoding: [0x00,0x00,0x00,0x00]
addiupc $2, 20              # CHECK: addiupc $2, 20         # encoding: [0x79,0x00,0x00,0x05]
addiupc $7, 16777212        # CHECK: addiupc $7, 16777212   # encoding: [0x7b,0xbf,0xff,0xff]
addiupc $7, -16777216       # CHECK: addiupc $7, -16777216  # encoding: [0x7b,0xc0,0x00,0x00]
ei                          # CHECK: ei                     # encoding: [0x00,0x00,0x57,0x7c]
ei $10                      # CHECK: ei $10                 # encoding: [0x00,0x0a,0x57,0x7c]
cachee 1, 8($5)             # CHECK: cachee 1, 8($5)        # encoding: [0x60,0x25,0xa6,0x08]
prefe 1, 8($5)              # CHECK: prefe 1, 8($5)         # encoding: [0x60,0x25,0xa4,0x08]
prefx 1, $3($5)             # CHECK: prefx 1, $3($5)        # encoding: [0x54,0x65,0x09,0xa0]
lhue $4, 8($2)              # CHECK: lhue $4, 8($2)         # encoding: [0x60,0x82,0x62,0x08]
lbe $4, 8($2)               # CHECK: lbe $4, 8($2)          # encoding: [0x60,0x82,0x68,0x08]
lbue $4, 8($2)              # CHECK: lbue $4, 8($2)         # encoding: [0x60,0x82,0x60,0x08]
lhe $4, 8($2)               # CHECK: lhe $4, 8($2)          # encoding: [0x60,0x82,0x6a,0x08]
lwe $4, 8($2)               # CHECK: lwe $4, 8($2)          # encoding: [0x60,0x82,0x6e,0x08]
sbe $5, 8($4)               # CHECK: sbe $5, 8($4)          # encoding: [0x60,0xa4,0xa8,0x08]
she $5, 8($4)               # CHECK: she $5, 8($4)          # encoding: [0x60,0xa4,0xaa,0x08]
swe $5, 8($4)               # CHECK: swe $5, 8($4)          # encoding: [0x60,0xa4,0xae,0x08]
swre $24, 5($3)             # CHECK: swre $24, 5($3)        # encoding: [0x63,0x03,0xa2,0x05]
swle $24, 5($3)             # CHECK: swle $24, 5($3)        # encoding: [0x63,0x03,0xa0,0x05]
lwre $24, 5($3)             # CHECK: lwre $24, 5($3)        # encoding: [0x63,0x03,0x66,0x05]
lwle $24, 2($4)             # CHECK: lwle $24, 2($4)        # encoding: [0x63,0x04,0x64,0x02]
lle $2, 8($4)               # CHECK: lle $2, 8($4)          # encoding: [0x60,0x44,0x6c,0x08]
sce $2, 8($4)               # CHECK: sce $2, 8($4)          # encoding: [0x60,0x44,0xac,0x08]
syscall                     # CHECK: syscall                # encoding: [0x00,0x00,0x8b,0x7c]
syscall 396                 # CHECK: syscall 396            # encoding: [0x01,0x8c,0x8b,0x7c]
# FIXME: ldc1 should accept uneven registers
# ldc1 $f7, 300($10)        # -CHECK: ldc1 $f7, 300($10)    # encoding: [0xbc,0xea,0x01,0x2c]
ldc1 $f8, 300($10)          # CHECK: ldc1 $f8, 300($10)     # encoding: [0xbd,0x0a,0x01,0x2c]
lwc1 $f2, 4($6)             # CHECK: lwc1 $f2, 4($6)        # encoding: [0x9c,0x46,0x00,0x04]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} LWC1_MM
sdc1 $f2, 4($6)             # CHECK: sdc1 $f2, 4($6)        # encoding: [0xb8,0x46,0x00,0x04]
# FIXME: sdc1 should accept uneven registers
# sdc1 $f7, 64($10)         # -CHECK: sdc1 $f7, 64($10)     # encoding: [0xb8,0xea,0x00,0x40]
swc1 $f2, 4($6)             # CHECK: swc1 $f2, 4($6)        # encoding: [0x98,0x46,0x00,0x04]
                            # CHECK-NEXT:                   # <MCInst #{{.*}} SWC1_MM
cfc1 $1, $2                 # CHECK: cfc1 $1, $2            # encoding: [0x54,0x22,0x10,0x3b]
                            # CHECK:                        # <MCInst #{{.*}} CFC1_MM
cfc2 $3, $4                 # CHECK: cfc2 $3, $4            # encoding: [0x00,0x64,0xcd,0x3c]
ctc1 $5, $6                 # CHECK: ctc1 $5, $6            # encoding: [0x54,0xa6,0x18,0x3b]
                            # CHECK:                        # <MCInst #{{.*}} CTC1_MM
ctc2 $7, $8                 # CHECK: ctc2 $7, $8            # encoding: [0x00,0xe8,0xdd,0x3c]
recip.s $f2, $f4            # CHECK: recip.s $f2, $f4       # encoding: [0x54,0x44,0x12,0x3b]
recip.d $f2, $f4            # CHECK: recip.d $f2, $f4       # encoding: [0x54,0x44,0x52,0x3b]
rsqrt.s $f3, $f5            # CHECK: rsqrt.s $f3, $f5       # encoding: [0x54,0x65,0x02,0x3b]
rsqrt.d $f2, $f4            # CHECK: rsqrt.d $f2, $f4       # encoding: [0x54,0x44,0x42,0x3b]
c.eq.d   $fcc1, $f14, $f14  # CHECK: c.eq.d   $fcc1, $f14, $f14 # encoding: [0x55,0xce,0x24,0xbc]
c.eq.s   $fcc5, $f24, $f17  # CHECK: c.eq.s   $fcc5, $f24, $f17 # encoding: [0x56,0x38,0xa0,0xbc]
c.f.d    $fcc4, $f10, $f20  # CHECK: c.f.d    $fcc4, $f10, $f20 # encoding: [0x56,0x8a,0x84,0x3c]
c.f.s    $fcc4, $f30, $f7   # CHECK: c.f.s    $fcc4, $f30, $f7  # encoding: [0x54,0xfe,0x80,0x3c]
c.le.d   $fcc4, $f18, $f0   # CHECK: c.le.d   $fcc4, $f18, $f0  # encoding: [0x54,0x12,0x87,0xbc]
c.le.s   $fcc6, $f24, $f4   # CHECK: c.le.s   $fcc6, $f24, $f4  # encoding: [0x54,0x98,0xc3,0xbc]
c.lt.d   $fcc3, $f8, $f2    # CHECK: c.lt.d   $fcc3, $f8, $f2   # encoding: [0x54,0x48,0x67,0x3c]
c.lt.s   $fcc2, $f17, $f14  # CHECK: c.lt.s   $fcc2, $f17, $f14 # encoding: [0x55,0xd1,0x43,0x3c]
c.nge.d  $fcc5, $f20, $f16  # CHECK: c.nge.d  $fcc5, $f20, $f16 # encoding: [0x56,0x14,0xa7,0x7c]
c.nge.s  $fcc3, $f11, $f8   # CHECK: c.nge.s  $fcc3, $f11, $f8  # encoding: [0x55,0x0b,0x63,0x7c]
c.ngl.s  $fcc2, $f31, $f23  # CHECK: c.ngl.s  $fcc2, $f31, $f23 # encoding: [0x56,0xff,0x42,0xfc]
c.ngle.s $fcc2, $f18, $f23  # CHECK: c.ngle.s $fcc2, $f18, $f23 # encoding: [0x56,0xf2,0x42,0x7c]
c.ngl.d  $f28, $f28         # CHECK: c.ngl.d  $f28, $f28        # encoding: [0x57,0x9c,0x06,0xfc]
c.ngle.d $f0, $f16          # CHECK: c.ngle.d $f0, $f16         # encoding: [0x56,0x00,0x06,0x7c]
c.ngt.d  $fcc4, $f24, $f6   # CHECK: c.ngt.d  $fcc4, $f24, $f6  # encoding: [0x54,0xd8,0x87,0xfc]
c.ngt.s  $fcc5, $f8, $f13   # CHECK: c.ngt.s  $fcc5, $f8, $f13  # encoding: [0x55,0xa8,0xa3,0xfc]
c.ole.d  $fcc2, $f16, $f30  # CHECK: c.ole.d  $fcc2, $f16, $f30 # encoding: [0x57,0xd0,0x45,0xbc]
c.ole.s  $fcc3, $f7, $f20   # CHECK: c.ole.s  $fcc3, $f7, $f20  # encoding: [0x56,0x87,0x61,0xbc]
c.olt.d  $fcc4, $f18, $f28  # CHECK: c.olt.d  $fcc4, $f18, $f28 # encoding: [0x57,0x92,0x85,0x3c]
c.olt.s  $fcc6, $f20, $f7   # CHECK: c.olt.s  $fcc6, $f20, $f7  # encoding: [0x54,0xf4,0xc1,0x3c]
c.seq.d  $fcc4, $f30, $f6   # CHECK: c.seq.d  $fcc4, $f30, $f6  # encoding: [0x54,0xde,0x86,0xbc]
c.seq.s  $fcc7, $f1, $f25   # CHECK: c.seq.s  $fcc7, $f1, $f25  # encoding: [0x57,0x21,0xe2,0xbc]
c.sf.d   $f30, $f0          # CHECK: c.sf.d   $f30, $f0         # encoding: [0x54,0x1e,0x06,0x3c]
c.sf.s   $f14, $f22         # CHECK: c.sf.s   $f14, $f22        # encoding: [0x56,0xce,0x02,0x3c]
c.ueq.d  $fcc4, $f12, $f24  # CHECK: c.ueq.d  $fcc4, $f12, $f24 # encoding: [0x57,0x0c,0x84,0xfc]
c.ueq.s  $fcc6, $f3, $f30   # CHECK: c.ueq.s  $fcc6, $f3, $f30  # encoding: [0x57,0xc3,0xc0,0xfc]
c.ule.d  $fcc7, $f24, $f18  # CHECK: c.ule.d  $fcc7, $f24, $f18 # encoding: [0x56,0x58,0xe5,0xfc]
c.ule.s  $fcc7, $f21, $f30  # CHECK: c.ule.s  $fcc7, $f21, $f30 # encoding: [0x57,0xd5,0xe1,0xfc]
c.ult.d  $fcc6, $f6, $f16   # CHECK: c.ult.d  $fcc6, $f6, $f16  # encoding: [0x56,0x06,0xc5,0x7c]
c.ult.s  $fcc7, $f24, $f10  # CHECK: c.ult.s  $fcc7, $f24, $f10 # encoding: [0x55,0x58,0xe1,0x7c]
c.un.d   $fcc6, $f22, $f24  # CHECK: c.un.d   $fcc6, $f22, $f24 # encoding: [0x57,0x16,0xc4,0x7c]
c.un.s   $fcc1, $f30, $f4   # CHECK: c.un.s   $fcc1, $f30, $f4  # encoding: [0x54,0x9e,0x20,0x7c]
cvt.w.d $f0, $f2            # CHECK: cvt.w.d    $f0, $f2        # encoding: [0x54,0x02,0x49,0x3b]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} CVT_W_D32_MM
cvt.d.s $f0, $f2            # CHECK: cvt.d.s    $f0, $f2        # encoding: [0x54,0x02,0x13,0x7b]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} CVT_D32_S_MM
cvt.d.w $f0, $f2            # CHECK: cvt.d.w    $f0, $f2        # encoding: [0x54,0x02,0x33,0x7b]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} CVT_D32_W_MM
cvt.s.d $f0, $f2            # CHECK: cvt.s.d    $f0, $f2        # encoding: [0x54,0x02,0x1b,0x7b]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} CVT_S_D32_MM
bc1t 8                      # CHECK: bc1t 8                     # encoding: [0x43,0xa0,0x00,0x04]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} BC1T_MM
bc1f 16                     # CHECK: bc1f 16                    # encoding: [0x43,0x80,0x00,0x08]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} BC1F_MM
bc1t $fcc1, 4               # CHECK: bc1t $fcc1, 4              # encoding: [0x43,0xa4,0x00,0x02]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} BC1T_MM
bc1f $fcc2, -20             # CHECK: bc1f $fcc2, -20            # encoding: [0x43,0x88,0xff,0xf6]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} BC1F_MM
sync                        # CHECK: sync                       # encoding: [0x00,0x00,0x6b,0x7c]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} SYNC_MM
sync 0                      # CHECK: sync                       # encoding: [0x00,0x00,0x6b,0x7c]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} SYNC_MM
sync 1                      # CHECK: sync 1                     # encoding: [0x00,0x01,0x6b,0x7c]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} SYNC_MM
synci 64($5)                # CHECK: synci 64($5)               # encoding: [0x42,0x05,0x00,0x40]
                            # CHECK-NEXT:                       # <MCInst #{{[0-9]+}} SYNCI_MM
add.s  $f4, $f6, $f8        # CHECK:      add.s $f4, $f6, $f8   # encoding: [0x55,0x06,0x20,0x30]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FADD_S_MM
sub.s  $f4, $f6, $f8        # CHECK:       sub.s $f4, $f6, $f8  # encoding: [0x55,0x06,0x20,0x70]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FSUB_S_MM
mul.s  $f4, $f6, $f8        # CHECK:       mul.s $f4, $f6, $f8  # encoding: [0x55,0x06,0x20,0xb0]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FMUL_S_MM
div.s  $f4, $f6, $f8        # CHECK:       div.s $f4, $f6, $f8  # encoding: [0x55,0x06,0x20,0xf0]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FDIV_S_MM
add.d  $f4, $f6, $f8        # CHECK:       add.d $f4, $f6, $f8  # encoding: [0x55,0x06,0x21,0x30]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FADD_D32_MM
sub.d  $f4, $f6, $f8        # CHECK:       sub.d $f4, $f6, $f8  # encoding: [0x55,0x06,0x21,0x70]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FSUB_D32_MM
mul.d  $f4, $f6, $f8        # CHECK:       mul.d $f4, $f6, $f8  # encoding: [0x55,0x06,0x21,0xb0]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FMUL_D32_MM
div.d  $f4, $f6, $f8        # CHECK:       div.d $f4, $f6, $f8  # encoding: [0x55,0x06,0x21,0xf0]
                            # CHECK-NEXT:                       # <MCInst {{.*}} FDIV_D32_MM
