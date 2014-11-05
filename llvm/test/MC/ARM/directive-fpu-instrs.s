// RUN: llvm-mc -triple armv7-unknown-linux-gnueabi -mattr=+vfp3,-neon %s

.fpu neon
VAND d3, d5, d5
vldr d21, [r7, #296]

@ .thumb should not disable the prior .fpu neon
.thumb

vmov q4, q11 @ v4si
str r6, [r7, #264]
mov r6, r5
vldr d21, [r7, #296]
add r9, r7, #216

fstmfdd sp!, {d8, d9, d10, d11, d12, d13, d14, d15}
