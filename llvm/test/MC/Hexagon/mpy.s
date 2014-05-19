#REQUIRES: object-emission
#This test will be enabled when assembler support has been added.

#RUN: llvm-mc -filetype=obj %s | llvm-objdump -d - | FileCheck %s

# Check encoding bits for half-word multiply instructions.

r7=mpy(r28.l,r20.h):<<1:rnd
#CHECK: ecbcd427 { r7 = mpy(r28.l, r20.h):<<1:rnd }

r18=mpy(r9.l,r21.h):rnd
#CHECK: ec29d532 { r18 = mpy(r9.l, r21.h):rnd }

r19=mpyu(r23.l,r20.l)
#CHECK: ec57d413 { r19 = mpyu(r23.l, r20.l) }

r22=mpyu(r19.l,r30.l):<<1
#CHECK: ecd3de16 { r22 = mpyu(r19.l, r30.l):<<1 }

r19=mpy(r16.h,r19.h)
#CHECK: ec10d373 { r19 = mpy(r16.h, r19.h) }

r30=mpy(r0.h,r16.h):<<1
#CHECK: ec80d07e { r30 = mpy(r0.h, r16.h):<<1 }

lr=mpy(r15.h,r25.l)
#CHECK: ec0fd95f { r31 = mpy(r15.h, r25.l) }

r29=mpy(r28.h,r15.l):<<1
#CHECK: ec9ccf5d { r29 = mpy(r28.h, r15.l):<<1 }

r20=mpy(r31.l,r19.h)
#CHECK: ec1fd334 { r20 = mpy(r31.l, r19.h) }

r24=mpy(r19.l,r15.h):<<1
#CHECK: ec93cf38 { r24 = mpy(r19.l, r15.h):<<1 }

r30=mpy(r10.l,sp.l)
#CHECK: ec0add1e { r30 = mpy(r10.l, r29.l) }

r7=mpy(r3.l,r4.l):<<1
#CHECK: ec83c407 { r7 = mpy(r3.l, r4.l):<<1 }

r30=mpy(r23.h,r2.h):rnd:sat
#CHECK: ec37c2fe { r30 = mpy(r23.h, r2.h):rnd:sat }

r5=mpy(r28.h,r27.h):<<1:rnd:sat
#CHECK: ecbcdbe5 { r5 = mpy(r28.h, r27.h):<<1:rnd:sat }

r26=mpy(r21.l,r23.l):rnd
#CHECK: ec35d71a { r26 = mpy(r21.l, r23.l):rnd }

sp=mpy(r25.h,r12.h):<<1:rnd
#CHECK: ecb9cc7d { r29 = mpy(r25.h, r12.h):<<1:rnd }

r1=mpy(r27.h,r29.h):rnd
#CHECK: ec3bdd61 { r1 = mpy(r27.h, r29.h):rnd }

r0=mpy(r2.h,r11.h):<<1:sat
#CHECK: ec82cbe0 { r0 = mpy(r2.h, r11.h):<<1:sat }

r3=mpy(r20.l,r30.l):rnd:sat
#CHECK: ec34de83 { r3 = mpy(r20.l, r30.l):rnd:sat }

r4=mpy(r21.h,r5.l):<<1:sat
#CHECK: ec95c5c4 { r4 = mpy(r21.h, r5.l):<<1:sat }

fp=mpy(r20.l,r12.h):rnd:sat
#CHECK: ec34ccbe { r30 = mpy(r20.l, r12.h):rnd:sat }

r12=mpy(sp.l,r30.h):<<1:rnd:sat
#CHECK: ecbddeac { r12 = mpy(r29.l, r30.h):<<1:rnd:sat }

r6=mpy(r10.h,fp.l):rnd:sat
#CHECK: ec2adec6 { r6 = mpy(r10.h, r30.l):rnd:sat }

r24=mpy(r12.h,r1.h):sat
#CHECK: ec0cc1f8 { r24 = mpy(r12.h, r1.h):sat }

r29=mpyu(r25.h,sp.l)
#CHECK: ec59dd5d { r29 = mpyu(r25.h, r29.l) }

r24=mpyu(lr.h,r29.l):<<1
#CHECK: ecdfdd58 { r24 = mpyu(r31.h, r29.l):<<1 }

r26=mpyu(r21.l,r18.h)
#CHECK: ec55d23a { r26 = mpyu(r21.l, r18.h) }

r29=mpyu(r4.l,r26.h):<<1
#CHECK: ecc4da3d { r29 = mpyu(r4.l, r26.h):<<1 }

fp=mpy(r8.l,r0.l):sat
#CHECK: ec08c09e { r30 = mpy(r8.l, r0.l):sat }

r1=mpy(r26.l,r16.l):<<1:sat
#CHECK: ec9ad081 { r1 = mpy(r26.l, r16.l):<<1:sat }

r16=mpyu(r26.h,r6.h)
#CHECK: ec5ac670 { r16 = mpyu(r26.h, r6.h) }

lr=mpyu(r23.h,r13.h):<<1
#CHECK: ecd7cd7f { r31 = mpyu(r23.h, r13.h):<<1 }

r14=mpy(r2.l,r7.h):sat
#CHECK: ec02c7ae { r14 = mpy(r2.l, r7.h):sat }

r9=mpy(r1.l,r9.h):<<1:sat
#CHECK: ec81c9a9 { r9 = mpy(r1.l, r9.h):<<1:sat }

r9=mpy(r30.l,r4.l):<<1:rnd:sat
#CHECK: ecbec489 { r9 = mpy(r30.l, r4.l):<<1:rnd:sat }

r9=mpy(r15.h,r27.l):<<1:rnd
#CHECK: ecafdb49 { r9 = mpy(r15.h, r27.l):<<1:rnd }

r16=mpy(r6.h,r16.l):rnd
#CHECK: ec26d050 { r16 = mpy(r6.h, r16.l):rnd }

r1=mpy(r10.l,r29.l):<<1:rnd
#CHECK: ecaadd01 { r1 = mpy(r10.l, r29.l):<<1:rnd }

r7=mpy(r4.h,r23.l):sat
#CHECK: ec04d7c7 { r7 = mpy(r4.h, r23.l):sat }

r17=mpy(r12.h,r26.l):<<1:rnd:sat
#CHECK: ecacdad1 { r17 = mpy(r12.h, r26.l):<<1:rnd:sat }
