//===-- ARM_GCC_Registers.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_ARM_GCC_Registers_h_
#define utility_ARM_GCC_Registers_h_

enum
{
    gcc_r0 = 0,
    gcc_r1,
    gcc_r2,
    gcc_r3,
    gcc_r4,
    gcc_r5,
    gcc_r6,
    gcc_r7,
    gcc_r8,
    gcc_r9,
    gcc_r10,
    gcc_r11,
    gcc_r12,
    gcc_sp,
    gcc_lr,
    gcc_pc,
    gcc_cpsr
};

enum
{
//  Name                    Nr   Rel Offset    Size  Type            Raw value
    gdb_arm_r0          =   0, //  0      0       4 int32_t
    gdb_arm_r1          =   1, //  1      4       4 int32_t
    gdb_arm_r2          =   2, //  2      8       4 int32_t
    gdb_arm_r3          =   3, //  3     12       4 int32_t
    gdb_arm_r4          =   4, //  4     16       4 int32_t
    gdb_arm_r5          =   5, //  5     20       4 int32_t
    gdb_arm_r6          =   6, //  6     24       4 int32_t
    gdb_arm_r7          =   7, //  7     28       4 int32_t
    gdb_arm_r8          =   8, //  8     32       4 int32_t
    gdb_arm_r9          =   9, //  9     36       4 int32_t
    gdb_arm_r10         =  10, // 10     40       4 int32_t
    gdb_arm_r11         =  11, // 11     44       4 int32_t
    gdb_arm_r12         =  12, // 12     48       4 int32_t
    gdb_arm_sp          =  13, // 13     52       4 int32_t
    gdb_arm_lr          =  14, // 14     56       4 int32_t
    gdb_arm_pc          =  15, // 15     60       4 int32_t
    gdb_arm_f0          =  16, // 16     64      12 _arm_ext_littlebyte_bigword
    gdb_arm_f1          =  17, // 17     76      12 _arm_ext_littlebyte_bigword
    gdb_arm_f2          =  18, // 18     88      12 _arm_ext_littlebyte_bigword
    gdb_arm_f3          =  19, // 19    100      12 _arm_ext_littlebyte_bigword
    gdb_arm_f4          =  20, // 20    112      12 _arm_ext_littlebyte_bigword
    gdb_arm_f5          =  21, // 21    124      12 _arm_ext_littlebyte_bigword
    gdb_arm_f6          =  22, // 22    136      12 _arm_ext_littlebyte_bigword
    gdb_arm_f7          =  23, // 23    148      12 _arm_ext_littlebyte_bigword
    gdb_arm_cpsr        =  24, // 24    172       4 int32_t
    gdb_arm_s0          =  25, // 25    176       4 _ieee_single_little
    gdb_arm_s1          =  26, // 26    180       4 _ieee_single_little
    gdb_arm_s2          =  27, // 27    184       4 _ieee_single_little
    gdb_arm_s3          =  28, // 28    188       4 _ieee_single_little
    gdb_arm_s4          =  29, // 29    192       4 _ieee_single_little
    gdb_arm_s5          =  30, // 30    196       4 _ieee_single_little
    gdb_arm_s6          =  31, // 31    200       4 _ieee_single_little
    gdb_arm_s7          =  32, // 32    204       4 _ieee_single_little
    gdb_arm_s8          =  33, // 33    208       4 _ieee_single_little
    gdb_arm_s9          =  34, // 34    212       4 _ieee_single_little
    gdb_arm_s10         =  35, // 35    216       4 _ieee_single_little
    gdb_arm_s11         =  36, // 36    220       4 _ieee_single_little
    gdb_arm_s12         =  37, // 37    224       4 _ieee_single_little
    gdb_arm_s13         =  38, // 38    228       4 _ieee_single_little
    gdb_arm_s14         =  39, // 39    232       4 _ieee_single_little
    gdb_arm_s15         =  40, // 40    236       4 _ieee_single_little
    gdb_arm_s16         =  41, // 41    240       4 _ieee_single_little
    gdb_arm_s17         =  42, // 42    244       4 _ieee_single_little
    gdb_arm_s18         =  43, // 43    248       4 _ieee_single_little
    gdb_arm_s19         =  44, // 44    252       4 _ieee_single_little
    gdb_arm_s20         =  45, // 45    256       4 _ieee_single_little
    gdb_arm_s21         =  46, // 46    260       4 _ieee_single_little
    gdb_arm_s22         =  47, // 47    264       4 _ieee_single_little
    gdb_arm_s23         =  48, // 48    268       4 _ieee_single_little
    gdb_arm_s24         =  49, // 49    272       4 _ieee_single_little
    gdb_arm_s25         =  50, // 50    276       4 _ieee_single_little
    gdb_arm_s26         =  51, // 51    280       4 _ieee_single_little
    gdb_arm_s27         =  52, // 52    284       4 _ieee_single_little
    gdb_arm_s28         =  53, // 53    288       4 _ieee_single_little
    gdb_arm_s29         =  54, // 54    292       4 _ieee_single_little
    gdb_arm_s30         =  55, // 55    296       4 _ieee_single_little
    gdb_arm_s31         =  56, // 56    300       4 _ieee_single_little
    gdb_arm_fpscr       =  57, // 57    304       4 int32_t
    gdb_arm_d16         =  58, // 58    308       8 _ieee_double_little
    gdb_arm_d17         =  59, // 59    316       8 _ieee_double_little
    gdb_arm_d18         =  60, // 60    324       8 _ieee_double_little
    gdb_arm_d19         =  61, // 61    332       8 _ieee_double_little
    gdb_arm_d20         =  62, // 62    340       8 _ieee_double_little
    gdb_arm_d21         =  63, // 63    348       8 _ieee_double_little
    gdb_arm_d22         =  64, // 64    356       8 _ieee_double_little
    gdb_arm_d23         =  65, // 65    364       8 _ieee_double_little
    gdb_arm_d24         =  66, // 66    372       8 _ieee_double_little
    gdb_arm_d25         =  67, // 67    380       8 _ieee_double_little
    gdb_arm_d26         =  68, // 68    388       8 _ieee_double_little
    gdb_arm_d27         =  69, // 69    396       8 _ieee_double_little
    gdb_arm_d28         =  70, // 70    404       8 _ieee_double_little
    gdb_arm_d29         =  71, // 71    412       8 _ieee_double_little
    gdb_arm_d30         =  72, // 72    420       8 _ieee_double_little
    gdb_arm_d31         =  73, // 73    428       8 _ieee_double_little
    gdb_arm_d0          =  74, //  0    436       8 _ieee_double_little
    gdb_arm_d1          =  75, //  1    444       8 _ieee_double_little
    gdb_arm_d2          =  76, //  2    452       8 _ieee_double_little
    gdb_arm_d3          =  77, //  3    460       8 _ieee_double_little
    gdb_arm_d4          =  78, //  4    468       8 _ieee_double_little
    gdb_arm_d5          =  79, //  5    476       8 _ieee_double_little
    gdb_arm_d6          =  80, //  6    484       8 _ieee_double_little
    gdb_arm_d7          =  81, //  7    492       8 _ieee_double_little
    gdb_arm_d8          =  82, //  8    500       8 _ieee_double_little
    gdb_arm_d9          =  83, //  9    508       8 _ieee_double_little
    gdb_arm_d10         =  84, // 10    516       8 _ieee_double_little
    gdb_arm_d11         =  85, // 11    524       8 _ieee_double_little
    gdb_arm_d12         =  86, // 12    532       8 _ieee_double_little
    gdb_arm_d13         =  87, // 13    540       8 _ieee_double_little
    gdb_arm_d14         =  88, // 14    548       8 _ieee_double_little
    gdb_arm_d15         =  89, // 15    556       8 _ieee_double_little
    gdb_arm_q0          =  90, // 16    564      16 _vec128
    gdb_arm_q1          =  91, // 17    580      16 _vec128
    gdb_arm_q2          =  92, // 18    596      16 _vec128
    gdb_arm_q3          =  93, // 19    612      16 _vec128
    gdb_arm_q4          =  94, // 20    628      16 _vec128
    gdb_arm_q5          =  95, // 21    644      16 _vec128
    gdb_arm_q6          =  96, // 22    660      16 _vec128
    gdb_arm_q7          =  97, // 23    676      16 _vec128
    gdb_arm_q8          =  98, // 24    692      16 _vec128
    gdb_arm_q9          =  99, // 25    708      16 _vec128
    gdb_arm_q10         = 100, // 26    724      16 _vec128
    gdb_arm_q11         = 101, // 27    740      16 _vec128
    gdb_arm_q12         = 102, // 28    756      16 _vec128
    gdb_arm_q13         = 103, // 29    772      16 _vec128
    gdb_arm_q14         = 104, // 30    788      16 _vec128
    gdb_arm_q15         = 105  // 31    804      16 _vec128
};
#endif // utility_ARM_GCC_Registers_h_

