// RUN: %clang_cc1 -E -dM -x assembler-with-cpp < /dev/null | FileCheck -check-prefix ASM %s
//
// ASM:#define __ASSEMBLER__ 1
//
// 
// RUN: %clang_cc1 -fblocks -E -dM < /dev/null | FileCheck -check-prefix BLOCKS %s
//
// BLOCKS:#define __BLOCKS__ 1
// BLOCKS:#define __block __attribute__((__blocks__(byref)))
//
// 
// RUN: %clang_cc1 -x c++ -std=c++0x -E -dM < /dev/null | FileCheck -check-prefix CXX0X %s
//
// CXX0X:#define __GNUG__
// CXX0X:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX0X:#define __GXX_RTTI 1
// CXX0X:#define __GXX_WEAK__ 1
// CXX0X:#define __cplusplus 199711L
// CXX0X:#define __private_extern__ extern
//
// 
// RUN: %clang_cc1 -x c++ -std=c++98 -E -dM < /dev/null | FileCheck -check-prefix CXX98 %s
// 
// CXX98:#define __GNUG__
// CXX98:#define __GXX_RTTI 1
// CXX98:#define __GXX_WEAK__ 1
// CXX98:#define __cplusplus 199711L
// CXX98:#define __private_extern__ extern
//
// 
// RUN: %clang_cc1 -fdeprecated-macro -E -dM < /dev/null | FileCheck -check-prefix DEPRECATED %s
//
// DEPRECATED:#define __DEPRECATED 1
//
// 
// RUN: %clang_cc1 -std=c99 -E -dM < /dev/null | FileCheck -check-prefix C99 %s
//
// C99:#define __STDC_VERSION__ 199901L
// C99:#define __STRICT_ANSI__ 1
//
// 
// RUN: %clang_cc1 -E -dM < /dev/null | FileCheck -check-prefix COMMON %s
//
// COMMON:#define __CONSTANT_CFSTRINGS__ 1
// COMMON:#define __FINITE_MATH_ONLY__ 0
// COMMON:#define __GNUC_MINOR__
// COMMON:#define __GNUC_PATCHLEVEL__
// COMMON:#define __GNUC_STDC_INLINE__ 1
// COMMON:#define __GNUC__
// COMMON:#define __GXX_ABI_VERSION
// COMMON:#define __STDC_HOSTED__ 1
// COMMON:#define __STDC_VERSION__
// COMMON:#define __STDC__ 1
// COMMON:#define __VERSION__
// COMMON:#define __clang__ 1
// COMMON:#define __clang_major__ {{[0-9]+}}
// COMMON:#define __clang_minor__ {{[0-9]+}}
// COMMON:#define __clang_patchlevel__ {{[0-9]+}}
// COMMON:#define __clang_version__
// COMMON:#define __llvm__ 1
//
// 
// RUN: %clang_cc1 -ffreestanding -E -dM < /dev/null | FileCheck -check-prefix FREESTANDING %s
// FREESTANDING:#define __STDC_HOSTED__ 0
// 
// RUN: %clang_cc1 -x c++ -std=gnu++98 -E -dM < /dev/null | FileCheck -check-prefix GXX98 %s
//
// GXX98:#define __GNUG__
// GXX98:#define __GXX_WEAK__ 1
// GXX98:#define __cplusplus 1
// GXX98:#define __private_extern__ extern
//
// 
// RUN: %clang_cc1 -std=iso9899:199409 -E -dM < /dev/null | FileCheck -check-prefix C94 %s
//
// C94:#define __STDC_VERSION__ 199409L
//
// 
// RUN: %clang_cc1 -fms-extensions -triple i686-pc-win32 -E -dM < /dev/null | FileCheck -check-prefix MSEXT %s
//
// MSEXT-NOT:#define __STDC__
// MSEXT:#define _INTEGRAL_MAX_BITS 64
// MSEXT:#define __int16 __INT16_TYPE__
// MSEXT:#define __int32 __INT32_TYPE__
// MSEXT:#define __int64 __INT64_TYPE__
// MSEXT:#define __int8 __INT8_TYPE__
//
// 
// RUN: %clang_cc1 -x objective-c -E -dM < /dev/null | FileCheck -check-prefix OBJC %s
//
// OBJC:#define OBJC_NEW_PROPERTIES 1
// OBJC:#define __NEXT_RUNTIME__ 1
// OBJC:#define __OBJC__ 1
//
//
// RUN: %clang_cc1 -x objective-c -fobjc-gc -E -dM < /dev/null | FileCheck -check-prefix OBJCGC %s
//
// OBJCGC:#define __OBJC_GC__ 1
//
// 
// RUN: %clang_cc1 -x objective-c -fobjc-nonfragile-abi -E -dM < /dev/null | FileCheck -check-prefix NONFRAGILE %s
//
// NONFRAGILE:#define OBJC_ZEROCOST_EXCEPTIONS 1
// NONFRAGILE:#define __OBJC2__ 1
//
// 
// RUN: %clang_cc1 -O1 -E -dM < /dev/null | FileCheck -check-prefix O1 %s
//
// O1:#define __OPTIMIZE__ 1
//
// 
// RUN: %clang_cc1 -fpascal-strings -E -dM < /dev/null | FileCheck -check-prefix PASCAL %s
//
// PASCAL:#define __PASCAL_STRINGS__ 1
//
// 
// RUN: %clang_cc1 -E -dM < /dev/null | FileCheck -check-prefix SCHAR %s
// 
// SCHAR:#define __STDC__ 1
// SCHAR-NOT:#define __UNSIGNED_CHAR__
// SCHAR:#define __clang__ 1
//
// RUN: %clang_cc1 -E -dM -fshort-wchar < /dev/null | FileCheck -check-prefix SHORTWCHAR %s
//
// SHORTWCHAR: #define __SIZEOF_WCHAR_T__ 2
// SHORTWCHAR: #define __WCHAR_MAX__ 65535U
// SHORTWCHAR: #define __WCHAR_TYPE__ unsigned short
// SHORTWCHAR: #define __WCHAR_WIDTH__ 16
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-none-none < /dev/null | FileCheck -check-prefix ARM %s
//
// ARM:#define __APCS_32__ 1
// ARM:#define __ARMEL__ 1
// ARM:#define __ARM_ARCH_6J__ 1
// ARM:#define __CHAR16_TYPE__ unsigned short
// ARM:#define __CHAR32_TYPE__ unsigned int
// ARM:#define __CHAR_BIT__ 8
// ARM:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// ARM:#define __DBL_DIG__ 15
// ARM:#define __DBL_EPSILON__ 2.2204460492503131e-16
// ARM:#define __DBL_HAS_DENORM__ 1
// ARM:#define __DBL_HAS_INFINITY__ 1
// ARM:#define __DBL_HAS_QUIET_NAN__ 1
// ARM:#define __DBL_MANT_DIG__ 53
// ARM:#define __DBL_MAX_10_EXP__ 308
// ARM:#define __DBL_MAX_EXP__ 1024
// ARM:#define __DBL_MAX__ 1.7976931348623157e+308
// ARM:#define __DBL_MIN_10_EXP__ (-307)
// ARM:#define __DBL_MIN_EXP__ (-1021)
// ARM:#define __DBL_MIN__ 2.2250738585072014e-308
// ARM:#define __DECIMAL_DIG__ 17
// ARM:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// ARM:#define __FLT_DIG__ 6
// ARM:#define __FLT_EPSILON__ 1.19209290e-7F
// ARM:#define __FLT_EVAL_METHOD__ 0
// ARM:#define __FLT_HAS_DENORM__ 1
// ARM:#define __FLT_HAS_INFINITY__ 1
// ARM:#define __FLT_HAS_QUIET_NAN__ 1
// ARM:#define __FLT_MANT_DIG__ 24
// ARM:#define __FLT_MAX_10_EXP__ 38
// ARM:#define __FLT_MAX_EXP__ 128
// ARM:#define __FLT_MAX__ 3.40282347e+38F
// ARM:#define __FLT_MIN_10_EXP__ (-37)
// ARM:#define __FLT_MIN_EXP__ (-125)
// ARM:#define __FLT_MIN__ 1.17549435e-38F
// ARM:#define __FLT_RADIX__ 2
// ARM:#define __INT16_TYPE__ short
// ARM:#define __INT32_TYPE__ int
// ARM:#define __INT64_C_SUFFIX__ LL
// ARM:#define __INT64_TYPE__ long long int
// ARM:#define __INT8_TYPE__ char
// ARM:#define __INTMAX_MAX__ 9223372036854775807LL
// ARM:#define __INTMAX_TYPE__ long long int
// ARM:#define __INTMAX_WIDTH__ 64
// ARM:#define __INTPTR_TYPE__ long int
// ARM:#define __INTPTR_WIDTH__ 32
// ARM:#define __INT_MAX__ 2147483647
// ARM:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// ARM:#define __LDBL_DIG__ 15
// ARM:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// ARM:#define __LDBL_HAS_DENORM__ 1
// ARM:#define __LDBL_HAS_INFINITY__ 1
// ARM:#define __LDBL_HAS_QUIET_NAN__ 1
// ARM:#define __LDBL_MANT_DIG__ 53
// ARM:#define __LDBL_MAX_10_EXP__ 308
// ARM:#define __LDBL_MAX_EXP__ 1024
// ARM:#define __LDBL_MAX__ 1.7976931348623157e+308
// ARM:#define __LDBL_MIN_10_EXP__ (-307)
// ARM:#define __LDBL_MIN_EXP__ (-1021)
// ARM:#define __LDBL_MIN__ 2.2250738585072014e-308
// ARM:#define __LITTLE_ENDIAN__ 1
// ARM:#define __LONG_LONG_MAX__ 9223372036854775807LL
// ARM:#define __LONG_MAX__ 2147483647L
// ARM:#define __NO_INLINE__ 1
// ARM:#define __POINTER_WIDTH__ 32
// ARM:#define __PTRDIFF_TYPE__ int
// ARM:#define __PTRDIFF_WIDTH__ 32
// ARM:#define __REGISTER_PREFIX__
// ARM:#define __SCHAR_MAX__ 127
// ARM:#define __SHRT_MAX__ 32767
// ARM:#define __SIG_ATOMIC_WIDTH__ 32
// ARM:#define __SIZEOF_DOUBLE__ 8
// ARM:#define __SIZEOF_FLOAT__ 4
// ARM:#define __SIZEOF_INT__ 4
// ARM:#define __SIZEOF_LONG_DOUBLE__ 8
// ARM:#define __SIZEOF_LONG_LONG__ 8
// ARM:#define __SIZEOF_LONG__ 4
// ARM:#define __SIZEOF_POINTER__ 4
// ARM:#define __SIZEOF_PTRDIFF_T__ 4
// ARM:#define __SIZEOF_SHORT__ 2
// ARM:#define __SIZEOF_SIZE_T__ 4
// ARM:#define __SIZEOF_WCHAR_T__ 4
// ARM:#define __SIZEOF_WINT_T__ 4
// ARM:#define __SIZE_TYPE__ unsigned int
// ARM:#define __SIZE_WIDTH__ 32
// ARM:#define __THUMB_INTERWORK__ 1
// ARM:#define __UINTMAX_TYPE__ long long unsigned int
// ARM:#define __USER_LABEL_PREFIX__ _
// ARM:#define __WCHAR_MAX__ 2147483647
// ARM:#define __WCHAR_TYPE__ int
// ARM:#define __WCHAR_WIDTH__ 32
// ARM:#define __WINT_TYPE__ int
// ARM:#define __WINT_WIDTH__ 32
// ARM:#define __arm 1
// ARM:#define __arm__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=bfin-none-none < /dev/null | FileCheck -check-prefix BFIN %s
//
// BFIN:#define BFIN 1
// BFIN:#define __ADSPBLACKFIN__ 1
// BFIN:#define __ADSPLPBLACKFIN__ 1
// BFIN:#define __BFIN 1
// BFIN:#define __BFIN__ 1
// BFIN:#define __CHAR16_TYPE__ unsigned short
// BFIN:#define __CHAR32_TYPE__ unsigned int
// BFIN:#define __CHAR_BIT__ 8
// BFIN:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// BFIN:#define __DBL_DIG__ 15
// BFIN:#define __DBL_EPSILON__ 2.2204460492503131e-16
// BFIN:#define __DBL_HAS_DENORM__ 1
// BFIN:#define __DBL_HAS_INFINITY__ 1
// BFIN:#define __DBL_HAS_QUIET_NAN__ 1
// BFIN:#define __DBL_MANT_DIG__ 53
// BFIN:#define __DBL_MAX_10_EXP__ 308
// BFIN:#define __DBL_MAX_EXP__ 1024
// BFIN:#define __DBL_MAX__ 1.7976931348623157e+308
// BFIN:#define __DBL_MIN_10_EXP__ (-307)
// BFIN:#define __DBL_MIN_EXP__ (-1021)
// BFIN:#define __DBL_MIN__ 2.2250738585072014e-308
// BFIN:#define __DECIMAL_DIG__ 17
// BFIN:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// BFIN:#define __FLT_DIG__ 6
// BFIN:#define __FLT_EPSILON__ 1.19209290e-7F
// BFIN:#define __FLT_EVAL_METHOD__ 0
// BFIN:#define __FLT_HAS_DENORM__ 1
// BFIN:#define __FLT_HAS_INFINITY__ 1
// BFIN:#define __FLT_HAS_QUIET_NAN__ 1
// BFIN:#define __FLT_MANT_DIG__ 24
// BFIN:#define __FLT_MAX_10_EXP__ 38
// BFIN:#define __FLT_MAX_EXP__ 128
// BFIN:#define __FLT_MAX__ 3.40282347e+38F
// BFIN:#define __FLT_MIN_10_EXP__ (-37)
// BFIN:#define __FLT_MIN_EXP__ (-125)
// BFIN:#define __FLT_MIN__ 1.17549435e-38F
// BFIN:#define __FLT_RADIX__ 2
// BFIN:#define __INT16_TYPE__ short
// BFIN:#define __INT32_TYPE__ int
// BFIN:#define __INT64_C_SUFFIX__ LL
// BFIN:#define __INT64_TYPE__ long long int
// BFIN:#define __INT8_TYPE__ char
// BFIN:#define __INTMAX_MAX__ 9223372036854775807LL
// BFIN:#define __INTMAX_TYPE__ long long int
// BFIN:#define __INTMAX_WIDTH__ 64
// BFIN:#define __INTPTR_TYPE__ long int
// BFIN:#define __INTPTR_WIDTH__ 32
// BFIN:#define __INT_MAX__ 2147483647
// BFIN:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// BFIN:#define __LDBL_DIG__ 15
// BFIN:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// BFIN:#define __LDBL_HAS_DENORM__ 1
// BFIN:#define __LDBL_HAS_INFINITY__ 1
// BFIN:#define __LDBL_HAS_QUIET_NAN__ 1
// BFIN:#define __LDBL_MANT_DIG__ 53
// BFIN:#define __LDBL_MAX_10_EXP__ 308
// BFIN:#define __LDBL_MAX_EXP__ 1024
// BFIN:#define __LDBL_MAX__ 1.7976931348623157e+308
// BFIN:#define __LDBL_MIN_10_EXP__ (-307)
// BFIN:#define __LDBL_MIN_EXP__ (-1021)
// BFIN:#define __LDBL_MIN__ 2.2250738585072014e-308
// BFIN:#define __LONG_LONG_MAX__ 9223372036854775807LL
// BFIN:#define __LONG_MAX__ 2147483647L
// BFIN:#define __NO_INLINE__ 1
// BFIN:#define __POINTER_WIDTH__ 32
// BFIN:#define __PTRDIFF_TYPE__ long int
// BFIN:#define __PTRDIFF_WIDTH__ 32
// BFIN:#define __SCHAR_MAX__ 127
// BFIN:#define __SHRT_MAX__ 32767
// BFIN:#define __SIG_ATOMIC_WIDTH__ 32
// BFIN:#define __SIZEOF_DOUBLE__ 8
// BFIN:#define __SIZEOF_FLOAT__ 4
// BFIN:#define __SIZEOF_INT__ 4
// BFIN:#define __SIZEOF_LONG_DOUBLE__ 8
// BFIN:#define __SIZEOF_LONG_LONG__ 8
// BFIN:#define __SIZEOF_LONG__ 4
// BFIN:#define __SIZEOF_POINTER__ 4
// BFIN:#define __SIZEOF_PTRDIFF_T__ 4
// BFIN:#define __SIZEOF_SHORT__ 2
// BFIN:#define __SIZEOF_SIZE_T__ 4
// BFIN:#define __SIZEOF_WCHAR_T__ 4
// BFIN:#define __SIZEOF_WINT_T__ 4
// BFIN:#define __SIZE_TYPE__ long unsigned int
// BFIN:#define __SIZE_WIDTH__ 32
// BFIN:#define __UINTMAX_TYPE__ long long unsigned int
// BFIN:#define __USER_LABEL_PREFIX__ _
// BFIN:#define __WCHAR_MAX__ 2147483647
// BFIN:#define __WCHAR_TYPE__ int
// BFIN:#define __WCHAR_WIDTH__ 32
// BFIN:#define __WINT_TYPE__ int
// BFIN:#define __WINT_WIDTH__ 32
// BFIN:#define __bfin 1
// BFIN:#define __bfin__ 1
// BFIN:#define bfin 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-none-none < /dev/null | FileCheck -check-prefix I386 %s
//
// I386:#define __CHAR16_TYPE__ unsigned short
// I386:#define __CHAR32_TYPE__ unsigned int
// I386:#define __CHAR_BIT__ 8
// I386:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// I386:#define __DBL_DIG__ 15
// I386:#define __DBL_EPSILON__ 2.2204460492503131e-16
// I386:#define __DBL_HAS_DENORM__ 1
// I386:#define __DBL_HAS_INFINITY__ 1
// I386:#define __DBL_HAS_QUIET_NAN__ 1
// I386:#define __DBL_MANT_DIG__ 53
// I386:#define __DBL_MAX_10_EXP__ 308
// I386:#define __DBL_MAX_EXP__ 1024
// I386:#define __DBL_MAX__ 1.7976931348623157e+308
// I386:#define __DBL_MIN_10_EXP__ (-307)
// I386:#define __DBL_MIN_EXP__ (-1021)
// I386:#define __DBL_MIN__ 2.2250738585072014e-308
// I386:#define __DECIMAL_DIG__ 21
// I386:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// I386:#define __FLT_DIG__ 6
// I386:#define __FLT_EPSILON__ 1.19209290e-7F
// I386:#define __FLT_EVAL_METHOD__ 0
// I386:#define __FLT_HAS_DENORM__ 1
// I386:#define __FLT_HAS_INFINITY__ 1
// I386:#define __FLT_HAS_QUIET_NAN__ 1
// I386:#define __FLT_MANT_DIG__ 24
// I386:#define __FLT_MAX_10_EXP__ 38
// I386:#define __FLT_MAX_EXP__ 128
// I386:#define __FLT_MAX__ 3.40282347e+38F
// I386:#define __FLT_MIN_10_EXP__ (-37)
// I386:#define __FLT_MIN_EXP__ (-125)
// I386:#define __FLT_MIN__ 1.17549435e-38F
// I386:#define __FLT_RADIX__ 2
// I386:#define __INT16_TYPE__ short
// I386:#define __INT32_TYPE__ int
// I386:#define __INT64_C_SUFFIX__ LL
// I386:#define __INT64_TYPE__ long long int
// I386:#define __INT8_TYPE__ char
// I386:#define __INTMAX_MAX__ 9223372036854775807LL
// I386:#define __INTMAX_TYPE__ long long int
// I386:#define __INTMAX_WIDTH__ 64
// I386:#define __INTPTR_TYPE__ int
// I386:#define __INTPTR_WIDTH__ 32
// I386:#define __INT_MAX__ 2147483647
// I386:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// I386:#define __LDBL_DIG__ 18
// I386:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// I386:#define __LDBL_HAS_DENORM__ 1
// I386:#define __LDBL_HAS_INFINITY__ 1
// I386:#define __LDBL_HAS_QUIET_NAN__ 1
// I386:#define __LDBL_MANT_DIG__ 64
// I386:#define __LDBL_MAX_10_EXP__ 4932
// I386:#define __LDBL_MAX_EXP__ 16384
// I386:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// I386:#define __LDBL_MIN_10_EXP__ (-4931)
// I386:#define __LDBL_MIN_EXP__ (-16381)
// I386:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// I386:#define __LITTLE_ENDIAN__ 1
// I386:#define __LONG_LONG_MAX__ 9223372036854775807LL
// I386:#define __LONG_MAX__ 2147483647L
// I386:#define __NO_INLINE__ 1
// I386:#define __NO_MATH_INLINES 1
// I386:#define __POINTER_WIDTH__ 32
// I386:#define __PTRDIFF_TYPE__ int
// I386:#define __PTRDIFF_WIDTH__ 32
// I386:#define __REGISTER_PREFIX__ 
// I386:#define __SCHAR_MAX__ 127
// I386:#define __SHRT_MAX__ 32767
// I386:#define __SIG_ATOMIC_WIDTH__ 32
// I386:#define __SIZEOF_DOUBLE__ 8
// I386:#define __SIZEOF_FLOAT__ 4
// I386:#define __SIZEOF_INT__ 4
// I386:#define __SIZEOF_LONG_DOUBLE__ 12
// I386:#define __SIZEOF_LONG_LONG__ 8
// I386:#define __SIZEOF_LONG__ 4
// I386:#define __SIZEOF_POINTER__ 4
// I386:#define __SIZEOF_PTRDIFF_T__ 4
// I386:#define __SIZEOF_SHORT__ 2
// I386:#define __SIZEOF_SIZE_T__ 4
// I386:#define __SIZEOF_WCHAR_T__ 4
// I386:#define __SIZEOF_WINT_T__ 4
// I386:#define __SIZE_TYPE__ unsigned int
// I386:#define __SIZE_WIDTH__ 32
// I386:#define __UINTMAX_TYPE__ long long unsigned int
// I386:#define __USER_LABEL_PREFIX__ _
// I386:#define __WCHAR_MAX__ 2147483647
// I386:#define __WCHAR_TYPE__ int
// I386:#define __WCHAR_WIDTH__ 32
// I386:#define __WINT_TYPE__ int
// I386:#define __WINT_WIDTH__ 32
// I386:#define __i386 1
// I386:#define __i386__ 1
// I386:#define __nocona 1
// I386:#define __nocona__ 1
// I386:#define __tune_nocona__ 1
// I386:#define i386 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-pc-linux-gnu < /dev/null | FileCheck -check-prefix I386-LINUX %s
//
// I386-LINUX:#define __CHAR16_TYPE__ unsigned short
// I386-LINUX:#define __CHAR32_TYPE__ unsigned int
// I386-LINUX:#define __CHAR_BIT__ 8
// I386-LINUX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// I386-LINUX:#define __DBL_DIG__ 15
// I386-LINUX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// I386-LINUX:#define __DBL_HAS_DENORM__ 1
// I386-LINUX:#define __DBL_HAS_INFINITY__ 1
// I386-LINUX:#define __DBL_HAS_QUIET_NAN__ 1
// I386-LINUX:#define __DBL_MANT_DIG__ 53
// I386-LINUX:#define __DBL_MAX_10_EXP__ 308
// I386-LINUX:#define __DBL_MAX_EXP__ 1024
// I386-LINUX:#define __DBL_MAX__ 1.7976931348623157e+308
// I386-LINUX:#define __DBL_MIN_10_EXP__ (-307)
// I386-LINUX:#define __DBL_MIN_EXP__ (-1021)
// I386-LINUX:#define __DBL_MIN__ 2.2250738585072014e-308
// I386-LINUX:#define __DECIMAL_DIG__ 21
// I386-LINUX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// I386-LINUX:#define __FLT_DIG__ 6
// I386-LINUX:#define __FLT_EPSILON__ 1.19209290e-7F
// I386-LINUX:#define __FLT_EVAL_METHOD__ 0
// I386-LINUX:#define __FLT_HAS_DENORM__ 1
// I386-LINUX:#define __FLT_HAS_INFINITY__ 1
// I386-LINUX:#define __FLT_HAS_QUIET_NAN__ 1
// I386-LINUX:#define __FLT_MANT_DIG__ 24
// I386-LINUX:#define __FLT_MAX_10_EXP__ 38
// I386-LINUX:#define __FLT_MAX_EXP__ 128
// I386-LINUX:#define __FLT_MAX__ 3.40282347e+38F
// I386-LINUX:#define __FLT_MIN_10_EXP__ (-37)
// I386-LINUX:#define __FLT_MIN_EXP__ (-125)
// I386-LINUX:#define __FLT_MIN__ 1.17549435e-38F
// I386-LINUX:#define __FLT_RADIX__ 2
// I386-LINUX:#define __INT16_TYPE__ short
// I386-LINUX:#define __INT32_TYPE__ int
// I386-LINUX:#define __INT64_C_SUFFIX__ LL
// I386-LINUX:#define __INT64_TYPE__ long long int
// I386-LINUX:#define __INT8_TYPE__ char
// I386-LINUX:#define __INTMAX_MAX__ 9223372036854775807LL
// I386-LINUX:#define __INTMAX_TYPE__ long long int
// I386-LINUX:#define __INTMAX_WIDTH__ 64
// I386-LINUX:#define __INTPTR_TYPE__ int
// I386-LINUX:#define __INTPTR_WIDTH__ 32
// I386-LINUX:#define __INT_MAX__ 2147483647
// I386-LINUX:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// I386-LINUX:#define __LDBL_DIG__ 18
// I386-LINUX:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// I386-LINUX:#define __LDBL_HAS_DENORM__ 1
// I386-LINUX:#define __LDBL_HAS_INFINITY__ 1
// I386-LINUX:#define __LDBL_HAS_QUIET_NAN__ 1
// I386-LINUX:#define __LDBL_MANT_DIG__ 64
// I386-LINUX:#define __LDBL_MAX_10_EXP__ 4932
// I386-LINUX:#define __LDBL_MAX_EXP__ 16384
// I386-LINUX:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// I386-LINUX:#define __LDBL_MIN_10_EXP__ (-4931)
// I386-LINUX:#define __LDBL_MIN_EXP__ (-16381)
// I386-LINUX:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// I386-LINUX:#define __LITTLE_ENDIAN__ 1
// I386-LINUX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// I386-LINUX:#define __LONG_MAX__ 2147483647L
// I386-LINUX:#define __NO_INLINE__ 1
// I386-LINUX:#define __NO_MATH_INLINES 1
// I386-LINUX:#define __POINTER_WIDTH__ 32
// I386-LINUX:#define __PTRDIFF_TYPE__ int
// I386-LINUX:#define __PTRDIFF_WIDTH__ 32
// I386-LINUX:#define __REGISTER_PREFIX__ 
// I386-LINUX:#define __SCHAR_MAX__ 127
// I386-LINUX:#define __SHRT_MAX__ 32767
// I386-LINUX:#define __SIG_ATOMIC_WIDTH__ 32
// I386-LINUX:#define __SIZEOF_DOUBLE__ 8
// I386-LINUX:#define __SIZEOF_FLOAT__ 4
// I386-LINUX:#define __SIZEOF_INT__ 4
// I386-LINUX:#define __SIZEOF_LONG_DOUBLE__ 12
// I386-LINUX:#define __SIZEOF_LONG_LONG__ 8
// I386-LINUX:#define __SIZEOF_LONG__ 4
// I386-LINUX:#define __SIZEOF_POINTER__ 4
// I386-LINUX:#define __SIZEOF_PTRDIFF_T__ 4
// I386-LINUX:#define __SIZEOF_SHORT__ 2
// I386-LINUX:#define __SIZEOF_SIZE_T__ 4
// I386-LINUX:#define __SIZEOF_WCHAR_T__ 4
// I386-LINUX:#define __SIZEOF_WINT_T__ 4
// I386-LINUX:#define __SIZE_TYPE__ unsigned int
// I386-LINUX:#define __SIZE_WIDTH__ 32
// I386-LINUX:#define __UINTMAX_TYPE__ long long unsigned int
// I386-LINUX:#define __USER_LABEL_PREFIX__
// I386-LINUX:#define __WCHAR_MAX__ 2147483647
// I386-LINUX:#define __WCHAR_TYPE__ int
// I386-LINUX:#define __WCHAR_WIDTH__ 32
// I386-LINUX:#define __WINT_TYPE__ unsigned int
// I386-LINUX:#define __WINT_WIDTH__ 32
// I386-LINUX:#define __i386 1
// I386-LINUX:#define __i386__ 1
// I386-LINUX:#define __nocona 1
// I386-LINUX:#define __nocona__ 1
// I386-LINUX:#define __tune_nocona__ 1
// I386-LINUX:#define i386 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=msp430-none-none < /dev/null | FileCheck -check-prefix MSP430 %s
//
// MSP430:#define MSP430 1
// MSP430:#define __CHAR16_TYPE__ unsigned short
// MSP430:#define __CHAR32_TYPE__ unsigned int
// MSP430:#define __CHAR_BIT__ 8
// MSP430:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MSP430:#define __DBL_DIG__ 15
// MSP430:#define __DBL_EPSILON__ 2.2204460492503131e-16
// MSP430:#define __DBL_HAS_DENORM__ 1
// MSP430:#define __DBL_HAS_INFINITY__ 1
// MSP430:#define __DBL_HAS_QUIET_NAN__ 1
// MSP430:#define __DBL_MANT_DIG__ 53
// MSP430:#define __DBL_MAX_10_EXP__ 308
// MSP430:#define __DBL_MAX_EXP__ 1024
// MSP430:#define __DBL_MAX__ 1.7976931348623157e+308
// MSP430:#define __DBL_MIN_10_EXP__ (-307)
// MSP430:#define __DBL_MIN_EXP__ (-1021)
// MSP430:#define __DBL_MIN__ 2.2250738585072014e-308
// MSP430:#define __DECIMAL_DIG__ 17
// MSP430:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// MSP430:#define __FLT_DIG__ 6
// MSP430:#define __FLT_EPSILON__ 1.19209290e-7F
// MSP430:#define __FLT_EVAL_METHOD__ 0
// MSP430:#define __FLT_HAS_DENORM__ 1
// MSP430:#define __FLT_HAS_INFINITY__ 1
// MSP430:#define __FLT_HAS_QUIET_NAN__ 1
// MSP430:#define __FLT_MANT_DIG__ 24
// MSP430:#define __FLT_MAX_10_EXP__ 38
// MSP430:#define __FLT_MAX_EXP__ 128
// MSP430:#define __FLT_MAX__ 3.40282347e+38F
// MSP430:#define __FLT_MIN_10_EXP__ (-37)
// MSP430:#define __FLT_MIN_EXP__ (-125)
// MSP430:#define __FLT_MIN__ 1.17549435e-38F
// MSP430:#define __FLT_RADIX__ 2
// MSP430:#define __INT16_TYPE__ short
// MSP430:#define __INT32_C_SUFFIX__ L
// MSP430:#define __INT32_TYPE__ long int
// MSP430:#define __INT8_TYPE__ char
// MSP430:#define __INTMAX_MAX__ 2147483647L
// MSP430:#define __INTMAX_TYPE__ long int
// MSP430:#define __INTMAX_WIDTH__ 32
// MSP430:#define __INTPTR_TYPE__ short
// MSP430:#define __INTPTR_WIDTH__ 16
// MSP430:#define __INT_MAX__ 32767
// MSP430:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// MSP430:#define __LDBL_DIG__ 15
// MSP430:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// MSP430:#define __LDBL_HAS_DENORM__ 1
// MSP430:#define __LDBL_HAS_INFINITY__ 1
// MSP430:#define __LDBL_HAS_QUIET_NAN__ 1
// MSP430:#define __LDBL_MANT_DIG__ 53
// MSP430:#define __LDBL_MAX_10_EXP__ 308
// MSP430:#define __LDBL_MAX_EXP__ 1024
// MSP430:#define __LDBL_MAX__ 1.7976931348623157e+308
// MSP430:#define __LDBL_MIN_10_EXP__ (-307)
// MSP430:#define __LDBL_MIN_EXP__ (-1021)
// MSP430:#define __LDBL_MIN__ 2.2250738585072014e-308
// MSP430:#define __LONG_LONG_MAX__ 9223372036854775807LL
// MSP430:#define __LONG_MAX__ 2147483647L
// MSP430:#define __MSP430__ 1
// MSP430:#define __NO_INLINE__ 1
// MSP430:#define __POINTER_WIDTH__ 16
// MSP430:#define __PTRDIFF_TYPE__ int
// MSP430:#define __PTRDIFF_WIDTH__ 16 
// MSP430:#define __SCHAR_MAX__ 127
// MSP430:#define __SHRT_MAX__ 32767
// MSP430:#define __SIG_ATOMIC_WIDTH__ 32
// MSP430:#define __SIZEOF_DOUBLE__ 8
// MSP430:#define __SIZEOF_FLOAT__ 4
// MSP430:#define __SIZEOF_INT__ 2
// MSP430:#define __SIZEOF_LONG_DOUBLE__ 8
// MSP430:#define __SIZEOF_LONG_LONG__ 8
// MSP430:#define __SIZEOF_LONG__ 4
// MSP430:#define __SIZEOF_POINTER__ 2
// MSP430:#define __SIZEOF_PTRDIFF_T__ 2
// MSP430:#define __SIZEOF_SHORT__ 2
// MSP430:#define __SIZEOF_SIZE_T__ 2
// MSP430:#define __SIZEOF_WCHAR_T__ 2
// MSP430:#define __SIZEOF_WINT_T__ 2
// MSP430:#define __SIZE_TYPE__ unsigned int
// MSP430:#define __SIZE_WIDTH__ 16
// MSP430:#define __UINTMAX_TYPE__ long unsigned int
// MSP430:#define __USER_LABEL_PREFIX__ _
// MSP430:#define __WCHAR_MAX__ 32767
// MSP430:#define __WCHAR_TYPE__ int
// MSP430:#define __WCHAR_WIDTH__ 16
// MSP430:#define __WINT_TYPE__ int
// MSP430:#define __WINT_WIDTH__ 16
// MSP430:#define __clang__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -fno-signed-char < /dev/null | FileCheck -check-prefix PPC64 %s
//
// PPC64:#define _ARCH_PPC 1
// PPC64:#define _ARCH_PPC64 1
// PPC64:#define _BIG_ENDIAN 1
// PPC64:#define _LP64 1
// PPC64:#define __BIG_ENDIAN__ 1
// PPC64:#define __CHAR16_TYPE__ unsigned short
// PPC64:#define __CHAR32_TYPE__ unsigned int
// PPC64:#define __CHAR_BIT__ 8
// PPC64:#define __CHAR_UNSIGNED__ 1
// PPC64:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC64:#define __DBL_DIG__ 15
// PPC64:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC64:#define __DBL_HAS_DENORM__ 1
// PPC64:#define __DBL_HAS_INFINITY__ 1
// PPC64:#define __DBL_HAS_QUIET_NAN__ 1
// PPC64:#define __DBL_MANT_DIG__ 53
// PPC64:#define __DBL_MAX_10_EXP__ 308
// PPC64:#define __DBL_MAX_EXP__ 1024
// PPC64:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC64:#define __DBL_MIN_10_EXP__ (-307)
// PPC64:#define __DBL_MIN_EXP__ (-1021)
// PPC64:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC64:#define __DECIMAL_DIG__ 17
// PPC64:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC64:#define __FLT_DIG__ 6
// PPC64:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC64:#define __FLT_EVAL_METHOD__ 0
// PPC64:#define __FLT_HAS_DENORM__ 1
// PPC64:#define __FLT_HAS_INFINITY__ 1
// PPC64:#define __FLT_HAS_QUIET_NAN__ 1
// PPC64:#define __FLT_MANT_DIG__ 24
// PPC64:#define __FLT_MAX_10_EXP__ 38
// PPC64:#define __FLT_MAX_EXP__ 128
// PPC64:#define __FLT_MAX__ 3.40282347e+38F
// PPC64:#define __FLT_MIN_10_EXP__ (-37)
// PPC64:#define __FLT_MIN_EXP__ (-125)
// PPC64:#define __FLT_MIN__ 1.17549435e-38F
// PPC64:#define __FLT_RADIX__ 2
// PPC64:#define __INT16_TYPE__ short
// PPC64:#define __INT32_TYPE__ int
// PPC64:#define __INT64_C_SUFFIX__ L
// PPC64:#define __INT64_TYPE__ long int
// PPC64:#define __INT8_TYPE__ char
// PPC64:#define __INTMAX_MAX__ 9223372036854775807L
// PPC64:#define __INTMAX_TYPE__ long int
// PPC64:#define __INTMAX_WIDTH__ 64
// PPC64:#define __INTPTR_TYPE__ long int
// PPC64:#define __INTPTR_WIDTH__ 64
// PPC64:#define __INT_MAX__ 2147483647
// PPC64:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC64:#define __LDBL_DIG__ 15
// PPC64:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// PPC64:#define __LDBL_HAS_DENORM__ 1
// PPC64:#define __LDBL_HAS_INFINITY__ 1
// PPC64:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC64:#define __LDBL_MANT_DIG__ 53
// PPC64:#define __LDBL_MAX_10_EXP__ 308
// PPC64:#define __LDBL_MAX_EXP__ 1024
// PPC64:#define __LDBL_MAX__ 1.7976931348623157e+308
// PPC64:#define __LDBL_MIN_10_EXP__ (-307)
// PPC64:#define __LDBL_MIN_EXP__ (-1021)
// PPC64:#define __LDBL_MIN__ 2.2250738585072014e-308
// PPC64:#define __LONG_DOUBLE_128__ 1
// PPC64:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC64:#define __LONG_MAX__ 9223372036854775807L
// PPC64:#define __LP64__ 1
// PPC64:#define __NATURAL_ALIGNMENT__ 1
// PPC64:#define __NO_INLINE__ 1
// PPC64:#define __POINTER_WIDTH__ 64
// PPC64:#define __POWERPC__ 1
// PPC64:#define __PTRDIFF_TYPE__ long int
// PPC64:#define __PTRDIFF_WIDTH__ 64
// PPC64:#define __REGISTER_PREFIX__ 
// PPC64:#define __SCHAR_MAX__ 127
// PPC64:#define __SHRT_MAX__ 32767
// PPC64:#define __SIG_ATOMIC_WIDTH__ 32
// PPC64:#define __SIZEOF_DOUBLE__ 8
// PPC64:#define __SIZEOF_FLOAT__ 4
// PPC64:#define __SIZEOF_INT__ 4
// PPC64:#define __SIZEOF_LONG_DOUBLE__ 8
// PPC64:#define __SIZEOF_LONG_LONG__ 8
// PPC64:#define __SIZEOF_LONG__ 8
// PPC64:#define __SIZEOF_POINTER__ 8
// PPC64:#define __SIZEOF_PTRDIFF_T__ 8
// PPC64:#define __SIZEOF_SHORT__ 2
// PPC64:#define __SIZEOF_SIZE_T__ 8
// PPC64:#define __SIZEOF_WCHAR_T__ 4
// PPC64:#define __SIZEOF_WINT_T__ 4
// PPC64:#define __SIZE_TYPE__ long unsigned int
// PPC64:#define __SIZE_WIDTH__ 64
// PPC64:#define __UINTMAX_TYPE__ long unsigned int
// PPC64:#define __USER_LABEL_PREFIX__ _
// PPC64:#define __WCHAR_MAX__ 2147483647
// PPC64:#define __WCHAR_TYPE__ int
// PPC64:#define __WCHAR_WIDTH__ 32
// PPC64:#define __WINT_TYPE__ int
// PPC64:#define __WINT_WIDTH__ 32
// PPC64:#define __ppc64__ 1
// PPC64:#define __ppc__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-none-none -fno-signed-char < /dev/null | FileCheck -check-prefix PPC %s
//
// PPC:#define _ARCH_PPC 1
// PPC:#define _BIG_ENDIAN 1
// PPC:#define __BIG_ENDIAN__ 1
// PPC:#define __CHAR16_TYPE__ unsigned short
// PPC:#define __CHAR32_TYPE__ unsigned int
// PPC:#define __CHAR_BIT__ 8
// PPC:#define __CHAR_UNSIGNED__ 1
// PPC:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC:#define __DBL_DIG__ 15
// PPC:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC:#define __DBL_HAS_DENORM__ 1
// PPC:#define __DBL_HAS_INFINITY__ 1
// PPC:#define __DBL_HAS_QUIET_NAN__ 1
// PPC:#define __DBL_MANT_DIG__ 53
// PPC:#define __DBL_MAX_10_EXP__ 308
// PPC:#define __DBL_MAX_EXP__ 1024
// PPC:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC:#define __DBL_MIN_10_EXP__ (-307)
// PPC:#define __DBL_MIN_EXP__ (-1021)
// PPC:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC:#define __DECIMAL_DIG__ 17
// PPC:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC:#define __FLT_DIG__ 6
// PPC:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC:#define __FLT_EVAL_METHOD__ 0
// PPC:#define __FLT_HAS_DENORM__ 1
// PPC:#define __FLT_HAS_INFINITY__ 1
// PPC:#define __FLT_HAS_QUIET_NAN__ 1
// PPC:#define __FLT_MANT_DIG__ 24
// PPC:#define __FLT_MAX_10_EXP__ 38
// PPC:#define __FLT_MAX_EXP__ 128
// PPC:#define __FLT_MAX__ 3.40282347e+38F
// PPC:#define __FLT_MIN_10_EXP__ (-37)
// PPC:#define __FLT_MIN_EXP__ (-125)
// PPC:#define __FLT_MIN__ 1.17549435e-38F
// PPC:#define __FLT_RADIX__ 2
// PPC:#define __INT16_TYPE__ short
// PPC:#define __INT32_TYPE__ int
// PPC:#define __INT64_C_SUFFIX__ LL
// PPC:#define __INT64_TYPE__ long long int
// PPC:#define __INT8_TYPE__ char
// PPC:#define __INTMAX_MAX__ 9223372036854775807LL
// PPC:#define __INTMAX_TYPE__ long long int
// PPC:#define __INTMAX_WIDTH__ 64
// PPC:#define __INTPTR_TYPE__ long int
// PPC:#define __INTPTR_WIDTH__ 32
// PPC:#define __INT_MAX__ 2147483647
// PPC:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC:#define __LDBL_DIG__ 15
// PPC:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// PPC:#define __LDBL_HAS_DENORM__ 1
// PPC:#define __LDBL_HAS_INFINITY__ 1
// PPC:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC:#define __LDBL_MANT_DIG__ 53
// PPC:#define __LDBL_MAX_10_EXP__ 308
// PPC:#define __LDBL_MAX_EXP__ 1024
// PPC:#define __LDBL_MAX__ 1.7976931348623157e+308
// PPC:#define __LDBL_MIN_10_EXP__ (-307)
// PPC:#define __LDBL_MIN_EXP__ (-1021)
// PPC:#define __LDBL_MIN__ 2.2250738585072014e-308
// PPC:#define __LONG_DOUBLE_128__ 1
// PPC:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC:#define __LONG_MAX__ 2147483647L
// PPC:#define __NATURAL_ALIGNMENT__ 1
// PPC:#define __NO_INLINE__ 1
// PPC:#define __POINTER_WIDTH__ 32
// PPC:#define __POWERPC__ 1
// PPC:#define __PTRDIFF_TYPE__ long int
// PPC:#define __PTRDIFF_WIDTH__ 32
// PPC:#define __REGISTER_PREFIX__ 
// PPC:#define __SCHAR_MAX__ 127
// PPC:#define __SHRT_MAX__ 32767
// PPC:#define __SIG_ATOMIC_WIDTH__ 32
// PPC:#define __SIZEOF_DOUBLE__ 8
// PPC:#define __SIZEOF_FLOAT__ 4
// PPC:#define __SIZEOF_INT__ 4
// PPC:#define __SIZEOF_LONG_DOUBLE__ 8
// PPC:#define __SIZEOF_LONG_LONG__ 8
// PPC:#define __SIZEOF_LONG__ 4
// PPC:#define __SIZEOF_POINTER__ 4
// PPC:#define __SIZEOF_PTRDIFF_T__ 4
// PPC:#define __SIZEOF_SHORT__ 2
// PPC:#define __SIZEOF_SIZE_T__ 4
// PPC:#define __SIZEOF_WCHAR_T__ 4
// PPC:#define __SIZEOF_WINT_T__ 4
// PPC:#define __SIZE_TYPE__ long unsigned int
// PPC:#define __SIZE_WIDTH__ 32
// PPC:#define __UINTMAX_TYPE__ long long unsigned int
// PPC:#define __USER_LABEL_PREFIX__ _
// PPC:#define __WCHAR_MAX__ 2147483647
// PPC:#define __WCHAR_TYPE__ int
// PPC:#define __WCHAR_WIDTH__ 32
// PPC:#define __WINT_TYPE__ int
// PPC:#define __WINT_WIDTH__ 32
// PPC:#define __ppc__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=s390x-none-none -fno-signed-char < /dev/null | FileCheck -check-prefix S390X %s
//
// S390X:#define __CHAR16_TYPE__ unsigned short
// S390X:#define __CHAR32_TYPE__ unsigned int
// S390X:#define __CHAR_BIT__ 8
// S390X:#define __CHAR_UNSIGNED__ 1
// S390X:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// S390X:#define __DBL_DIG__ 15
// S390X:#define __DBL_EPSILON__ 2.2204460492503131e-16
// S390X:#define __DBL_HAS_DENORM__ 1
// S390X:#define __DBL_HAS_INFINITY__ 1
// S390X:#define __DBL_HAS_QUIET_NAN__ 1
// S390X:#define __DBL_MANT_DIG__ 53
// S390X:#define __DBL_MAX_10_EXP__ 308
// S390X:#define __DBL_MAX_EXP__ 1024
// S390X:#define __DBL_MAX__ 1.7976931348623157e+308
// S390X:#define __DBL_MIN_10_EXP__ (-307)
// S390X:#define __DBL_MIN_EXP__ (-1021)
// S390X:#define __DBL_MIN__ 2.2250738585072014e-308
// S390X:#define __DECIMAL_DIG__ 17
// S390X:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// S390X:#define __FLT_DIG__ 6
// S390X:#define __FLT_EPSILON__ 1.19209290e-7F
// S390X:#define __FLT_EVAL_METHOD__ 0
// S390X:#define __FLT_HAS_DENORM__ 1
// S390X:#define __FLT_HAS_INFINITY__ 1
// S390X:#define __FLT_HAS_QUIET_NAN__ 1
// S390X:#define __FLT_MANT_DIG__ 24
// S390X:#define __FLT_MAX_10_EXP__ 38
// S390X:#define __FLT_MAX_EXP__ 128
// S390X:#define __FLT_MAX__ 3.40282347e+38F
// S390X:#define __FLT_MIN_10_EXP__ (-37)
// S390X:#define __FLT_MIN_EXP__ (-125)
// S390X:#define __FLT_MIN__ 1.17549435e-38F
// S390X:#define __FLT_RADIX__ 2
// S390X:#define __INT16_TYPE__ short
// S390X:#define __INT32_TYPE__ int
// S390X:#define __INT64_C_SUFFIX__ L
// S390X:#define __INT64_TYPE__ long long int
// S390X:#define __INT8_TYPE__ char
// S390X:#define __INTMAX_MAX__ 9223372036854775807LL
// S390X:#define __INTMAX_TYPE__ long long int
// S390X:#define __INTMAX_WIDTH__ 64
// S390X:#define __INTPTR_TYPE__ long int
// S390X:#define __INTPTR_WIDTH__ 64
// S390X:#define __INT_MAX__ 2147483647
// S390X:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// S390X:#define __LDBL_DIG__ 15
// S390X:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// S390X:#define __LDBL_HAS_DENORM__ 1
// S390X:#define __LDBL_HAS_INFINITY__ 1
// S390X:#define __LDBL_HAS_QUIET_NAN__ 1
// S390X:#define __LDBL_MANT_DIG__ 53
// S390X:#define __LDBL_MAX_10_EXP__ 308
// S390X:#define __LDBL_MAX_EXP__ 1024
// S390X:#define __LDBL_MAX__ 1.7976931348623157e+308
// S390X:#define __LDBL_MIN_10_EXP__ (-307)
// S390X:#define __LDBL_MIN_EXP__ (-1021)
// S390X:#define __LDBL_MIN__ 2.2250738585072014e-308
// S390X:#define __LONG_LONG_MAX__ 9223372036854775807LL
// S390X:#define __LONG_MAX__ 9223372036854775807L
// S390X:#define __NO_INLINE__ 1
// S390X:#define __POINTER_WIDTH__ 64
// S390X:#define __PTRDIFF_TYPE__ long int
// S390X:#define __PTRDIFF_WIDTH__ 64
// S390X:#define __SCHAR_MAX__ 127
// S390X:#define __SHRT_MAX__ 32767
// S390X:#define __SIG_ATOMIC_WIDTH__ 32
// S390X:#define __SIZEOF_DOUBLE__ 8
// S390X:#define __SIZEOF_FLOAT__ 4
// S390X:#define __SIZEOF_INT__ 4
// S390X:#define __SIZEOF_LONG_DOUBLE__ 8
// S390X:#define __SIZEOF_LONG_LONG__ 8
// S390X:#define __SIZEOF_LONG__ 8
// S390X:#define __SIZEOF_POINTER__ 8
// S390X:#define __SIZEOF_PTRDIFF_T__ 8
// S390X:#define __SIZEOF_SHORT__ 2
// S390X:#define __SIZEOF_SIZE_T__ 8
// S390X:#define __SIZEOF_WCHAR_T__ 4
// S390X:#define __SIZEOF_WINT_T__ 4
// S390X:#define __SIZE_TYPE__ long unsigned int
// S390X:#define __SIZE_WIDTH__ 64
// S390X:#define __UINTMAX_TYPE__ long long unsigned int
// S390X:#define __USER_LABEL_PREFIX__ _
// S390X:#define __WCHAR_MAX__ 2147483647
// S390X:#define __WCHAR_TYPE__ int
// S390X:#define __WCHAR_WIDTH__ 32
// S390X:#define __WINT_TYPE__ int
// S390X:#define __WINT_WIDTH__ 32
// S390X:#define __s390__ 1
// S390X:#define __s390x__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc-none-none < /dev/null | FileCheck -check-prefix SPARC %s
//
// SPARC:#define __CHAR16_TYPE__ unsigned short
// SPARC:#define __CHAR32_TYPE__ unsigned int
// SPARC:#define __CHAR_BIT__ 8
// SPARC:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// SPARC:#define __DBL_DIG__ 15
// SPARC:#define __DBL_EPSILON__ 2.2204460492503131e-16
// SPARC:#define __DBL_HAS_DENORM__ 1
// SPARC:#define __DBL_HAS_INFINITY__ 1
// SPARC:#define __DBL_HAS_QUIET_NAN__ 1
// SPARC:#define __DBL_MANT_DIG__ 53
// SPARC:#define __DBL_MAX_10_EXP__ 308
// SPARC:#define __DBL_MAX_EXP__ 1024
// SPARC:#define __DBL_MAX__ 1.7976931348623157e+308
// SPARC:#define __DBL_MIN_10_EXP__ (-307)
// SPARC:#define __DBL_MIN_EXP__ (-1021)
// SPARC:#define __DBL_MIN__ 2.2250738585072014e-308
// SPARC:#define __DECIMAL_DIG__ 17
// SPARC:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// SPARC:#define __FLT_DIG__ 6
// SPARC:#define __FLT_EPSILON__ 1.19209290e-7F
// SPARC:#define __FLT_EVAL_METHOD__ 0
// SPARC:#define __FLT_HAS_DENORM__ 1
// SPARC:#define __FLT_HAS_INFINITY__ 1
// SPARC:#define __FLT_HAS_QUIET_NAN__ 1
// SPARC:#define __FLT_MANT_DIG__ 24
// SPARC:#define __FLT_MAX_10_EXP__ 38
// SPARC:#define __FLT_MAX_EXP__ 128
// SPARC:#define __FLT_MAX__ 3.40282347e+38F
// SPARC:#define __FLT_MIN_10_EXP__ (-37)
// SPARC:#define __FLT_MIN_EXP__ (-125)
// SPARC:#define __FLT_MIN__ 1.17549435e-38F
// SPARC:#define __FLT_RADIX__ 2
// SPARC:#define __INT16_TYPE__ short
// SPARC:#define __INT32_TYPE__ int
// SPARC:#define __INT64_C_SUFFIX__ LL
// SPARC:#define __INT64_TYPE__ long long int
// SPARC:#define __INT8_TYPE__ char
// SPARC:#define __INTMAX_MAX__ 9223372036854775807LL
// SPARC:#define __INTMAX_TYPE__ long long int
// SPARC:#define __INTMAX_WIDTH__ 64
// SPARC:#define __INTPTR_TYPE__ long int
// SPARC:#define __INTPTR_WIDTH__ 32
// SPARC:#define __INT_MAX__ 2147483647
// SPARC:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324
// SPARC:#define __LDBL_DIG__ 15
// SPARC:#define __LDBL_EPSILON__ 2.2204460492503131e-16
// SPARC:#define __LDBL_HAS_DENORM__ 1
// SPARC:#define __LDBL_HAS_INFINITY__ 1
// SPARC:#define __LDBL_HAS_QUIET_NAN__ 1
// SPARC:#define __LDBL_MANT_DIG__ 53
// SPARC:#define __LDBL_MAX_10_EXP__ 308
// SPARC:#define __LDBL_MAX_EXP__ 1024
// SPARC:#define __LDBL_MAX__ 1.7976931348623157e+308
// SPARC:#define __LDBL_MIN_10_EXP__ (-307)
// SPARC:#define __LDBL_MIN_EXP__ (-1021)
// SPARC:#define __LDBL_MIN__ 2.2250738585072014e-308
// SPARC:#define __LONG_LONG_MAX__ 9223372036854775807LL
// SPARC:#define __LONG_MAX__ 2147483647L
// SPARC:#define __NO_INLINE__ 1
// SPARC:#define __POINTER_WIDTH__ 32
// SPARC:#define __PTRDIFF_TYPE__ long int
// SPARC:#define __PTRDIFF_WIDTH__ 32
// SPARC:#define __REGISTER_PREFIX__
// SPARC:#define __SCHAR_MAX__ 127
// SPARC:#define __SHRT_MAX__ 32767
// SPARC:#define __SIG_ATOMIC_WIDTH__ 32
// SPARC:#define __SIZEOF_DOUBLE__ 8
// SPARC:#define __SIZEOF_FLOAT__ 4
// SPARC:#define __SIZEOF_INT__ 4
// SPARC:#define __SIZEOF_LONG_DOUBLE__ 8
// SPARC:#define __SIZEOF_LONG_LONG__ 8
// SPARC:#define __SIZEOF_LONG__ 4
// SPARC:#define __SIZEOF_POINTER__ 4
// SPARC:#define __SIZEOF_PTRDIFF_T__ 4
// SPARC:#define __SIZEOF_SHORT__ 2
// SPARC:#define __SIZEOF_SIZE_T__ 4
// SPARC:#define __SIZEOF_WCHAR_T__ 4
// SPARC:#define __SIZEOF_WINT_T__ 4
// SPARC:#define __SIZE_TYPE__ long unsigned int
// SPARC:#define __SIZE_WIDTH__ 32
// SPARC:#define __UINTMAX_TYPE__ long long unsigned int
// SPARC:#define __USER_LABEL_PREFIX__ _
// SPARC:#define __VERSION__ "4.2.1 Compatible
// SPARC:#define __WCHAR_MAX__ 2147483647
// SPARC:#define __WCHAR_TYPE__ int
// SPARC:#define __WCHAR_WIDTH__ 32
// SPARC:#define __WINT_TYPE__ int
// SPARC:#define __WINT_WIDTH__ 32
// SPARC:#define __sparc 1
// SPARC:#define __sparc__ 1
// SPARC:#define __sparcv8 1
// SPARC:#define sparc 1
// 
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=tce-none-none < /dev/null | FileCheck -check-prefix TCE %s
//
// TCE:#define __CHAR16_TYPE__ unsigned short
// TCE:#define __CHAR32_TYPE__ unsigned int
// TCE:#define __CHAR_BIT__ 8
// TCE:#define __DBL_DENORM_MIN__ 1.40129846e-45F
// TCE:#define __DBL_DIG__ 6
// TCE:#define __DBL_EPSILON__ 1.19209290e-7F
// TCE:#define __DBL_HAS_DENORM__ 1
// TCE:#define __DBL_HAS_INFINITY__ 1
// TCE:#define __DBL_HAS_QUIET_NAN__ 1
// TCE:#define __DBL_MANT_DIG__ 24
// TCE:#define __DBL_MAX_10_EXP__ 38
// TCE:#define __DBL_MAX_EXP__ 128
// TCE:#define __DBL_MAX__ 3.40282347e+38F
// TCE:#define __DBL_MIN_10_EXP__ (-37)
// TCE:#define __DBL_MIN_EXP__ (-125)
// TCE:#define __DBL_MIN__ 1.17549435e-38F
// TCE:#define __DECIMAL_DIG__ -1
// TCE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// TCE:#define __FLT_DIG__ 6
// TCE:#define __FLT_EPSILON__ 1.19209290e-7F
// TCE:#define __FLT_EVAL_METHOD__ 0
// TCE:#define __FLT_HAS_DENORM__ 1
// TCE:#define __FLT_HAS_INFINITY__ 1
// TCE:#define __FLT_HAS_QUIET_NAN__ 1
// TCE:#define __FLT_MANT_DIG__ 24
// TCE:#define __FLT_MAX_10_EXP__ 38
// TCE:#define __FLT_MAX_EXP__ 128
// TCE:#define __FLT_MAX__ 3.40282347e+38F
// TCE:#define __FLT_MIN_10_EXP__ (-37)
// TCE:#define __FLT_MIN_EXP__ (-125)
// TCE:#define __FLT_MIN__ 1.17549435e-38F
// TCE:#define __FLT_RADIX__ 2
// TCE:#define __INT16_TYPE__ short
// TCE:#define __INT32_TYPE__ int
// TCE:#define __INT8_TYPE__ char
// TCE:#define __INTMAX_MAX__ 2147483647L
// TCE:#define __INTMAX_TYPE__ long int
// TCE:#define __INTMAX_WIDTH__ 32
// TCE:#define __INTPTR_TYPE__ int
// TCE:#define __INTPTR_WIDTH__ 32
// TCE:#define __INT_MAX__ 2147483647
// TCE:#define __LDBL_DENORM_MIN__ 1.40129846e-45F
// TCE:#define __LDBL_DIG__ 6
// TCE:#define __LDBL_EPSILON__ 1.19209290e-7F
// TCE:#define __LDBL_HAS_DENORM__ 1
// TCE:#define __LDBL_HAS_INFINITY__ 1
// TCE:#define __LDBL_HAS_QUIET_NAN__ 1
// TCE:#define __LDBL_MANT_DIG__ 24
// TCE:#define __LDBL_MAX_10_EXP__ 38
// TCE:#define __LDBL_MAX_EXP__ 128
// TCE:#define __LDBL_MAX__ 3.40282347e+38F
// TCE:#define __LDBL_MIN_10_EXP__ (-37)
// TCE:#define __LDBL_MIN_EXP__ (-125)
// TCE:#define __LDBL_MIN__ 1.17549435e-38F
// TCE:#define __LONG_LONG_MAX__ 2147483647LL
// TCE:#define __LONG_MAX__ 2147483647L
// TCE:#define __NO_INLINE__ 1
// TCE:#define __POINTER_WIDTH__ 32
// TCE:#define __PTRDIFF_TYPE__ int
// TCE:#define __PTRDIFF_WIDTH__ 32
// TCE:#define __SCHAR_MAX__ 127
// TCE:#define __SHRT_MAX__ 32767
// TCE:#define __SIG_ATOMIC_WIDTH__ 32
// TCE:#define __SIZEOF_DOUBLE__ 4
// TCE:#define __SIZEOF_FLOAT__ 4
// TCE:#define __SIZEOF_INT__ 4
// TCE:#define __SIZEOF_LONG_DOUBLE__ 4
// TCE:#define __SIZEOF_LONG_LONG__ 4
// TCE:#define __SIZEOF_LONG__ 4
// TCE:#define __SIZEOF_POINTER__ 4
// TCE:#define __SIZEOF_PTRDIFF_T__ 4
// TCE:#define __SIZEOF_SHORT__ 2
// TCE:#define __SIZEOF_SIZE_T__ 4
// TCE:#define __SIZEOF_WCHAR_T__ 4
// TCE:#define __SIZEOF_WINT_T__ 4
// TCE:#define __SIZE_TYPE__ unsigned int
// TCE:#define __SIZE_WIDTH__ 32
// TCE:#define __TCE_V1__ 1
// TCE:#define __TCE__ 1
// TCE:#define __UINTMAX_TYPE__ long unsigned int
// TCE:#define __USER_LABEL_PREFIX__ _
// TCE:#define __WCHAR_MAX__ 2147483647
// TCE:#define __WCHAR_TYPE__ int
// TCE:#define __WCHAR_WIDTH__ 32
// TCE:#define __WINT_TYPE__ int
// TCE:#define __WINT_WIDTH__ 32
// TCE:#define __tce 1
// TCE:#define __tce__ 1
// TCE:#define tce 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-none-none < /dev/null | FileCheck -check-prefix X86_64 %s
//
// X86_64:#define _LP64 1
// X86_64:#define __CHAR16_TYPE__ unsigned short
// X86_64:#define __CHAR32_TYPE__ unsigned int
// X86_64:#define __CHAR_BIT__ 8
// X86_64:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X86_64:#define __DBL_DIG__ 15
// X86_64:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X86_64:#define __DBL_HAS_DENORM__ 1
// X86_64:#define __DBL_HAS_INFINITY__ 1
// X86_64:#define __DBL_HAS_QUIET_NAN__ 1
// X86_64:#define __DBL_MANT_DIG__ 53
// X86_64:#define __DBL_MAX_10_EXP__ 308
// X86_64:#define __DBL_MAX_EXP__ 1024
// X86_64:#define __DBL_MAX__ 1.7976931348623157e+308
// X86_64:#define __DBL_MIN_10_EXP__ (-307)
// X86_64:#define __DBL_MIN_EXP__ (-1021)
// X86_64:#define __DBL_MIN__ 2.2250738585072014e-308
// X86_64:#define __DECIMAL_DIG__ 21
// X86_64:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X86_64:#define __FLT_DIG__ 6
// X86_64:#define __FLT_EPSILON__ 1.19209290e-7F
// X86_64:#define __FLT_EVAL_METHOD__ 0
// X86_64:#define __FLT_HAS_DENORM__ 1
// X86_64:#define __FLT_HAS_INFINITY__ 1
// X86_64:#define __FLT_HAS_QUIET_NAN__ 1
// X86_64:#define __FLT_MANT_DIG__ 24
// X86_64:#define __FLT_MAX_10_EXP__ 38
// X86_64:#define __FLT_MAX_EXP__ 128
// X86_64:#define __FLT_MAX__ 3.40282347e+38F
// X86_64:#define __FLT_MIN_10_EXP__ (-37)
// X86_64:#define __FLT_MIN_EXP__ (-125)
// X86_64:#define __FLT_MIN__ 1.17549435e-38F
// X86_64:#define __FLT_RADIX__ 2
// X86_64:#define __INT16_TYPE__ short
// X86_64:#define __INT32_TYPE__ int
// X86_64:#define __INT64_C_SUFFIX__ L
// X86_64:#define __INT64_TYPE__ long int
// X86_64:#define __INT8_TYPE__ char
// X86_64:#define __INTMAX_MAX__ 9223372036854775807L
// X86_64:#define __INTMAX_TYPE__ long int
// X86_64:#define __INTMAX_WIDTH__ 64
// X86_64:#define __INTPTR_TYPE__ long int
// X86_64:#define __INTPTR_WIDTH__ 64
// X86_64:#define __INT_MAX__ 2147483647
// X86_64:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X86_64:#define __LDBL_DIG__ 18
// X86_64:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X86_64:#define __LDBL_HAS_DENORM__ 1
// X86_64:#define __LDBL_HAS_INFINITY__ 1
// X86_64:#define __LDBL_HAS_QUIET_NAN__ 1
// X86_64:#define __LDBL_MANT_DIG__ 64
// X86_64:#define __LDBL_MAX_10_EXP__ 4932
// X86_64:#define __LDBL_MAX_EXP__ 16384
// X86_64:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X86_64:#define __LDBL_MIN_10_EXP__ (-4931)
// X86_64:#define __LDBL_MIN_EXP__ (-16381)
// X86_64:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X86_64:#define __LITTLE_ENDIAN__ 1
// X86_64:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X86_64:#define __LONG_MAX__ 9223372036854775807L
// X86_64:#define __LP64__ 1
// X86_64:#define __MMX__ 1
// X86_64:#define __NO_INLINE__ 1
// X86_64:#define __NO_MATH_INLINES 1
// X86_64:#define __POINTER_WIDTH__ 64
// X86_64:#define __PTRDIFF_TYPE__ long int
// X86_64:#define __PTRDIFF_WIDTH__ 64
// X86_64:#define __REGISTER_PREFIX__ 
// X86_64:#define __SCHAR_MAX__ 127
// X86_64:#define __SHRT_MAX__ 32767
// X86_64:#define __SIG_ATOMIC_WIDTH__ 32
// X86_64:#define __SIZEOF_DOUBLE__ 8
// X86_64:#define __SIZEOF_FLOAT__ 4
// X86_64:#define __SIZEOF_INT__ 4
// X86_64:#define __SIZEOF_LONG_DOUBLE__ 16
// X86_64:#define __SIZEOF_LONG_LONG__ 8
// X86_64:#define __SIZEOF_LONG__ 8
// X86_64:#define __SIZEOF_POINTER__ 8
// X86_64:#define __SIZEOF_PTRDIFF_T__ 8
// X86_64:#define __SIZEOF_SHORT__ 2
// X86_64:#define __SIZEOF_SIZE_T__ 8
// X86_64:#define __SIZEOF_WCHAR_T__ 4
// X86_64:#define __SIZEOF_WINT_T__ 4
// X86_64:#define __SIZE_TYPE__ long unsigned int
// X86_64:#define __SIZE_WIDTH__ 64
// X86_64:#define __SSE2_MATH__ 1
// X86_64:#define __SSE2__ 1
// X86_64:#define __SSE_MATH__ 1
// X86_64:#define __SSE__ 1
// X86_64:#define __UINTMAX_TYPE__ long unsigned int
// X86_64:#define __USER_LABEL_PREFIX__ _
// X86_64:#define __WCHAR_MAX__ 2147483647
// X86_64:#define __WCHAR_TYPE__ int
// X86_64:#define __WCHAR_WIDTH__ 32
// X86_64:#define __WINT_TYPE__ int
// X86_64:#define __WINT_WIDTH__ 32
// X86_64:#define __amd64 1
// X86_64:#define __amd64__ 1
// X86_64:#define __nocona 1
// X86_64:#define __nocona__ 1
// X86_64:#define __tune_nocona__ 1
// X86_64:#define __x86_64 1
// X86_64:#define __x86_64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-pc-linux-gnu < /dev/null | FileCheck -check-prefix X86_64-LINUX %s
//
// X86_64-LINUX:#define _LP64 1
// X86_64-LINUX:#define __CHAR16_TYPE__ unsigned short
// X86_64-LINUX:#define __CHAR32_TYPE__ unsigned int
// X86_64-LINUX:#define __CHAR_BIT__ 8
// X86_64-LINUX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X86_64-LINUX:#define __DBL_DIG__ 15
// X86_64-LINUX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X86_64-LINUX:#define __DBL_HAS_DENORM__ 1
// X86_64-LINUX:#define __DBL_HAS_INFINITY__ 1
// X86_64-LINUX:#define __DBL_HAS_QUIET_NAN__ 1
// X86_64-LINUX:#define __DBL_MANT_DIG__ 53
// X86_64-LINUX:#define __DBL_MAX_10_EXP__ 308
// X86_64-LINUX:#define __DBL_MAX_EXP__ 1024
// X86_64-LINUX:#define __DBL_MAX__ 1.7976931348623157e+308
// X86_64-LINUX:#define __DBL_MIN_10_EXP__ (-307)
// X86_64-LINUX:#define __DBL_MIN_EXP__ (-1021)
// X86_64-LINUX:#define __DBL_MIN__ 2.2250738585072014e-308
// X86_64-LINUX:#define __DECIMAL_DIG__ 21
// X86_64-LINUX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X86_64-LINUX:#define __FLT_DIG__ 6
// X86_64-LINUX:#define __FLT_EPSILON__ 1.19209290e-7F
// X86_64-LINUX:#define __FLT_EVAL_METHOD__ 0
// X86_64-LINUX:#define __FLT_HAS_DENORM__ 1
// X86_64-LINUX:#define __FLT_HAS_INFINITY__ 1
// X86_64-LINUX:#define __FLT_HAS_QUIET_NAN__ 1
// X86_64-LINUX:#define __FLT_MANT_DIG__ 24
// X86_64-LINUX:#define __FLT_MAX_10_EXP__ 38
// X86_64-LINUX:#define __FLT_MAX_EXP__ 128
// X86_64-LINUX:#define __FLT_MAX__ 3.40282347e+38F
// X86_64-LINUX:#define __FLT_MIN_10_EXP__ (-37)
// X86_64-LINUX:#define __FLT_MIN_EXP__ (-125)
// X86_64-LINUX:#define __FLT_MIN__ 1.17549435e-38F
// X86_64-LINUX:#define __FLT_RADIX__ 2
// X86_64-LINUX:#define __INT16_TYPE__ short
// X86_64-LINUX:#define __INT32_TYPE__ int
// X86_64-LINUX:#define __INT64_C_SUFFIX__ L
// X86_64-LINUX:#define __INT64_TYPE__ long int
// X86_64-LINUX:#define __INT8_TYPE__ char
// X86_64-LINUX:#define __INTMAX_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __INTMAX_TYPE__ long int
// X86_64-LINUX:#define __INTMAX_WIDTH__ 64
// X86_64-LINUX:#define __INTPTR_TYPE__ long int
// X86_64-LINUX:#define __INTPTR_WIDTH__ 64
// X86_64-LINUX:#define __INT_MAX__ 2147483647
// X86_64-LINUX:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X86_64-LINUX:#define __LDBL_DIG__ 18
// X86_64-LINUX:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X86_64-LINUX:#define __LDBL_HAS_DENORM__ 1
// X86_64-LINUX:#define __LDBL_HAS_INFINITY__ 1
// X86_64-LINUX:#define __LDBL_HAS_QUIET_NAN__ 1
// X86_64-LINUX:#define __LDBL_MANT_DIG__ 64
// X86_64-LINUX:#define __LDBL_MAX_10_EXP__ 4932
// X86_64-LINUX:#define __LDBL_MAX_EXP__ 16384
// X86_64-LINUX:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X86_64-LINUX:#define __LDBL_MIN_10_EXP__ (-4931)
// X86_64-LINUX:#define __LDBL_MIN_EXP__ (-16381)
// X86_64-LINUX:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X86_64-LINUX:#define __LITTLE_ENDIAN__ 1
// X86_64-LINUX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X86_64-LINUX:#define __LONG_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __LP64__ 1
// X86_64-LINUX:#define __MMX__ 1
// X86_64-LINUX:#define __NO_INLINE__ 1
// X86_64-LINUX:#define __NO_MATH_INLINES 1
// X86_64-LINUX:#define __POINTER_WIDTH__ 64
// X86_64-LINUX:#define __PTRDIFF_TYPE__ long int
// X86_64-LINUX:#define __PTRDIFF_WIDTH__ 64
// X86_64-LINUX:#define __REGISTER_PREFIX__ 
// X86_64-LINUX:#define __SCHAR_MAX__ 127
// X86_64-LINUX:#define __SHRT_MAX__ 32767
// X86_64-LINUX:#define __SIG_ATOMIC_WIDTH__ 32
// X86_64-LINUX:#define __SIZEOF_DOUBLE__ 8
// X86_64-LINUX:#define __SIZEOF_FLOAT__ 4
// X86_64-LINUX:#define __SIZEOF_INT__ 4
// X86_64-LINUX:#define __SIZEOF_LONG_DOUBLE__ 16
// X86_64-LINUX:#define __SIZEOF_LONG_LONG__ 8
// X86_64-LINUX:#define __SIZEOF_LONG__ 8
// X86_64-LINUX:#define __SIZEOF_POINTER__ 8
// X86_64-LINUX:#define __SIZEOF_PTRDIFF_T__ 8
// X86_64-LINUX:#define __SIZEOF_SHORT__ 2
// X86_64-LINUX:#define __SIZEOF_SIZE_T__ 8
// X86_64-LINUX:#define __SIZEOF_WCHAR_T__ 4
// X86_64-LINUX:#define __SIZEOF_WINT_T__ 4
// X86_64-LINUX:#define __SIZE_TYPE__ long unsigned int
// X86_64-LINUX:#define __SIZE_WIDTH__ 64
// X86_64-LINUX:#define __SSE2_MATH__ 1
// X86_64-LINUX:#define __SSE2__ 1
// X86_64-LINUX:#define __SSE_MATH__ 1
// X86_64-LINUX:#define __SSE__ 1
// X86_64-LINUX:#define __UINTMAX_TYPE__ long unsigned int
// X86_64-LINUX:#define __USER_LABEL_PREFIX__
// X86_64-LINUX:#define __WCHAR_MAX__ 2147483647
// X86_64-LINUX:#define __WCHAR_TYPE__ int
// X86_64-LINUX:#define __WCHAR_WIDTH__ 32
// X86_64-LINUX:#define __WINT_TYPE__ unsigned int
// X86_64-LINUX:#define __WINT_WIDTH__ 32
// X86_64-LINUX:#define __amd64 1
// X86_64-LINUX:#define __amd64__ 1
// X86_64-LINUX:#define __nocona 1
// X86_64-LINUX:#define __nocona__ 1
// X86_64-LINUX:#define __tune_nocona__ 1
// X86_64-LINUX:#define __x86_64 1
// X86_64-LINUX:#define __x86_64__ 1
//
// RUN: %clang_cc1 -x c++ -triple i686-pc-linux-gnu -E -dM < /dev/null | FileCheck -check-prefix GNUSOURCE %s
// GNUSOURCE:#define _GNU_SOURCE 1
// 
// RUN: %clang_cc1 -x c++ -std=c++98 -fno-rtti -E -dM < /dev/null | FileCheck -check-prefix NORTTI %s
// NORTTI: __GXX_ABI_VERSION
// NORTTI-NOT:#define __GXX_RTTI
// NORTTI: __STDC__
