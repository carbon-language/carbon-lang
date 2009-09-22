/* ===-- apple_versioning.c - Adds versioning symbols for ld ---------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */



#if __APPLE__
  #if __arm__
    #define NOT_HERE_BEFORE_10_6(sym) 
  #elif __ppc__
    #define NOT_HERE_BEFORE_10_6(sym) \
        extern const char sym##_tmp3 __asm("$ld$hide$os10.3$_" #sym ); \
            __attribute__((visibility("default"))) const char sym##_tmp3 = 0; \
         extern const char sym##_tmp4 __asm("$ld$hide$os10.4$_" #sym ); \
            __attribute__((visibility("default"))) const char sym##_tmp4 = 0; \
        extern const char sym##_tmp5 __asm("$ld$hide$os10.5$_" #sym ); \
            __attribute__((visibility("default"))) const char sym##_tmp5 = 0; 
  #else
    #define NOT_HERE_BEFORE_10_6(sym) \
         extern const char sym##_tmp4 __asm("$ld$hide$os10.4$_" #sym ); \
            __attribute__((visibility("default"))) const char sym##_tmp4 = 0; \
        extern const char sym##_tmp5 __asm("$ld$hide$os10.5$_" #sym ); \
            __attribute__((visibility("default"))) const char sym##_tmp5 = 0; 
  #endif /* __ppc__ */


/* Symbols in libSystem.dylib in 10.6 and later, 
 *  but are in libgcc_s.dylib in earlier versions
 */

NOT_HERE_BEFORE_10_6(__absvdi2)
NOT_HERE_BEFORE_10_6(__absvsi2)
NOT_HERE_BEFORE_10_6(__absvti2)
NOT_HERE_BEFORE_10_6(__addvdi3)
NOT_HERE_BEFORE_10_6(__addvsi3)
NOT_HERE_BEFORE_10_6(__addvti3)
NOT_HERE_BEFORE_10_6(__ashldi3)
NOT_HERE_BEFORE_10_6(__ashlti3)
NOT_HERE_BEFORE_10_6(__ashrdi3)
NOT_HERE_BEFORE_10_6(__ashrti3)
NOT_HERE_BEFORE_10_6(__clear_cache)
NOT_HERE_BEFORE_10_6(__clzdi2)
NOT_HERE_BEFORE_10_6(__clzsi2)
NOT_HERE_BEFORE_10_6(__clzti2)
NOT_HERE_BEFORE_10_6(__cmpdi2)
NOT_HERE_BEFORE_10_6(__cmpti2)
NOT_HERE_BEFORE_10_6(__ctzdi2)
NOT_HERE_BEFORE_10_6(__ctzsi2)
NOT_HERE_BEFORE_10_6(__ctzti2)
NOT_HERE_BEFORE_10_6(__divdc3)
NOT_HERE_BEFORE_10_6(__divdi3)
NOT_HERE_BEFORE_10_6(__divsc3)
NOT_HERE_BEFORE_10_6(__divtc3)
NOT_HERE_BEFORE_10_6(__divti3)
NOT_HERE_BEFORE_10_6(__divxc3)
NOT_HERE_BEFORE_10_6(__enable_execute_stack)
NOT_HERE_BEFORE_10_6(__ffsdi2)
NOT_HERE_BEFORE_10_6(__ffsti2)
NOT_HERE_BEFORE_10_6(__fixdfdi)
NOT_HERE_BEFORE_10_6(__fixdfti)
NOT_HERE_BEFORE_10_6(__fixsfdi)
NOT_HERE_BEFORE_10_6(__fixsfti)
NOT_HERE_BEFORE_10_6(__fixtfdi)
NOT_HERE_BEFORE_10_6(__fixunsdfdi)
NOT_HERE_BEFORE_10_6(__fixunsdfsi)
NOT_HERE_BEFORE_10_6(__fixunsdfti)
NOT_HERE_BEFORE_10_6(__fixunssfdi)
NOT_HERE_BEFORE_10_6(__fixunssfsi)
NOT_HERE_BEFORE_10_6(__fixunssfti)
NOT_HERE_BEFORE_10_6(__fixunstfdi)
NOT_HERE_BEFORE_10_6(__fixunsxfdi)
NOT_HERE_BEFORE_10_6(__fixunsxfsi)
NOT_HERE_BEFORE_10_6(__fixunsxfti)
NOT_HERE_BEFORE_10_6(__fixxfdi)
NOT_HERE_BEFORE_10_6(__fixxfti)
NOT_HERE_BEFORE_10_6(__floatdidf)
NOT_HERE_BEFORE_10_6(__floatdisf)
NOT_HERE_BEFORE_10_6(__floatditf)
NOT_HERE_BEFORE_10_6(__floatdixf)
NOT_HERE_BEFORE_10_6(__floattidf)
NOT_HERE_BEFORE_10_6(__floattisf)
NOT_HERE_BEFORE_10_6(__floattixf)
NOT_HERE_BEFORE_10_6(__floatundidf)
NOT_HERE_BEFORE_10_6(__floatundisf)
NOT_HERE_BEFORE_10_6(__floatunditf)
NOT_HERE_BEFORE_10_6(__floatundixf)
NOT_HERE_BEFORE_10_6(__floatuntidf)
NOT_HERE_BEFORE_10_6(__floatuntisf)
NOT_HERE_BEFORE_10_6(__floatuntixf)
NOT_HERE_BEFORE_10_6(__gcc_personality_v0)
NOT_HERE_BEFORE_10_6(__lshrdi3)
NOT_HERE_BEFORE_10_6(__lshrti3)
NOT_HERE_BEFORE_10_6(__moddi3)
NOT_HERE_BEFORE_10_6(__modti3)
NOT_HERE_BEFORE_10_6(__muldc3)
NOT_HERE_BEFORE_10_6(__muldi3)
NOT_HERE_BEFORE_10_6(__mulsc3)
NOT_HERE_BEFORE_10_6(__multc3)
NOT_HERE_BEFORE_10_6(__multi3)
NOT_HERE_BEFORE_10_6(__mulvdi3)
NOT_HERE_BEFORE_10_6(__mulvsi3)
NOT_HERE_BEFORE_10_6(__mulvti3)
NOT_HERE_BEFORE_10_6(__mulxc3)
NOT_HERE_BEFORE_10_6(__negdi2)
NOT_HERE_BEFORE_10_6(__negti2)
NOT_HERE_BEFORE_10_6(__negvdi2)
NOT_HERE_BEFORE_10_6(__negvsi2)
NOT_HERE_BEFORE_10_6(__negvti2)
NOT_HERE_BEFORE_10_6(__paritydi2)
NOT_HERE_BEFORE_10_6(__paritysi2)
NOT_HERE_BEFORE_10_6(__parityti2)
NOT_HERE_BEFORE_10_6(__popcountdi2)
NOT_HERE_BEFORE_10_6(__popcountsi2)
NOT_HERE_BEFORE_10_6(__popcountti2)
NOT_HERE_BEFORE_10_6(__powidf2)
NOT_HERE_BEFORE_10_6(__powisf2)
NOT_HERE_BEFORE_10_6(__powitf2)
NOT_HERE_BEFORE_10_6(__powixf2)
NOT_HERE_BEFORE_10_6(__subvdi3)
NOT_HERE_BEFORE_10_6(__subvsi3)
NOT_HERE_BEFORE_10_6(__subvti3)
NOT_HERE_BEFORE_10_6(__ucmpdi2)
NOT_HERE_BEFORE_10_6(__ucmpti2)
NOT_HERE_BEFORE_10_6(__udivdi3)
NOT_HERE_BEFORE_10_6(__udivmoddi4)
NOT_HERE_BEFORE_10_6(__udivmodti4)
NOT_HERE_BEFORE_10_6(__udivti3)
NOT_HERE_BEFORE_10_6(__umoddi3)
NOT_HERE_BEFORE_10_6(__umodti3)


#if __ppc__
NOT_HERE_BEFORE_10_6(__gcc_qadd)
NOT_HERE_BEFORE_10_6(__gcc_qdiv)
NOT_HERE_BEFORE_10_6(__gcc_qmul)
NOT_HERE_BEFORE_10_6(__gcc_qsub)
NOT_HERE_BEFORE_10_6(__trampoline_setup)
#endif /* __ppc__ */

#else /* !__APPLE__ */

extern int avoid_empty_file;

#endif /* !__APPLE__*/
