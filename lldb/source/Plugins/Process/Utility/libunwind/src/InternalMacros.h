/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- InternalMacros.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef INTERNAL_MACROS_H
#define INTERNAL_MACROS_H

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif
	extern void __assert_rtn(const char *, const char *, int, const char *) __attribute__((noreturn));
#ifdef __cplusplus
}
#endif

#define UNW_STEP_SUCCESS 1
#define UNW_STEP_END     0


struct v128 { unsigned int vec[4]; };


#define EXPORT __attribute__((visibility("default"))) 

#define COMPILE_TIME_ASSERT( expr )    \
		extern int compile_time_assert_failed[ ( expr ) ? 1 : -1 ] __attribute__( ( unused ) );

#define ABORT(msg) __assert_rtn(__func__, __FILE__, __LINE__, msg) 

#if NDEBUG
	#define DEBUG_MESSAGE(msg, ...)  
	#define DEBUG_PRINT_API(msg, ...)
	#define DEBUG_PRINT_UNWINDING_TEST 0
	#define DEBUG_PRINT_UNWINDING(msg, ...)
	#define DEBUG_LOG_NON_ZERO(x) x;
	#define INITIALIZE_DEBUG_PRINT_API
	#define INITIALIZE_DEBUG_PRINT_UNWINDING
#else
	#define DEBUG_MESSAGE(msg, ...)  fprintf(stderr, "libuwind: " msg, __VA_ARGS__)
	#ifdef __cplusplus
		extern "C" {
	#endif
		extern  bool logAPIs();
		extern  bool logUnwinding();
	#ifdef __cplusplus
		}
	#endif
	#define DEBUG_LOG_NON_ZERO(x) { int _err = x; if ( _err != 0 ) fprintf(stderr, "libuwind: " #x "=%d in %s", _err, __FUNCTION__); }
	#define DEBUG_PRINT_API(msg, ...) do { if ( logAPIs() ) fprintf(stderr,  msg, __VA_ARGS__); } while(0)
	#define DEBUG_PRINT_UNWINDING(msg, ...) do { if ( logUnwinding() ) fprintf(stderr,  msg, __VA_ARGS__); } while(0)
	#define DEBUG_PRINT_UNWINDING_TEST logUnwinding()
	#define INITIALIZE_DEBUG_PRINT_API bool logAPIs() { static bool log = (getenv("LIBUNWIND_PRINT_APIS") != NULL); return log; }
	#define INITIALIZE_DEBUG_PRINT_UNWINDING bool logUnwinding() { static bool log = (getenv("LIBUNWIND_PRINT_UNWINDING") != NULL); return log; }
#endif


// note hack for <rdar://problem/6175741>
// Once libgcc_s.dylib vectors to libSystem, then we can remove the $ld$hide$os10.6$ lines
#if __ppc__
	#define NOT_HERE_BEFORE_10_6(sym) \
		extern const char sym##_tmp3 __asm("$ld$hide$os10.3$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp3 = 0; \
 		extern const char sym##_tmp4 __asm("$ld$hide$os10.4$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp4 = 0; \
		extern const char sym##_tmp5 __asm("$ld$hide$os10.5$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp5 = 0; 
	#define NEVER_HERE(sym) \
		extern const char sym##_tmp3 __asm("$ld$hide$os10.3$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp3 = 0; \
 		extern const char sym##_tmp4 __asm("$ld$hide$os10.4$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp4 = 0; \
		extern const char sym##_tmp5 __asm("$ld$hide$os10.5$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp5 = 0; \
		extern const char sym##_tmp6 __asm("$ld$hide$os10.6$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp6 = 0;
#else
	#define NOT_HERE_BEFORE_10_6(sym) \
 		extern const char sym##_tmp4 __asm("$ld$hide$os10.4$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp4 = 0; \
		extern const char sym##_tmp5 __asm("$ld$hide$os10.5$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp5 = 0; 
	#define NEVER_HERE(sym) \
 		extern const char sym##_tmp4 __asm("$ld$hide$os10.4$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp4 = 0; \
		extern const char sym##_tmp5 __asm("$ld$hide$os10.5$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp5 = 0; \
		extern const char sym##_tmp6 __asm("$ld$hide$os10.6$_" #sym ); __attribute__((visibility("default"))) const char sym##_tmp6 = 0;
#endif



#endif // INTERNAL_MACROS_H
