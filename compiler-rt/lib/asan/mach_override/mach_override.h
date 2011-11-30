/*******************************************************************************
	mach_override.h
		Copyright (c) 2003-2009 Jonathan 'Wolf' Rentzsch: <http://rentzsch.com>
		Some rights reserved: <http://opensource.org/licenses/mit-license.php>

	***************************************************************************/

/***************************************************************************//**
	@mainpage	mach_override
	@author		Jonathan 'Wolf' Rentzsch: <http://rentzsch.com>
	
	This package, coded in C to the Mach API, allows you to override ("patch")
	program- and system-supplied functions at runtime. You can fully replace
	functions with your implementations, or merely head- or tail-patch the
	original implementations.
	
	Use it by #include'ing mach_override.h from your .c, .m or .mm file(s).
	
	@todo	Discontinue use of Carbon's MakeDataExecutable() and
			CompareAndSwap() calls and start using the Mach equivalents, if they
			exist. If they don't, write them and roll them in. That way, this
			code will be pure Mach, which will make it easier to use everywhere.
			Update: MakeDataExecutable() has been replaced by
			msync(MS_INVALIDATE). There is an OSCompareAndSwap in libkern, but
			I'm currently unsure if I can link against it. May have to roll in
			my own version...
	@todo	Stop using an entire 4K high-allocated VM page per 28-byte escape
			branch island. Done right, this will dramatically speed up escape
			island allocations when they number over 250. Then again, if you're
			overriding more than 250 functions, maybe speed isn't your main
			concern...
	@todo	Add detection of: b, bl, bla, bc, bcl, bcla, bcctrl, bclrl
			first-instructions. Initially, we should refuse to override
			functions beginning with these instructions. Eventually, we should
			dynamically rewrite them to make them position-independent.
	@todo	Write mach_unoverride(), which would remove an override placed on a
			function. Must be multiple-override aware, which means an almost
			complete rewrite under the covers, because the target address can't
			be spread across two load instructions like it is now since it will
			need to be atomically updatable.
	@todo	Add non-rentry variants of overrides to test_mach_override.

	***************************************************************************/

#ifndef		_mach_override_
#define		_mach_override_

#include <sys/types.h>
#include <mach/error.h>

#ifdef	__cplusplus
	extern	"C"	{
#endif

/**
	Returned if the function to be overrided begins with a 'mfctr' instruction.
*/
#define	err_cannot_override	(err_local|1)

/************************************************************************************//**
	Dynamically overrides the function implementation referenced by
	originalFunctionAddress with the implentation pointed to by overrideFunctionAddress.
	Optionally returns a pointer to a "reentry island" which, if jumped to, will resume
	the original implementation.
	
	@param	originalFunctionAddress			->	Required address of the function to
												override (with overrideFunctionAddress).
	@param	overrideFunctionAddress			->	Required address to the overriding
												function.
	@param	originalFunctionReentryIsland	<-	Optional pointer to pointer to the
												reentry island. Can be NULL.
	@result									<-	err_cannot_override if the original
												function's implementation begins with
												the 'mfctr' instruction.

	************************************************************************************/

    mach_error_t
mach_override_ptr(
	void *originalFunctionAddress,
    const void *overrideFunctionAddress,
    void **originalFunctionReentryIsland );

/************************************************************************************//**
	

	************************************************************************************/
 
#ifdef	__cplusplus

#define MACH_OVERRIDE( ORIGINAL_FUNCTION_RETURN_TYPE, ORIGINAL_FUNCTION_NAME, ORIGINAL_FUNCTION_ARGS, ERR )			\
	{																												\
		static ORIGINAL_FUNCTION_RETURN_TYPE (*ORIGINAL_FUNCTION_NAME##_reenter)ORIGINAL_FUNCTION_ARGS;				\
		static bool ORIGINAL_FUNCTION_NAME##_overriden = false;														\
		class mach_override_class__##ORIGINAL_FUNCTION_NAME {														\
		public:																										\
			static kern_return_t override(void *originalFunctionPtr) {												\
				kern_return_t result = err_none;																	\
				if (!ORIGINAL_FUNCTION_NAME##_overriden) {															\
					ORIGINAL_FUNCTION_NAME##_overriden = true;														\
					result = mach_override_ptr( (void*)originalFunctionPtr,											\
												(void*)mach_override_class__##ORIGINAL_FUNCTION_NAME::replacement,	\
												(void**)&ORIGINAL_FUNCTION_NAME##_reenter );						\
				}																									\
				return result;																						\
			}																										\
			static ORIGINAL_FUNCTION_RETURN_TYPE replacement ORIGINAL_FUNCTION_ARGS {

#define END_MACH_OVERRIDE( ORIGINAL_FUNCTION_NAME )																	\
			}																										\
		};																											\
																													\
		err = mach_override_class__##ORIGINAL_FUNCTION_NAME::override((void*)ORIGINAL_FUNCTION_NAME);				\
	}
 
#endif

#ifdef	__cplusplus
	}
#endif
#endif	//	_mach_override_
