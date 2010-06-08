/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- UnwindLevel1-gcc-ext.c ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/* 
 *  Implements gcc extensions to the C++ ABI Exception Handling Level 1 as documented at:
 *			<http://www.codesourcery.com/cxx-abi/abi-eh.html>
 *  using libunwind
 * 
 */
 
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "libunwind.h"
#include "unwind.h"
#include "libunwind_priv.h"
#include "InternalMacros.h"


#if __ppc__ || __i386__ ||  __x86_64__

//
//  Called by __cxa_rethrow()
//
EXPORT _Unwind_Reason_Code _Unwind_Resume_or_Rethrow(struct _Unwind_Exception* exception_object)
{
	DEBUG_PRINT_API("_Unwind_Resume_or_Rethrow(ex_obj=%p), private_1=%ld\n", exception_object, exception_object->private_1);
	// if this is non-forced and a stopping place was found, then this is a re-throw
	// call _Unwind_RaiseException() as if this was a new exception
	if ( exception_object->private_1 == 0 )
		_Unwind_RaiseException(exception_object); 
	
	// call through to _Unwind_Resume() which distiguishes between forced and regular exceptions
	_Unwind_Resume(exception_object); 
	ABORT("_Unwind_Resume_or_Rethrow() called _Unwind_RaiseException() which unexpectedly returned");
}



//
// Called by personality handler during phase 2 to get base address for data relative encodings
//
EXPORT uintptr_t _Unwind_GetDataRelBase(struct _Unwind_Context* context)
{
	DEBUG_PRINT_API("_Unwind_GetDataRelBase(context=%p)\n", context);
	ABORT("_Unwind_GetDataRelBase() not implemented");
}

//
// Called by personality handler during phase 2 to get base address for text relative encodings
//
EXPORT uintptr_t _Unwind_GetTextRelBase(struct _Unwind_Context* context)
{
	DEBUG_PRINT_API("_Unwind_GetTextRelBase(context=%p)\n", context);
	ABORT("_Unwind_GetTextRelBase() not implemented");
}



//
//  Scans unwind information to find the function that contains the
//  specified code address "pc".
//
EXPORT void*  _Unwind_FindEnclosingFunction(void* pc)
{
	DEBUG_PRINT_API("_Unwind_FindEnclosingFunction(pc=%p)\n", pc);
	ABORT("_Unwind_FindEnclosingFunction() not implemented");
}


//
// Walk every frame and call trace function at each one.  If trace function
// returns anything other than _URC_NO_REASON, then walk is terminated.
//
EXPORT _Unwind_Reason_Code _Unwind_Backtrace(_Unwind_Trace_Fn callback, void* ref)
{
	unw_cursor_t cursor; 
	unw_context_t uc;
	unw_getcontext(&uc);
	unw_init_local(&cursor, &uc);
	
	DEBUG_PRINT_API("_Unwind_Backtrace(callback=%p)\n", callback);

	// walk each frame 
	while ( true ) {

		// ask libuwind to get next frame (skip over first frame which is _Unwind_Backtrace())
		if ( unw_step(&cursor) <= 0 ) {
			DEBUG_PRINT_UNWINDING(" _backtrace: ended because cursor reached bottom of stack, returning %d\n", _URC_END_OF_STACK);
			return _URC_END_OF_STACK;
		}
		
		// debugging
		if ( DEBUG_PRINT_UNWINDING_TEST ) {
			char				functionName[512];
			unw_proc_info_t		frameInfo;
			unw_word_t			offset;
			unw_get_proc_name(&cursor, functionName, 512, &offset);
			unw_get_proc_info(&cursor, &frameInfo);
			DEBUG_PRINT_UNWINDING(" _backtrace: start_ip=0x%llX, func=%s, lsda=0x%llX, context=%p\n", 
							 frameInfo.start_ip, functionName, frameInfo.lsda, &cursor);
		}
		
		// call trace function with this frame
		_Unwind_Reason_Code result = (*callback)((struct _Unwind_Context*)(&cursor), ref);
		if ( result != _URC_NO_REASON ) {
			DEBUG_PRINT_UNWINDING(" _backtrace: ended because callback returned %d\n",  result);
			return result;
		}
	}
}


//
// Find dwarf unwind info for an address 'pc' in some function.
//
EXPORT const void* _Unwind_Find_FDE(const void* pc, struct dwarf_eh_bases* bases)
{
	// This is slow, but works. 
	// We create an unwind cursor then alter the IP to be pc 
	unw_cursor_t	cursor; 
	unw_context_t	uc;
	unw_proc_info_t	info;
	unw_getcontext(&uc);
	unw_init_local(&cursor, &uc);
	unw_set_reg(&cursor, UNW_REG_IP, (unw_word_t)(long)pc);
	unw_get_proc_info(&cursor, &info);
	bases->tbase = info.extra;
	bases->dbase = 0;	// dbase not used on Mac OS X
	bases->func = info.start_ip;
	DEBUG_PRINT_API("_Unwind_Find_FDE(pc=%p) => %p\n", pc, (void*)(long)info.unwind_info);
	return (void*)(long)info.unwind_info;
}



EXPORT uintptr_t _Unwind_GetCFA(struct _Unwind_Context* context)
{
	unw_cursor_t* cursor = (unw_cursor_t*)context;
	unw_word_t result;
	unw_get_reg(cursor, UNW_REG_SP, &result);
	DEBUG_PRINT_API("_Unwind_GetCFA(context=%p) => 0x%llX\n", context, (uint64_t)result);
	return result;
}


//
// Called by personality handler during phase 2 to get instruction pointer.
// ipBefore is a boolean that says if IP is already adjusted to be the call
// site address.  Normally IP is the return address.
//
EXPORT uintptr_t _Unwind_GetIPInfo(struct _Unwind_Context* context, int* ipBefore)
{
	DEBUG_PRINT_API("_Unwind_GetIPInfo(context=%p)\n", context);
	*ipBefore = 0;
	return _Unwind_GetIP(context);
}


//
// Called by programs with dynamic code generators that want
// to register a dynamically generated FDE.
// This function has existed on Mac OS X since 10.4, but
// never worked before.  
//
EXPORT void	__register_frame(const void* fde)
{
	DEBUG_PRINT_API("__register_frame(%p)\n", fde);
	_unw_add_dynamic_fde((unw_word_t)(uintptr_t)fde);
}


//
// Called by programs with dynamic code generators that want
// to unregister a dynamically generated FDE.
// This function has existed on Mac OS X since 10.4, but
// never worked before.  
//
EXPORT void	__deregister_frame(const void* fde)
{
	DEBUG_PRINT_API("__deregister_frame(%p)\n", fde);
	_unw_remove_dynamic_fde((unw_word_t)(uintptr_t)fde);
}



//
// The following register/deregister functions are gcc extensions.  
// They have existed on Mac OS X, but have never worked because Mac OS X
// before 10.6 used keymgr to track known FDEs, but these functions
// never got updated to use keymgr.
// For now, we implement these as do-nothing functions to keep any existing
// applications working.  We also add the not in 10.6 symbol so that nwe
// application won't be able to use them.
//

EXPORT void	__register_frame_info_bases(const void* fde, void* ob, void* tb, void* db)
{
	DEBUG_PRINT_API("__register_frame_info_bases(%p,%p, %p, %p)\n", fde, ob, tb, db);
	// do nothing, this function never worked in Mac OS X
}

EXPORT void	__register_frame_info(const void* fde, void* ob)
{
	DEBUG_PRINT_API("__register_frame_info(%p, %p)\n", fde, ob);
	// do nothing, this function never worked in Mac OS X
}


EXPORT void	__register_frame_info_table_bases(const void* fde, void* ob, void* tb, void* db)
{
	DEBUG_PRINT_API("__register_frame_info_table_bases(%p,%p, %p, %p)\n", fde, ob, tb, db);
	// do nothing, this function never worked in Mac OS X
}

EXPORT void	__register_frame_info_table(const void* fde, void* ob)
{
	DEBUG_PRINT_API("__register_frame_info_table(%p, %p)\n", fde, ob);
	// do nothing, this function never worked in Mac OS X
}

EXPORT void	__register_frame_table(const void* fde)
{
	DEBUG_PRINT_API("__register_frame_table(%p)\n", fde);
	// do nothing, this function never worked in Mac OS X
}

EXPORT void* __deregister_frame_info(const void* fde)
{
	DEBUG_PRINT_API("__deregister_frame_info(%p)\n", fde);
	// do nothing, this function never worked in Mac OS X
	return NULL;
}

EXPORT void* __deregister_frame_info_bases(const void* fde)
{
	DEBUG_PRINT_API("__deregister_frame_info_bases(%p)\n", fde);
	// do nothing, this function never worked in Mac OS X
	return NULL;
}




//
// symbols in libSystem.dylib in 10.6 and later, but are in libgcc_s.dylib in earlier versions
//
NOT_HERE_BEFORE_10_6(_Unwind_Backtrace)
NOT_HERE_BEFORE_10_6(_Unwind_FindEnclosingFunction)
NOT_HERE_BEFORE_10_6(_Unwind_GetCFA)
NOT_HERE_BEFORE_10_6(_Unwind_GetDataRelBase)
NOT_HERE_BEFORE_10_6(_Unwind_GetTextRelBase)
NOT_HERE_BEFORE_10_6(_Unwind_Resume_or_Rethrow)
NOT_HERE_BEFORE_10_6(_Unwind_GetIPInfo)

NOT_HERE_BEFORE_10_6(__register_frame)
NOT_HERE_BEFORE_10_6(__deregister_frame)


//
// symbols in libSystem.dylib for compatibility, but we don't want any new code using them
//
NEVER_HERE(__register_frame_info_bases)
NEVER_HERE(__register_frame_info)
NEVER_HERE(__register_frame_info_table_bases)
NEVER_HERE(__register_frame_info_table)
NEVER_HERE(__register_frame_table)
NEVER_HERE(__deregister_frame_info)
NEVER_HERE(__deregister_frame_info_bases)


#endif // __ppc__ || __i386__ ||  __x86_64__

