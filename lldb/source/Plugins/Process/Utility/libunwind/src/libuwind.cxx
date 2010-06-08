/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- libuwind.cxx --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 
#if __ppc__ || __i386__ ||  __x86_64__

#include <mach/mach_types.h>
#include <mach/machine.h>
#include <new>

#include "libunwind.h"
#include "libunwind_priv.h"

#include "UnwindCursor.hpp"
#include "AddressSpace.hpp"

#include "RemoteProcInfo.hpp"

namespace lldb_private {

// setup debug logging hooks
INITIALIZE_DEBUG_PRINT_API
INITIALIZE_DEBUG_PRINT_UNWINDING

// internal object to represent this processes address space
static LocalAddressSpace sThisAddressSpace;

#pragma mark Local API

///
/// record the registers and stack position of the caller 
///
extern int unw_getcontext(unw_context_t*);
// note: unw_getcontext() implemented in assembly

///
/// create a cursor of a thread in this process given 'context' recorded by unw_getcontext()
///
EXPORT int unw_init_local(unw_cursor_t* cursor, unw_context_t* context)
{
	DEBUG_PRINT_API("unw_init_local(cursor=%p, context=%p)\n", cursor, context);
	// use "placement new" to allocate UnwindCursor in the cursor buffer
#if __i386__
	new ((void*)cursor) UnwindCursor<LocalAddressSpace,Registers_x86>(context, sThisAddressSpace);
#elif __x86_64__
	new ((void*)cursor) UnwindCursor<LocalAddressSpace,Registers_x86_64>(context, sThisAddressSpace);
#elif __ppc__
	new ((void*)cursor) UnwindCursor<LocalAddressSpace,Registers_ppc>(context, sThisAddressSpace);
#endif
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
	co->setInfoBasedOnIPRegister(NULL);

	return UNW_ESUCCESS;
}

///
/// move cursor to next frame 
///
EXPORT int unw_step(unw_cursor_t* cursor)
{
	DEBUG_PRINT_API("unw_step(cursor=%p)\n", cursor);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
    return co->step();
}

///
/// get value of specified register at cursor position in stack frame
///
EXPORT int unw_get_reg(unw_cursor_t* cursor, unw_regnum_t regNum, unw_word_t* value)
{
	DEBUG_PRINT_API("unw_get_reg(cursor=%p, regNum=%d, &value=%p)\n", cursor, regNum, value);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;

    if (co->validReg(regNum) == 0)
        return UNW_EBADREG;
	return co->getReg(regNum, value);
}

///
/// get value of specified float register at cursor position in stack frame
///
EXPORT int unw_get_fpreg(unw_cursor_t* cursor, unw_regnum_t regNum, unw_fpreg_t* value)
{
	DEBUG_PRINT_API("unw_get_fpreg(cursor=%p, regNum=%d, &value=%p)\n", cursor, regNum, value);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;

	if ( co->validFloatReg(regNum) ) {
		return co->getFloatReg(regNum, value);
	}
	return UNW_EBADREG;
}

///
/// set value of specified register at cursor position in stack frame
///
EXPORT int unw_set_reg(unw_cursor_t* cursor, unw_regnum_t regNum, unw_word_t value)
{
	DEBUG_PRINT_API("unw_set_reg(cursor=%p, regNum=%d, value=0x%llX)\n", cursor, regNum, value);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;

	if ( co->validReg(regNum) ) {
		co->setReg(regNum, value);
		// specical case altering IP to re-find info (being called by personality function)
		if ( regNum == UNW_REG_IP ) {
			unw_proc_info_t info;
			co->getInfo(&info);
			uint64_t orgArgSize = info.gp;
			uint64_t orgFuncStart = info.start_ip;
			co->setInfoBasedOnIPRegister(false);
			// and adjust REG_SP if there was a DW_CFA_GNU_args_size
			if ( (orgFuncStart == info.start_ip) && (orgArgSize != 0) )
				co->setReg(UNW_REG_SP, co->getReg(UNW_REG_SP) + orgArgSize);
		}
		return UNW_ESUCCESS;
	}
	return UNW_EBADREG;
}

///
/// set value of specified float register at cursor position in stack frame
///
EXPORT int unw_set_fpreg(unw_cursor_t* cursor, unw_regnum_t regNum, unw_fpreg_t value)
{
	DEBUG_PRINT_API("unw_set_fpreg(cursor=%p, regNum=%d, value=%g)\n", cursor, regNum, value);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;

	if ( co->validFloatReg(regNum) ) {
		return co->setFloatReg(regNum, value);
	}
	return UNW_EBADREG;
}

///
/// resume execution at cursor position (aka longjump) 
///
EXPORT int unw_resume(unw_cursor_t* cursor)
{
	DEBUG_PRINT_API("unw_resume(cursor=%p)\n", cursor);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;

	co->jumpto();
	return UNW_EUNSPEC;
}

///
/// returns the name of a register
///
EXPORT const char* unw_regname(unw_cursor_t* cursor, unw_regnum_t regNum)
{
	DEBUG_PRINT_API("unw_regname(cursor=%p, regNum=%d)\n", cursor, regNum);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
	return co->getRegisterName(regNum);
}

///
/// get unwind info at cursor position in stack frame 
///
EXPORT int unw_get_proc_info(unw_cursor_t* cursor, unw_proc_info_t* info)
{
	DEBUG_PRINT_API("unw_get_proc_info(cursor=%p, &info=%p)\n", cursor, info);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
	co->getInfo(info);
	if ( info->end_ip == 0 )
		return UNW_ENOINFO;
	else
		return UNW_ESUCCESS;
}

///
/// checks if a register is a floating-point register 
///
EXPORT int unw_is_fpreg(unw_cursor_t* cursor, unw_regnum_t regNum)
{
	DEBUG_PRINT_API("unw_is_fpreg(cursor=%p, regNum=%d)\n", cursor, regNum);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
	return co->validFloatReg(regNum);
}

///
/// checks if current frame is signal trampoline 
///
EXPORT int unw_is_signal_frame(unw_cursor_t* cursor)
{
	DEBUG_PRINT_API("unw_is_signal_frame(cursor=%p)\n", cursor);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
	return co->isSignalFrame();
}

///
/// get name of function at cursor position in stack frame 
///
EXPORT int unw_get_proc_name(unw_cursor_t* cursor, char* buf, size_t bufLen, unw_word_t* offset)
{
	DEBUG_PRINT_API("unw_get_proc_name(cursor=%p, &buf=%p, bufLen=%ld)\n", cursor, buf, bufLen);
	AbstractUnwindCursor* co = (AbstractUnwindCursor*)cursor;
	if ( co->getFunctionName(buf, bufLen, offset) )
		return UNW_ESUCCESS;
	else
		return UNW_EUNSPEC;
}

#pragma mark Remote API
 
#if defined (SUPPORT_REMOTE_UNWINDING)
EXPORT int unw_init_remote(unw_cursor_t *cursor, unw_addr_space_t as, void *arg)
{
    DEBUG_PRINT_API("init_remote(c=%p, as=%p, arg=%p)\n", cursor, as, arg);
	
    // API docs at http://www.nongnu.org/libunwind/docs.html say we should 
    // handle a local address space but we're not doing the "remote" unwinding
    // with local process accessors so punt on that.

    if(as->type != UNW_REMOTE)
    {
        ABORT("unw_init_remote was passed a non-remote address space");
        return UNW_EINVAL;
    }

	unw_accessors_t* acc = unw_get_accessors(as);
	if(!acc) {
		ABORT("unw_get_accessors returned NULL");
		return UNW_EINVAL;
	}
	
    unw_addr_space_remote* remote = (unw_addr_space_remote*)as;
	
    // use "placement new" to allocate UnwindCursor in the cursor buffer
    // It isn't really necessary to use placement new in the remote API but we'll stay consistent
    // with the rest of the code here.
    switch ( remote->ras->getTargetArch() ) {
        case UNW_TARGET_I386:
	{
                Registers_x86 *r = new Registers_x86;
                OtherAddressSpace<Pointer32<LittleEndian> > *addrSpace = new OtherAddressSpace<Pointer32<LittleEndian> >(as, arg);
                getRemoteContext (remote->ras, *r, arg);
                unw_context_t *context = (unw_context_t*) r;
                new ((void*)cursor) RemoteUnwindCursor<OtherAddressSpace<Pointer32<LittleEndian> >, Registers_x86>(*addrSpace, context, arg);
                break;
	}
            break;
        case UNW_TARGET_X86_64:
	{
                Registers_x86_64 *r = new Registers_x86_64;
                OtherAddressSpace<Pointer64<LittleEndian> > *addrSpace = new OtherAddressSpace<Pointer64<LittleEndian> >(as, arg);
                getRemoteContext (remote->ras, *r, arg);
                unw_context_t *context = (unw_context_t*) r;
                new ((void*)cursor) RemoteUnwindCursor<OtherAddressSpace<Pointer64<LittleEndian> >, Registers_x86_64>(*addrSpace, context, arg);
                break;
	}

        case UNW_TARGET_PPC:
              ABORT("ppc not supported for remote unwinds");
            break;

        case UNW_TARGET_ARM:
              ABORT("arm not supported for remote unwinds");
            break;

        default:
            return UNW_EUNSPEC;
    }
	
    AbstractRemoteUnwindCursor* co = (AbstractRemoteUnwindCursor*)cursor;
    co->setRemoteContext(arg);
	
	return UNW_ESUCCESS;
}

// The documentation disagrees about whether or not this returns a pointer.  Now it does.
EXPORT unw_accessors_t* unw_get_accessors(unw_addr_space_t as)
{
	if(as->type != UNW_REMOTE)
	{
		ABORT("unw_get_accessors was passed a non-remote address space");
		return NULL;
	}
	unw_addr_space_remote* remote = (unw_addr_space_remote*)as;
	
	if(remote->type != UNW_REMOTE)
		return NULL;
	
    return remote->ras->getAccessors();
}

EXPORT unw_addr_space_t unw_create_addr_space(unw_accessors_t *ap, unw_targettype_t targarch)
{
	unw_addr_space_remote* remote = (unw_addr_space_remote*)malloc(sizeof(unw_addr_space_remote));
	remote->type = UNW_REMOTE;
    remote->ras = new RemoteProcInfo(ap, targarch);
	return (unw_addr_space_t)remote;
}

EXPORT void unw_flush_caches(unw_addr_space_t as)
{
	if(as->type != UNW_REMOTE)
	{
		ABORT("unw_flush_caches was passed a non-remote address space");
		return;
	}
	unw_addr_space_remote* remote = (unw_addr_space_remote*)as;
	remote->ras->flushAllCaches();
	
	return;
}

EXPORT void unw_image_was_unloaded (unw_addr_space_t as, unw_word_t mh)
{
	if(as->type != UNW_REMOTE)
	{
		ABORT("unw_image_was_unloaded was passed a non-remote address space");
		return;
	}
	unw_addr_space_remote* remote = (unw_addr_space_remote*)as;
	remote->ras->flushCacheByMachHeader(mh);

	return;
}


EXPORT int unw_set_caching_policy(unw_addr_space_t as, unw_caching_policy_t policy)
{
	if(as->type != UNW_REMOTE)
	{
		ABORT("unw_set_caching_policy was passed a non-remote address space");
		return UNW_EINVAL;
	}
	unw_addr_space_remote* remote = (unw_addr_space_remote*)as;
	return remote->ras->setCachingPolicy(policy);
}

EXPORT unw_addr_space_t unw_local_addr_space = (unw_addr_space_t)&sThisAddressSpace;

///
/// delete an address_space object
///
EXPORT void unw_destroy_addr_space(unw_addr_space_t asp)
{
    if(asp->type != UNW_REMOTE) {
        ABORT("unw_destroy_addr_space was passed a non-remote address space");
        return;
    }

    unw_addr_space_remote* remote = (unw_addr_space_remote*)asp;
    delete remote->ras;
}

EXPORT void unw_set_logging_level(unw_addr_space_t as, FILE *f, unw_log_level_t level)
{
    if (as->type != UNW_REMOTE) {
        ABORT("unw_set_logging_level was passed a non-remote address space");
        return;
    }

	unw_addr_space_remote* remote = (unw_addr_space_remote*)as;
	return remote->ras->setLoggingLevel(f, level);
}


EXPORT int unw_end_of_prologue_setup(unw_cursor_t* cursor, unw_word_t start, unw_word_t end, unw_word_t *endofprologue) 
{
    AbstractRemoteUnwindCursor* co = (AbstractRemoteUnwindCursor*)cursor;
    if (!co->remoteUnwindCursor())
        ABORT("unw_end_of_prologue_setup called with a non-remote unwind cursor.");

    return co->endOfPrologueInsns (start, end, endofprologue);
}


#endif // SUPPORT_REMOTE_UNWINDING

#pragma mark Dynamic unwinding API

#if !FOR_DYLD
///
/// SPI: walks cached dwarf entries
///
EXPORT void unw_iterate_dwarf_unwind_cache(void (*func)(unw_word_t ip_start, unw_word_t ip_end, unw_word_t fde, unw_word_t mh))
{
	DEBUG_PRINT_API("unw_iterate_dwarf_unwind_cache(func=%p)\n", func);
	DwarfFDECache<LocalAddressSpace>::iterateCacheEntries(func);
}
#endif // !FOR_DYLD

#if !FOR_DYLD
//
// IPI: for __register_frame()
//
void _unw_add_dynamic_fde(unw_word_t fde)
{
	CFI_Parser<LocalAddressSpace>::FDE_Info fdeInfo;
	CFI_Parser<LocalAddressSpace>::CIE_Info cieInfo;
	const char* message = CFI_Parser<LocalAddressSpace>::decodeFDE(sThisAddressSpace, (LocalAddressSpace::pint_t)fde, & fdeInfo, &cieInfo);
	if ( message == NULL ) {
		// dynamically registered FDEs don't have a mach_header group they are in.  Use fde as mh_group
		unw_word_t mh_group = fdeInfo.fdeStart;
		DwarfFDECache<LocalAddressSpace>::add(mh_group, fdeInfo.pcStart, fdeInfo.pcEnd, fdeInfo.fdeStart);
	}
	else {
		DEBUG_MESSAGE("_unw_add_dynamic_fde: bad fde: %s", message);
	}
}

//
// IPI: for __deregister_frame()
//
void _unw_remove_dynamic_fde(unw_word_t fde)
{
	// fde is own mh_group
	DwarfFDECache<LocalAddressSpace>::removeAllIn(fde);
}
#endif

}; // namespace lldb_private

#endif // __ppc__ || __i386__ ||  __x86_64__
