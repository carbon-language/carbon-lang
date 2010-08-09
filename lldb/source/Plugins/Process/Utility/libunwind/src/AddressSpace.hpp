/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- AddressSpace.hpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 
//
//	C++ interface to lower levels of libuwind 
//

#ifndef __ADDRESSSPACE_HPP__
#define __ADDRESSSPACE_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <mach-o/getsect.h>
#if !defined (SUPPORT_REMOTE_UNWINDING)
#include <mach-o/dyld_priv.h>
#endif
#include <mach/i386/thread_status.h>
#include <Availability.h>

#include "FileAbstraction.hpp"
#include "libunwind.h"
#include "InternalMacros.h"
#include "dwarf2.h"
#include "RemoteProcInfo.hpp"

#if defined (SUPPORT_REMOTE_UNWINDING)
bool _dyld_find_unwind_sections(void* addr, void* info)
{
    assert("unwinding with a non-remote process not supported.");
    return false;
}
#endif // SUPPORT_REMOTE_UNWINDING

namespace lldb_private {

///
/// LocalAddressSpace is used as a template parameter to UnwindCursor when unwinding a thread
/// in the same process.  It compiles away and making local unwinds very fast.
///
class LocalAddressSpace
{
public:
	
	#if __LP64__
		typedef uint64_t	pint_t;
		typedef  int64_t	sint_t;
	#else
		typedef uint32_t	pint_t;
		typedef  int32_t	sint_t;
	#endif
        int			    getBytes(pint_t addr, pint_t extent, uint8_t* buf) { memcpy(buf, (void*)addr, extent); return 1; }
		uint8_t			get8(pint_t addr)	{ return *((uint8_t*)addr); }
		uint16_t		get16(pint_t addr)	{ return *((uint16_t*)addr); }
		uint32_t		get32(pint_t addr)	{ return *((uint32_t*)addr); }
		uint64_t		get64(pint_t addr)	{ return *((uint64_t*)addr); }
		double			getDouble(pint_t addr)	{ return *((double*)addr); }
		v128			getVector(pint_t addr)	{ return *((v128*)addr); }

		uint8_t			get8(pint_t addr, int& err)	    { return *((uint8_t*)addr); err = 0; }
		uint16_t		get16(pint_t addr, int& err)	{ return *((uint16_t*)addr); err = 0; }
		uint32_t		get32(pint_t addr, int& err)	{ return *((uint32_t*)addr); err = 0; }
		uint64_t		get64(pint_t addr, int& err)	{ return *((uint64_t*)addr); err = 0; }
		double			getDouble(pint_t addr, int& err)	{ return *((double*)addr); err = 0; }
		v128			getVector(pint_t addr, int& err)	{ return *((v128*)addr); err = 0; }

		uintptr_t		getP(pint_t addr);
        uintptr_t       getP(pint_t addr, int &err);
        static uint64_t	getULEB128(pint_t& addr, pint_t end);
        static int64_t	getSLEB128(pint_t& addr, pint_t end);
	
		pint_t			getEncodedP(pint_t& addr, pint_t end, uint8_t encoding);
		bool			findFunctionName(pint_t addr, char* buf, size_t bufLen, unw_word_t* offset);
		bool			findUnwindSections(pint_t addr, pint_t& mh, pint_t& dwarfStart, pint_t& dwarfLen, pint_t& compactStart);

#if defined (SUPPORT_REMOTE_UNWINDING)
        RemoteProcInfo*         getRemoteProcInfo ()    { return NULL; }
        unw_accessors_t*        accessors()             { return NULL; }
        unw_addr_space_t        wrap()                  { return NULL; }
#endif
};

LocalAddressSpace sThisAddress;

inline uintptr_t LocalAddressSpace::getP(pint_t addr)
{
#if __LP64__
	return get64(addr);
#else
	return get32(addr);
#endif
}

inline uintptr_t LocalAddressSpace::getP(pint_t addr, int &err)
{
#if __LP64__
	return get64(addr);
#else
	return get32(addr);
#endif
    err = 0;
}

/* Read a ULEB128 into a 64-bit word.   */
inline uint64_t
LocalAddressSpace::getULEB128(pint_t& addr, pint_t end)
{
	const uint8_t* p = (uint8_t*)addr;
	const uint8_t* pend = (uint8_t*)end;
	uint64_t result = 0;
	int bit = 0;
	do  {
		uint64_t b;

		if ( p == pend )
			ABORT("truncated uleb128 expression");

		b = *p & 0x7f;

		if (bit >= 64 || b << bit >> bit != b) {
			ABORT("malformed uleb128 expression");
		}
		else {
			result |= b << bit;
			bit += 7;
		}
	} while ( *p++ >= 0x80 );
	addr = (pint_t)p;
	return result;
}

/* Read a SLEB128 into a 64-bit word.  */
inline int64_t
LocalAddressSpace::getSLEB128(pint_t& addr, pint_t end)
{
	const uint8_t* p = (uint8_t*)addr;
	int64_t result = 0;
	int bit = 0;
	uint8_t byte;
	do {
		byte = *p++;
		result |= ((byte & 0x7f) << bit);
		bit += 7;
	} while (byte & 0x80);
	// sign extend negative numbers
	if ( (byte & 0x40) != 0 )
		result |= (-1LL) << bit;
	addr = (pint_t)p;
	return result;
}

LocalAddressSpace::pint_t
LocalAddressSpace::getEncodedP(pint_t& addr, pint_t end, uint8_t encoding)
{
	pint_t startAddr = addr;
	const uint8_t* p = (uint8_t*)addr;
	pint_t result;
	
	// first get value
	switch (encoding & 0x0F) {
		case DW_EH_PE_ptr:
			result = getP(addr);
			p += sizeof(pint_t);
			addr = (pint_t)p;
			break;
		case DW_EH_PE_uleb128:
			result = getULEB128(addr, end);
			break;
		case DW_EH_PE_udata2:
			result = get16(addr);
			p += 2;
			addr = (pint_t)p;
			break;
		case DW_EH_PE_udata4:
			result = get32(addr);
			p += 4;
			addr = (pint_t)p;
			break;
		case DW_EH_PE_udata8:
			result = get64(addr);
			p += 8;
			addr = (pint_t)p;
			break;
		case DW_EH_PE_sleb128:
			result = getSLEB128(addr, end);
			break;
		case DW_EH_PE_sdata2:
			result = (int16_t)get16(addr);
			p += 2;
			addr = (pint_t)p;
			break;
		case DW_EH_PE_sdata4:
			result = (int32_t)get32(addr);
			p += 4;
			addr = (pint_t)p;
			break;
		case DW_EH_PE_sdata8:
			result = get64(addr);
			p += 8;
			addr = (pint_t)p;
			break;
		default:
			ABORT("unknown pointer encoding");
	}
	
	// then add relative offset
	switch ( encoding & 0x70 ) {
		case DW_EH_PE_absptr:
			// do nothing
			break;
		case DW_EH_PE_pcrel:
			result += startAddr;
			break;
		case DW_EH_PE_textrel:
			ABORT("DW_EH_PE_textrel pointer encoding not supported");
			break;
		case DW_EH_PE_datarel:
			ABORT("DW_EH_PE_datarel pointer encoding not supported");
			break;
		case DW_EH_PE_funcrel:
			ABORT("DW_EH_PE_funcrel pointer encoding not supported");
			break;
		case DW_EH_PE_aligned:
			ABORT("DW_EH_PE_aligned pointer encoding not supported");
			break;
		default:
			ABORT("unknown pointer encoding");
			break;
	}
	
	if ( encoding & DW_EH_PE_indirect )
		result = getP(result);
	
	return result;
}


inline bool LocalAddressSpace::findUnwindSections(pint_t addr, pint_t& mh, pint_t& dwarfStart, pint_t& dwarfLen, pint_t& compactStart)
{
#if !defined (SUPPORT_REMOTE_UNWINDING)
	dyld_unwind_sections info;
	if ( _dyld_find_unwind_sections((void*)addr, &info) ) {
		mh				= (pint_t)info.mh;
		dwarfStart		= (pint_t)info.dwarf_section;
		dwarfLen		= (pint_t)info.dwarf_section_length;
		compactStart	= (pint_t)info.compact_unwind_section;
		return true;
	}
#else
    assert("unwinding with a non-remote process not supported.");
#endif
	return false;
}


inline bool	LocalAddressSpace::findFunctionName(pint_t addr, char* buf, size_t bufLen, unw_word_t* offset)
{
	dl_info dyldInfo;
	if ( dladdr((void*)addr, &dyldInfo) ) {
		if ( dyldInfo.dli_sname != NULL ) {
			strlcpy(buf, dyldInfo.dli_sname, bufLen);
			*offset = (addr - (pint_t)dyldInfo.dli_saddr);
			return true;
		}
	}
	return false;
}

#if defined (SUPPORT_REMOTE_UNWINDING)
///
/// OtherAddressSpace is used as a template parameter to UnwindCursor when unwinding a thread
/// in the another process.  The other process can be a different endianness and a different
/// pointer size and is handled by the P template parameter.  
///
template <typename P>
class OtherAddressSpace
{
public:
        OtherAddressSpace (unw_addr_space_t remote_addr_space, void* arg) : fAddrSpace ((unw_addr_space_remote *)remote_addr_space), fArg(arg)
        {
          if (fAddrSpace->type != UNW_REMOTE)
            ABORT("OtherAddressSpace ctor called with non-remote address space.");
          fRemoteProcInfo = fAddrSpace->ras;
        }

		typedef typename P::uint_t	pint_t;
		typedef typename P::int_t	sint_t;

	    int			    getBytes(pint_t addr, pint_t extent, uint8_t* buf)   { return fRemoteProcInfo->getBytes (addr, extent, buf, fArg); }
        uint8_t         get8(pint_t addr)                                    { return fRemoteProcInfo->get8(addr, fArg); }
        uint16_t        get16(pint_t addr)                                   { return fRemoteProcInfo->get16(addr, fArg); }
        uint32_t        get32(pint_t addr)                                   { return fRemoteProcInfo->get32(addr, fArg); }
        uint64_t        get64(pint_t addr)                                   { return fRemoteProcInfo->get64(addr, fArg); }
        pint_t          getP(pint_t addr)                                    { return fRemoteProcInfo->getP(addr, fArg); }

        uint8_t         get8(pint_t addr, int& err)                          { return fRemoteProcInfo->get8(addr, err, fArg); }
        uint16_t        get16(pint_t addr, int& err)                         { return fRemoteProcInfo->get16(addr, err, fArg); }
        uint32_t        get32(pint_t addr, int& err)                         { return fRemoteProcInfo->get32(addr, err, fArg); }
        uint64_t        get64(pint_t addr, int& err)                         { return fRemoteProcInfo->get64(addr, err, fArg); }
        pint_t          getP(pint_t addr, int &err)                          { return fRemoteProcInfo->getP(addr, err, fArg); }

        uint64_t        getULEB128(pint_t& addr, pint_t end)                 { return fRemoteProcInfo->getULEB128 (addr, end, fArg); }
        int64_t         getSLEB128(pint_t& addr, pint_t end)                 { return fRemoteProcInfo->getSLEB128 (addr, end, fArg); }
        pint_t          getEncodedP(pint_t& addr, pint_t end, uint8_t encoding);
        double          getDouble(pint_t addr);
        v128            getVector(pint_t addr);
        bool            findFunctionName(pint_t addr, char* buf, size_t bufLen, unw_word_t* offset);
        bool            findFunctionExtent(pint_t addr, unw_word_t* begin, unw_word_t* end);
        bool            findUnwindSections(pint_t addr, pint_t& mh, pint_t& eh_frame_start, pint_t& eh_frame_len, pint_t& compactStart);
        RemoteProcInfo* getRemoteProcInfo () { return fRemoteProcInfo; }
        unw_accessors_t*    accessors()   { return fRemoteProcInfo->getAccessors(); }
        unw_addr_space_t    wrap()        { return (unw_addr_space_t) fAddrSpace; }
private:
		void*			localCopy(pint_t addr);
        unw_addr_space_remote *fAddrSpace;
        RemoteProcInfo* fRemoteProcInfo;
        void*           fArg;
};

template <typename P>
typename OtherAddressSpace<P>::pint_t OtherAddressSpace<P>::getEncodedP(pint_t& addr, pint_t end, uint8_t encoding)
{
	pint_t startAddr = addr;
	pint_t p = addr;
	pint_t result;
	
	// first get value
	switch (encoding & 0x0F) {
		case DW_EH_PE_ptr:
			result = fRemoteProcInfo->getP(addr, fArg);
			p += sizeof(pint_t);
			addr = p;
			break;
		case DW_EH_PE_uleb128:
			result = fRemoteProcInfo->getULEB128(addr, end, fArg);
			break;
		case DW_EH_PE_udata2:
			result = fRemoteProcInfo->get16(addr, fArg);
			p += 2;
			addr = p;
			break;
		case DW_EH_PE_udata4:
			result = fRemoteProcInfo->get32(addr, fArg);
			p += 4;
			addr = p;
			break;
		case DW_EH_PE_udata8:
			result = fRemoteProcInfo->get64(addr, fArg);
			p += 8;
			addr = p;
			break;
		case DW_EH_PE_sleb128:
			result = fRemoteProcInfo->getSLEB128(addr, end, fArg);
			break;
		case DW_EH_PE_sdata2:
			result = (int16_t)fRemoteProcInfo->get16(addr, fArg);
			p += 2;
			addr = p;
			break;
		case DW_EH_PE_sdata4:
			result = (int32_t)fRemoteProcInfo->get32(addr, fArg);
			p += 4;
			addr = p;
			break;
		case DW_EH_PE_sdata8:
			result = fRemoteProcInfo->get64(addr, fArg);
			p += 8;
			addr = p;
			break;
		default:
			ABORT("unknown pointer encoding");
	}
	
	// then add relative offset
	switch ( encoding & 0x70 ) {
		case DW_EH_PE_absptr:
			// do nothing
			break;
		case DW_EH_PE_pcrel:
			result += startAddr;
			break;
		case DW_EH_PE_textrel:
			ABORT("DW_EH_PE_textrel pointer encoding not supported");
			break;
		case DW_EH_PE_datarel:
			ABORT("DW_EH_PE_datarel pointer encoding not supported");
			break;
		case DW_EH_PE_funcrel:
			ABORT("DW_EH_PE_funcrel pointer encoding not supported");
			break;
		case DW_EH_PE_aligned:
			ABORT("DW_EH_PE_aligned pointer encoding not supported");
			break;
		default:
			ABORT("unknown pointer encoding");
			break;
	}
	
	if ( encoding & DW_EH_PE_indirect )
		result = fRemoteProcInfo->getP(result, fArg);
	
	return result;
}

template <typename P>
double OtherAddressSpace<P>::getDouble(pint_t addr)
{
    return fRemoteProcInfo->getDouble(addr, fArg);
}

template <typename P>
v128 OtherAddressSpace<P>::getVector(pint_t addr)
{
    return fRemoteProcInfo->getVector(addr, fArg);
}

template <typename P>
bool OtherAddressSpace<P>::findUnwindSections(pint_t addr, pint_t& mh, pint_t& eh_frame_start, pint_t& eh_frame_len, pint_t& compactStart)
{
    compactStart = 0;
    uint64_t t_mh, t_text_start, t_text_end, t_eh_frame_start, t_eh_frame_len, t_compact_start;
    if (fRemoteProcInfo->getImageAddresses (addr, t_mh, t_text_start, t_text_end, t_eh_frame_start, t_eh_frame_len, t_compact_start, fArg))
      {
        mh = t_mh;
        eh_frame_start = t_eh_frame_start;
        eh_frame_len = t_eh_frame_len;
        compactStart = t_compact_start;
        return true;
      }
    return false;
}

template <typename P>
bool OtherAddressSpace<P>::findFunctionName(pint_t addr, char* buf, size_t bufLen, unw_word_t* offset)
{
  return fRemoteProcInfo->findFunctionName (addr, buf, bufLen, offset, fArg);
}

#endif // SUPPORT_REMOTE_UNWINDING


} // namespace lldb_private 



#endif // __ADDRESSSPACE_HPP__
