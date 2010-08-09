/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- UnwindCursor.hpp ----------------------------------------*- C++ -*-===//
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

#ifndef __UNWINDCURSOR_HPP__
#define __UNWINDCURSOR_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdarg.h>

#include "libunwind.h"

#include "AddressSpace.hpp"
#include "Registers.hpp"
#include "DwarfInstructions.hpp"

#include "AssemblyParser.hpp"
#include "AssemblyInstructions.hpp"
#include "RemoteProcInfo.hpp"
#include "ArchDefaultUnwinder.hpp"
#include "RemoteDebuggerDummyUnwinder.hpp"

#include "CompactUnwinder.hpp"
#include "InternalMacros.h"

// private keymgr stuff
#define KEYMGR_GCC3_DW2_OBJ_LIST   302  
extern "C" {
	extern void	 _keymgr_set_and_unlock_processwide_ptr(int key, void* ptr);
	extern void* _keymgr_get_and_lock_processwide_ptr(int key);
};

// undocumented libgcc "struct object"
struct libgcc_object 
{
	void*			start;
	void*			unused1;
	void*			unused2;
	void*			fde;
	unsigned long	encoding;
	void*			fde_end;
	libgcc_object*	next;
};

// undocumented libgcc "struct km_object_info" referenced by KEYMGR_GCC3_DW2_OBJ_LIST
struct libgcc_object_info {
  struct libgcc_object*		seen_objects;
  struct libgcc_object*		unseen_objects;
  unsigned					spare[2];
};




namespace lldb_private {

#if !FOR_DYLD
template <typename A>
class DwarfFDECache 
{
public:
	typedef typename A::pint_t	pint_t;
	static pint_t					findFDE(pint_t mh, pint_t pc);
	static void						add(pint_t mh, pint_t ip_start, pint_t ip_end, pint_t fde);
	static void						removeAllIn(pint_t mh);
	static void						iterateCacheEntries(void (*func)(unw_word_t ip_start, unw_word_t ip_end, unw_word_t fde, unw_word_t mh));
private:
	static void						dyldUnloadHook(const struct mach_header* mh, intptr_t vmaddr_slide);
	
	struct entry { pint_t mh; pint_t ip_start; pint_t ip_end; pint_t fde; };

	// these fields are all static to avoid needing an initializer
	// there is only one instance of this class per process
	static pthread_rwlock_t			fgLock;	
	static bool						fgRegisteredForDyldUnloads;	
	// can't use std::vector<> here because this code must live in libSystem.dylib (which is below libstdc++.dylib)
	static entry*					fgBuffer;
	static entry*					fgBufferUsed;
	static entry*					fgBufferEnd;
	static entry					fgInitialBuffer[64];
};

template <typename A> typename DwarfFDECache<A>::entry* DwarfFDECache<A>::fgBuffer		= fgInitialBuffer;
template <typename A> typename DwarfFDECache<A>::entry* DwarfFDECache<A>::fgBufferUsed	= fgInitialBuffer;
template <typename A> typename DwarfFDECache<A>::entry* DwarfFDECache<A>::fgBufferEnd	= &fgInitialBuffer[64];
template <typename A> typename DwarfFDECache<A>::entry  DwarfFDECache<A>::fgInitialBuffer[64];

template <typename A>
pthread_rwlock_t DwarfFDECache<A>::fgLock = PTHREAD_RWLOCK_INITIALIZER;

template <typename A> 
bool DwarfFDECache<A>::fgRegisteredForDyldUnloads = false;


template <typename A>
typename A::pint_t DwarfFDECache<A>::findFDE(pint_t mh, pint_t pc)
{
	pint_t result = NULL;
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_rdlock(&fgLock));
	for(entry* p=fgBuffer; p < fgBufferUsed; ++p) {
		if ( (mh == p->mh) || (mh == 0) ) {
			if ( (p->ip_start <= pc) && (pc < p->ip_end) ) {
				result = p->fde;
				break;
			}
		}
	}
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_unlock(&fgLock));
	//fprintf(stderr, "DwarfFDECache::findFDE(mh=0x%llX, pc=0x%llX) => 0x%llX\n", (uint64_t)mh, (uint64_t)pc, (uint64_t)result);
	return result;
}

template <typename A>
void DwarfFDECache<A>::add(pint_t mh, pint_t ip_start, pint_t ip_end, pint_t fde)
{
	//fprintf(stderr, "DwarfFDECache::add(mh=0x%llX, ip_start=0x%llX, ip_end=0x%llX, fde=0x%llX) pthread=%p\n", 
	//		(uint64_t)mh, (uint64_t)ip_start, (uint64_t)ip_end, (uint64_t)fde, pthread_self());
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_wrlock(&fgLock));
	if ( fgBufferUsed >= fgBufferEnd ) {
		int oldSize = fgBufferEnd - fgBuffer;
		int newSize = oldSize*4;
		entry* newBuffer = (entry*)malloc(newSize*sizeof(entry));	// can't use operator new in libSystem.dylib
		memcpy(newBuffer, fgBuffer, oldSize*sizeof(entry));
		//fprintf(stderr, "DwarfFDECache::add() growing buffer to %d\n",  newSize);
		if ( fgBuffer != fgInitialBuffer )
			free(fgBuffer);
		fgBuffer = newBuffer;
		fgBufferUsed = &newBuffer[oldSize];
		fgBufferEnd = &newBuffer[newSize];
	}
	fgBufferUsed->mh = mh;
	fgBufferUsed->ip_start = ip_start;
	fgBufferUsed->ip_end = ip_end;
	fgBufferUsed->fde = fde;
	++fgBufferUsed;
#if !defined (SUPPORT_REMOTE_UNWINDING)
	if ( !fgRegisteredForDyldUnloads ) {
		_dyld_register_func_for_remove_image(&dyldUnloadHook);
		fgRegisteredForDyldUnloads = true;
	}
#endif
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_unlock(&fgLock));
}



template <typename A>
void DwarfFDECache<A>::removeAllIn(pint_t mh)
{
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_wrlock(&fgLock));
	entry* d=fgBuffer;
	for(const entry* s=fgBuffer; s < fgBufferUsed; ++s) {
		if ( s->mh != mh ) {
			if ( d != s ) 
				*d = *s;
			++d;
		}
	}
	fgBufferUsed = d;
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_unlock(&fgLock));
}


template <typename A>
void DwarfFDECache<A>::dyldUnloadHook(const struct mach_header* mh, intptr_t vmaddr_slide)
{
#if !defined (SUPPORT_REMOTE_UNWINDING)
	removeAllIn((pint_t)mh);
#endif
}

template <typename A>
void DwarfFDECache<A>::iterateCacheEntries(void (*func)(unw_word_t ip_start, unw_word_t ip_end, unw_word_t fde, unw_word_t mh))
{
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_wrlock(&fgLock));
	for(entry* p=fgBuffer; p < fgBufferUsed; ++p) {
		(*func)(p->ip_start, p->ip_end, p->fde, p->mh);
	}
	DEBUG_LOG_NON_ZERO(::pthread_rwlock_unlock(&fgLock));
}
#endif // !FOR_DYLD




#define arrayoffsetof(type, index, field) ((size_t)(&((type *)0)[index].field))

template <typename A>
class UnwindSectionHeader {
public:
					UnwindSectionHeader(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		version() const								INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, version)); }
	uint32_t		commonEncodingsArraySectionOffset() const	INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, commonEncodingsArraySectionOffset)); }
	uint32_t		commonEncodingsArrayCount() const			INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, commonEncodingsArrayCount)); }
	uint32_t		personalityArraySectionOffset() const		INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, personalityArraySectionOffset)); }
	uint32_t		personalityArrayCount() const				INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, personalityArrayCount)); }
	uint32_t		indexSectionOffset() const					INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, indexSectionOffset)); }
	uint32_t		indexCount() const							INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_section_header, indexCount)); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};

template <typename A>
class UnwindSectionIndexArray {
public:
					UnwindSectionIndexArray(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		functionOffset(int index) const					INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_section_header_index_entry, index, functionOffset)); }
	uint32_t		secondLevelPagesSectionOffset(int index) const	INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_section_header_index_entry, index, secondLevelPagesSectionOffset)); }
	uint32_t		lsdaIndexArraySectionOffset(int index) const	INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_section_header_index_entry, index, lsdaIndexArraySectionOffset)); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};


template <typename A>
class UnwindSectionRegularPageHeader {
public:
					UnwindSectionRegularPageHeader(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		kind() const				INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_regular_second_level_page_header, kind)); }
	uint16_t		entryPageOffset() const		INLINE { return fAddressSpace.get16(fAddr + offsetof(unwind_info_regular_second_level_page_header, entryPageOffset)); }
	uint16_t		entryCount() const			INLINE { return fAddressSpace.get16(fAddr + offsetof(unwind_info_regular_second_level_page_header, entryCount)); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};


template <typename A>
class UnwindSectionRegularArray {
public:
					UnwindSectionRegularArray(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		functionOffset(int index) const		INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_regular_second_level_entry, index, functionOffset)); }
	uint32_t		encoding(int index) const			INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_regular_second_level_entry, index, encoding)); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};


template <typename A>
class UnwindSectionCompressedPageHeader {
public:
					UnwindSectionCompressedPageHeader(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		kind() const				INLINE { return fAddressSpace.get32(fAddr + offsetof(unwind_info_compressed_second_level_page_header, kind)); }
	uint16_t		entryPageOffset() const		INLINE { return fAddressSpace.get16(fAddr + offsetof(unwind_info_compressed_second_level_page_header, entryPageOffset)); }
	uint16_t		entryCount() const			INLINE { return fAddressSpace.get16(fAddr + offsetof(unwind_info_compressed_second_level_page_header, entryCount)); }
	uint16_t		encodingsPageOffset() const	INLINE { return fAddressSpace.get16(fAddr + offsetof(unwind_info_compressed_second_level_page_header, encodingsPageOffset)); }
	uint16_t		encodingsCount() const		INLINE { return fAddressSpace.get16(fAddr + offsetof(unwind_info_compressed_second_level_page_header, encodingsCount)); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};


template <typename A>
class UnwindSectionCompressedArray {
public:
					UnwindSectionCompressedArray(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		functionOffset(int index) const		INLINE { return UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET( fAddressSpace.get32(fAddr + index*sizeof(uint32_t)) ); }
	uint16_t		encodingIndex(int index) const		INLINE { return UNWIND_INFO_COMPRESSED_ENTRY_ENCODING_INDEX( fAddressSpace.get32(fAddr + index*sizeof(uint32_t)) ); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};


template <typename A>
class UnwindSectionLsdaArray {
public:
					UnwindSectionLsdaArray(A& addressSpace, typename A::pint_t addr) : fAddressSpace(addressSpace), fAddr(addr) {}

	uint32_t		functionOffset(int index) const		INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_section_header_lsda_index_entry, index, functionOffset)); }
	int32_t			lsdaOffset(int index) const			INLINE { return fAddressSpace.get32(fAddr + arrayoffsetof(unwind_info_section_header_lsda_index_entry, index, lsdaOffset)); }
private:
	A&						fAddressSpace;
	typename A::pint_t		fAddr;
};


template <typename A, typename R>
class UnwindCursor
{
public:
						UnwindCursor(unw_context_t* context, A& as);
	virtual				~UnwindCursor() {}
	virtual bool		validReg(int);
	virtual uint64_t	getReg(int);
	virtual int	        getReg(int, uint64_t*);
	virtual int		    setReg(int, uint64_t);
	virtual bool		validFloatReg(int);
	virtual double		getFloatReg(int);
	virtual int		    getFloatReg(int, double*);
	virtual int 		setFloatReg(int, double);
    virtual int         step();
	virtual void		getInfo(unw_proc_info_t*);
	virtual void		jumpto();
	virtual const char*	getRegisterName(int num);
	virtual bool		isSignalFrame();
	virtual bool		getFunctionName(char* buf, size_t bufLen, unw_word_t* offset);
	virtual void		setInfoBasedOnIPRegister(bool isReturnAddress=false);

	void		        operator delete(void* p, size_t size) {}

protected:
	typedef typename A::pint_t		pint_t;	
	typedef uint32_t				EncodedUnwindInfo;

	virtual bool				getInfoFromCompactEncodingSection(pint_t pc, pint_t mh, pint_t unwindSectionStart);
	virtual bool				getInfoFromDwarfSection(pint_t pc, pint_t mh, pint_t ehSectionStart, uint32_t sectionLength, uint32_t sectionOffsetOfFDE);

	virtual int					stepWithDwarfFDE() 
							{ return DwarfInstructions<A,R>::stepWithDwarf(fAddressSpace, this->getReg(UNW_REG_IP), fInfo.unwind_info, fRegisters); }
	
    virtual int                 stepWithCompactEncoding() { R dummy; return stepWithCompactEncoding(dummy); }
	int					stepWithCompactEncoding(Registers_x86_64&) 
							{ return CompactUnwinder_x86_64<A>::stepWithCompactEncoding(fInfo.format, fInfo.start_ip, fAddressSpace, fRegisters); }
	int					stepWithCompactEncoding(Registers_x86&) 
							{ return CompactUnwinder_x86<A>::stepWithCompactEncoding(fInfo.format, fInfo.start_ip, fAddressSpace, fRegisters); }

#if FOR_DYLD
  #if __ppc__
	virtual bool				mustUseDwarf() const { return true; }
  #else
	virtual bool				mustUseDwarf() const { return false; }
  #endif
#else
    virtual bool                mustUseDwarf() const { R dummy; uint32_t offset; return dwarfWithOffset(dummy, offset); }
#endif

    virtual bool                dwarfWithOffset(uint32_t& offset) const { R dummy; return dwarfWithOffset(dummy, offset); }
	virtual bool				dwarfWithOffset(Registers_x86_64&, uint32_t& offset) const { 
							if ( (fInfo.format & UNWIND_X86_64_MODE_MASK) == UNWIND_X86_64_MODE_DWARF ) {
								offset = (fInfo.format & UNWIND_X86_64_DWARF_SECTION_OFFSET);
								return true;
							}
#if SUPPORT_OLD_BINARIES
							if ( (fInfo.format & UNWIND_X86_64_MODE_MASK) == UNWIND_X86_64_MODE_COMPATIBILITY ) {
								if ( (fInfo.format & UNWIND_X86_64_CASE_MASK) == UNWIND_X86_64_UNWIND_REQUIRES_DWARF ) {
									offset = 0;
									return true;
								}
							}
#endif
							return false;
						}
	virtual bool				dwarfWithOffset(Registers_x86&, uint32_t& offset) const { 
							if ( (fInfo.format & UNWIND_X86_MODE_MASK) == UNWIND_X86_MODE_DWARF ) {
								offset = (fInfo.format & UNWIND_X86_DWARF_SECTION_OFFSET);
								return true;
							}
#if SUPPORT_OLD_BINARIES
							if ( (fInfo.format & UNWIND_X86_MODE_MASK) == UNWIND_X86_MODE_COMPATIBILITY ) {
								if ( (fInfo.format & UNWIND_X86_CASE_MASK) == UNWIND_X86_UNWIND_REQUIRES_DWARF ) {
									offset = 0;
									return true;
								}
							}
#endif
							return false;
						}

    virtual compact_unwind_encoding_t       dwarfEncoding() const { R dummy; return dwarfEncoding(dummy); }
	virtual compact_unwind_encoding_t		dwarfEncoding(Registers_x86_64&) const { return UNWIND_X86_64_MODE_DWARF; }
	virtual compact_unwind_encoding_t		dwarfEncoding(Registers_x86&)	const { return UNWIND_X86_MODE_DWARF; }

	unw_proc_info_t				fInfo;
	R							fRegisters;
	A&							fAddressSpace;
	bool						fUnwindInfoMissing;
	bool						fIsSignalFrame;
};

typedef UnwindCursor<LocalAddressSpace,Registers_x86> AbstractUnwindCursor;

template <typename A, typename R>
UnwindCursor<A,R>::UnwindCursor(unw_context_t* context, A& as)
  : fRegisters(context), fAddressSpace(as), fUnwindInfoMissing(false), fIsSignalFrame(false)
{
	COMPILE_TIME_ASSERT( sizeof(UnwindCursor<A,R>) < sizeof(unw_cursor_t) );

	bzero(&fInfo, sizeof(fInfo));
}

template <typename A, typename R>
bool UnwindCursor<A,R>::validReg(int regNum) 
{ 
	return fRegisters.validRegister(regNum); 
}

template <typename A, typename R>
uint64_t UnwindCursor<A,R>::getReg(int regNum) 
{ 
	return fRegisters.getRegister(regNum); 
}

template <typename A, typename R>
int UnwindCursor<A,R>::getReg(int regNum, uint64_t *valp) 
{ 
	*valp = fRegisters.getRegister(regNum);
    return UNW_ESUCCESS;
}

template <typename A, typename R>
int UnwindCursor<A,R>::setReg(int regNum, uint64_t value) 
{ 
	fRegisters.setRegister(regNum, value); 
    return UNW_ESUCCESS;
}

template <typename A, typename R>
bool UnwindCursor<A,R>::validFloatReg(int regNum) 
{ 
	return fRegisters.validFloatRegister(regNum); 
}

template <typename A, typename R>
double UnwindCursor<A,R>::getFloatReg(int regNum) 
{ 
	return fRegisters.getFloatRegister(regNum); 
}

template <typename A, typename R>
int UnwindCursor<A,R>::getFloatReg(int regNum, double *valp) 
{ 
	*valp = fRegisters.getFloatRegister(regNum); 
    return UNW_ESUCCESS;
}

template <typename A, typename R>
int UnwindCursor<A,R>::setFloatReg(int regNum, double value) 
{ 
	fRegisters.setFloatRegister(regNum, value); 
    return UNW_ESUCCESS;
}

template <typename A, typename R>
void UnwindCursor<A,R>::jumpto() 
{ 
#if !defined (SUPPORT_REMOTE_UNWINDING)
	fRegisters.jumpto(); 
#endif
}

template <typename A, typename R>
const char* UnwindCursor<A,R>::getRegisterName(int regNum) 
{ 
	return fRegisters.getRegisterName(regNum); 
}

template <typename A, typename R>
bool UnwindCursor<A,R>::isSignalFrame() 
{ 
	 return fIsSignalFrame;
}


template <typename A, typename R>
bool UnwindCursor<A,R>::getInfoFromDwarfSection(pint_t pc, pint_t mh, pint_t ehSectionStart, uint32_t sectionLength, uint32_t sectionOffsetOfFDE)
{
	typename CFI_Parser<A>::FDE_Info fdeInfo;
	typename CFI_Parser<A>::CIE_Info cieInfo;
	bool foundFDE = false;
	bool foundInCache = false;
	// if compact encoding table gave offset into dwarf section, go directly there
	if ( sectionOffsetOfFDE != 0 ) {
		foundFDE = CFI_Parser<A>::findFDE(fAddressSpace, pc, ehSectionStart, sectionLength, ehSectionStart+sectionOffsetOfFDE, &fdeInfo, &cieInfo);
	}
#if !FOR_DYLD
	if ( !foundFDE ) {
		// otherwise, search cache of previously found FDEs
		pint_t cachedFDE = DwarfFDECache<A>::findFDE(mh, pc);
		//fprintf(stderr, "getInfoFromDwarfSection(pc=0x%llX) cachedFDE=0x%llX\n", (uint64_t)pc, (uint64_t)cachedFDE);
		if ( cachedFDE != 0 ) {
			foundFDE = CFI_Parser<A>::findFDE(fAddressSpace, pc, ehSectionStart, sectionLength, cachedFDE, &fdeInfo, &cieInfo);
			foundInCache = foundFDE;
			//fprintf(stderr, "cachedFDE=0x%llX, foundInCache=%d\n", (uint64_t)cachedFDE, foundInCache);
		}
	}
#endif
	if ( !foundFDE ) {
		// still not found, do full scan of __eh_frame section
		foundFDE = CFI_Parser<A>::findFDE(fAddressSpace, pc, ehSectionStart, sectionLength, 0, &fdeInfo, &cieInfo);
	}
	if ( foundFDE ) {
		typename CFI_Parser<A>::PrologInfo prolog;
		if ( CFI_Parser<A>::parseFDEInstructions(fAddressSpace, fdeInfo, cieInfo, pc, &prolog) ) {
			// save off parsed FDE info
			fInfo.start_ip			= fdeInfo.pcStart;
			fInfo.end_ip			= fdeInfo.pcEnd;
			fInfo.lsda				= fdeInfo.lsda;
			fInfo.handler			= cieInfo.personality;
			fInfo.gp				= prolog.spExtraArgSize;  // some frameless functions need SP altered when resuming in function
			fInfo.flags				= 0;
			fInfo.format			= dwarfEncoding();  
			fInfo.unwind_info		= fdeInfo.fdeStart;
			fInfo.unwind_info_size	= fdeInfo.fdeLength;
			fInfo.extra				= (unw_word_t)mh;
			if ( !foundInCache && (sectionOffsetOfFDE == 0) ) {
				// don't add to cache entries the compact encoding table can find quickly
				//fprintf(stderr, "getInfoFromDwarfSection(pc=0x%0llX), mh=0x%llX, start_ip=0x%0llX, fde=0x%0llX, personality=0x%0llX\n", 
				//	(uint64_t)pc, (uint64_t)mh, fInfo.start_ip, fInfo.unwind_info, fInfo.handler);
#if !FOR_DYLD
				DwarfFDECache<A>::add(mh, fdeInfo.pcStart, fdeInfo.pcEnd, fdeInfo.fdeStart);
#endif
			}
			return true;
		}
	}
	//DEBUG_MESSAGE("can't find/use FDE for pc=0x%llX\n", (uint64_t)pc);
	return false;
}

template <typename A, typename R>
bool UnwindCursor<A,R>::getInfoFromCompactEncodingSection(pint_t pc, pint_t mh, pint_t unwindSectionStart)
{	
	const bool log = false;
	if ( log ) fprintf(stderr, "getInfoFromCompactEncodingSection(pc=0x%llX, mh=0x%llX)\n", (uint64_t)pc, (uint64_t)mh);
	
	const UnwindSectionHeader<A> sectionHeader(fAddressSpace, unwindSectionStart);
	if ( sectionHeader.version() != UNWIND_SECTION_VERSION )
		return false;
	
	// do a binary search of top level index to find page with unwind info
	uint32_t targetFunctionOffset = pc - mh;
	const UnwindSectionIndexArray<A> topIndex(fAddressSpace, unwindSectionStart + sectionHeader.indexSectionOffset());
	uint32_t low = 0;
	uint32_t high = sectionHeader.indexCount();
	const uint32_t last_section_header = high - 1;
	while ( low < high ) {
		uint32_t mid = (low + high)/2;
		//if ( log ) fprintf(stderr, "\tmid=%d, low=%d, high=%d, *mid=0x%08X\n", mid, low, high, topIndex.functionOffset(mid));
		if ( topIndex.functionOffset(mid) <= targetFunctionOffset ) {
			if ( (mid == last_section_header) || (topIndex.functionOffset(mid+1) > targetFunctionOffset) ) {
				low = mid;
				break;
			}
			else {
				low = mid+1;
			}
		}
		else {
			high = mid;
		}
	}
	const uint32_t firstLevelFunctionOffset = topIndex.functionOffset(low);
	const uint32_t firstLevelNextPageFunctionOffset = topIndex.functionOffset(low+1);
	const pint_t secondLevelAddr    = unwindSectionStart+topIndex.secondLevelPagesSectionOffset(low);
	const pint_t lsdaArrayStartAddr = unwindSectionStart+topIndex.lsdaIndexArraySectionOffset(low);
	const pint_t lsdaArrayEndAddr   = unwindSectionStart+topIndex.lsdaIndexArraySectionOffset(low+1);
	if ( log ) fprintf(stderr, "\tfirst level search for result index=%d to secondLevelAddr=0x%llX\n", 
			low, (uint64_t)secondLevelAddr);
	// do a binary search of second level page index
	uint32_t encoding = 0;
	pint_t funcStart = 0;
	pint_t funcEnd = 0;
	pint_t lsda = 0;
	pint_t personality = 0;
	uint32_t pageKind = fAddressSpace.get32(secondLevelAddr);
	if ( pageKind == UNWIND_SECOND_LEVEL_REGULAR ) {
		// regular page
		UnwindSectionRegularPageHeader<A> pageHeader(fAddressSpace, secondLevelAddr);
		UnwindSectionRegularArray<A> pageIndex(fAddressSpace, secondLevelAddr + pageHeader.entryPageOffset());
		// binary search looks for entry with e where index[e].offset <= pc < index[e+1].offset
		if ( log ) fprintf(stderr, "\tbinary search for targetFunctionOffset=0x%08llX in regular page starting at secondLevelAddr=0x%llX\n", 
			(uint64_t)targetFunctionOffset, (uint64_t)secondLevelAddr);
		low = 0;
		high = pageHeader.entryCount();
		while ( low < high ) {
			uint32_t mid = (low + high)/2;
			if ( pageIndex.functionOffset(mid) <= targetFunctionOffset ) {
				if ( mid == (uint32_t)(pageHeader.entryCount()-1) ) {
					// at end of table
					low = mid;
					funcEnd = firstLevelNextPageFunctionOffset + mh;
					break;
				}
				else if ( pageIndex.functionOffset(mid+1) > targetFunctionOffset ) {
					// next is too big, so we found it
					low = mid;
					funcEnd = pageIndex.functionOffset(low+1) + mh;
					break;
				}
				else {
					low = mid+1;
				}
			}
			else {
				high = mid;
			}
		}
		encoding  = pageIndex.encoding(low);
		funcStart = pageIndex.functionOffset(low) + mh;
		if ( pc < funcStart  ) {
			if ( log ) fprintf(stderr, "\tpc not in table, pc=0x%llX, funcStart=0x%llX, funcEnd=0x%llX\n", (uint64_t)pc, (uint64_t)funcStart, (uint64_t)funcEnd);
			return false;
		}
		if ( pc > funcEnd ) {
			if ( log ) fprintf(stderr, "\tpc not in table, pc=0x%llX, funcStart=0x%llX, funcEnd=0x%llX\n", (uint64_t)pc, (uint64_t)funcStart, (uint64_t)funcEnd);
			return false;
		}
	}
	else if ( pageKind == UNWIND_SECOND_LEVEL_COMPRESSED ) {
		// compressed page
		UnwindSectionCompressedPageHeader<A> pageHeader(fAddressSpace, secondLevelAddr);
		UnwindSectionCompressedArray<A> pageIndex(fAddressSpace, secondLevelAddr + pageHeader.entryPageOffset());
		const uint32_t targetFunctionPageOffset = targetFunctionOffset - firstLevelFunctionOffset;
		// binary search looks for entry with e where index[e].offset <= pc < index[e+1].offset
		if ( log ) fprintf(stderr, "\tbinary search of compressed page starting at secondLevelAddr=0x%llX\n", (uint64_t)secondLevelAddr);
		low = 0;
		const uint32_t last_page_header = pageHeader.entryCount() - 1;
		high = pageHeader.entryCount();
		while ( low < high ) {
			uint32_t mid = (low + high)/2;
			if ( pageIndex.functionOffset(mid) <= targetFunctionPageOffset ) {
				if ( (mid == last_page_header) || (pageIndex.functionOffset(mid+1) > targetFunctionPageOffset) ) {
					low = mid;
					break;
				}
				else {
					low = mid+1;
				}
			}
			else {
				high = mid;
			}
		}
		funcStart = pageIndex.functionOffset(low) + firstLevelFunctionOffset + mh;
		if ( low < last_page_header )
			funcEnd = pageIndex.functionOffset(low+1) + firstLevelFunctionOffset + mh;
		else
			funcEnd = firstLevelNextPageFunctionOffset + mh;
		if ( pc < funcStart  ) {
			DEBUG_MESSAGE("malformed __unwind_info, pc=0x%llX not in second level compressed unwind table. funcStart=0x%llX\n", (uint64_t)pc, (uint64_t)funcStart);
			return false;
		}
		if ( pc > funcEnd ) {
			DEBUG_MESSAGE("malformed __unwind_info, pc=0x%llX not in second level compressed unwind table. funcEnd=0x%llX\n", (uint64_t)pc, (uint64_t)funcEnd);
			return false;
		}
		uint16_t encodingIndex = pageIndex.encodingIndex(low);
		if ( encodingIndex < sectionHeader.commonEncodingsArrayCount() ) {
			// encoding is in common table in section header
			encoding = fAddressSpace.get32(unwindSectionStart+sectionHeader.commonEncodingsArraySectionOffset()+encodingIndex*sizeof(uint32_t));
		}
		else {
			// encoding is in page specific table
			uint16_t pageEncodingIndex = encodingIndex-sectionHeader.commonEncodingsArrayCount();
			encoding = fAddressSpace.get32(secondLevelAddr+pageHeader.encodingsPageOffset()+pageEncodingIndex*sizeof(uint32_t));
		}
	}
	else {
		DEBUG_MESSAGE("malformed __unwind_info at 0x%0llX bad second level page\n", (uint64_t)unwindSectionStart);
		return false;
	}

	// look up LSDA, if encoding says function has one
	if ( encoding & UNWIND_HAS_LSDA ) {
		UnwindSectionLsdaArray<A>  lsdaIndex(fAddressSpace, lsdaArrayStartAddr);
		uint32_t funcStartOffset = funcStart - mh;
		low = 0;
		high = (lsdaArrayEndAddr-lsdaArrayStartAddr)/sizeof(unwind_info_section_header_lsda_index_entry);
		// binary search looks for entry with exact match for functionOffset
		if ( log ) fprintf(stderr, "\tbinary search of lsda table for targetFunctionOffset=0x%08X\n", funcStartOffset);
		while ( low < high ) {
			uint32_t mid = (low + high)/2;
			if ( lsdaIndex.functionOffset(mid) == funcStartOffset ) {
				lsda = lsdaIndex.lsdaOffset(mid) + mh;
				break;
			}
			else if ( lsdaIndex.functionOffset(mid) < funcStartOffset ) {
				low = mid+1;
			}
			else {
				high = mid;
			}
		}
		if ( lsda == 0 ) {
			DEBUG_MESSAGE("found encoding 0x%08X with HAS_LSDA bit set for pc=0x%0llX, but lsda table has no entry\n", encoding, (uint64_t)pc);
			return false;
		}
	}

	// extact personality routine, if encoding says function has one
	uint32_t personalityIndex = (encoding & UNWIND_PERSONALITY_MASK) >> (__builtin_ctz(UNWIND_PERSONALITY_MASK));	
	if ( personalityIndex != 0 ) {
		--personalityIndex; // change 1-based to zero-based index
		if ( personalityIndex > sectionHeader.personalityArrayCount() ) {
			DEBUG_MESSAGE("found encoding 0x%08X with personality index %d, but personality table has only %d entires\n", 
							encoding, personalityIndex, sectionHeader.personalityArrayCount());
			return false;
		}
		int32_t personalityDelta = fAddressSpace.get32(unwindSectionStart+sectionHeader.personalityArraySectionOffset()+personalityIndex*sizeof(uint32_t));
		pint_t personalityPointer = personalityDelta + mh;
		personality = fAddressSpace.getP(personalityPointer);
		if (log ) fprintf(stderr, "getInfoFromCompactEncodingSection(pc=0x%llX), personalityDelta=0x%08X, personality=0x%08llX\n", 
			(uint64_t)pc, personalityDelta, (uint64_t)personality);
	}
	
	if (log ) fprintf(stderr, "getInfoFromCompactEncodingSection(pc=0x%llX), encoding=0x%08X, lsda=0x%08llX for funcStart=0x%llX\n", 
						(uint64_t)pc, encoding, (uint64_t)lsda, (uint64_t)funcStart);
	fInfo.start_ip			= funcStart; 
	fInfo.end_ip			= funcEnd;
	fInfo.lsda				= lsda; 
	fInfo.handler			= personality;
	fInfo.gp				= 0;
	fInfo.flags				= 0;
	fInfo.format			= encoding;
	fInfo.unwind_info		= 0;
	fInfo.unwind_info_size	= 0;
	fInfo.extra				= mh;
	return true;
}

template <typename A, typename R>
void UnwindCursor<A,R>::setInfoBasedOnIPRegister(bool isReturnAddress)
{
	pint_t pc = this->getReg(UNW_REG_IP);
	
	// if the last line of a function is a "throw" the compile sometimes
	// emits no instructions after the call to __cxa_throw.  This means 
	// the return address is actually the start of the next function.
	// To disambiguate this, back up the pc when we know it is a return
	// address.  
	if ( isReturnAddress ) 
		--pc;
	
	// ask address space object to find unwind sections for this pc
	pint_t mh;
	pint_t dwarfStart;
	pint_t dwarfLength;
	pint_t compactStart;
	if ( fAddressSpace.findUnwindSections(pc, mh, dwarfStart, dwarfLength, compactStart) ) {
		// if there is a compact unwind encoding table, look there first
		if ( compactStart != 0 ) {
			if ( this->getInfoFromCompactEncodingSection(pc, mh, compactStart) ) {
#if !FOR_DYLD
				// found info in table, done unless encoding says to use dwarf
				uint32_t offsetInDwarfSection;
				if ( (dwarfStart != 0) && dwarfWithOffset(offsetInDwarfSection) ) {
					if ( this->getInfoFromDwarfSection(pc, mh, dwarfStart, dwarfLength, offsetInDwarfSection) ) {
						// found info in dwarf, done
						return;
					}
				}
#endif
				// if unwind table has entry, but entry says there is no unwind info, note that
				if ( fInfo.format == 0 )
					fUnwindInfoMissing = true;

				// old compact encoding 
				if ( !mustUseDwarf() ) {
					return;
				}	
			}
		}
#if !FOR_DYLD || __ppc__
		// if there is dwarf unwind info, look there next
		if ( dwarfStart != 0 ) {
			if ( this->getInfoFromDwarfSection(pc, mh, dwarfStart, dwarfLength, 0) ) {
				// found info in dwarf, done
				return;
			}
		}
#endif
	}
	
#if !FOR_DYLD 
	// the PC is not in code loaded by dyld, look through __register_frame() registered FDEs
	pint_t cachedFDE = DwarfFDECache<A>::findFDE(0, pc);
	if ( cachedFDE != 0 ) {
		typename CFI_Parser<A>::FDE_Info fdeInfo;
		typename CFI_Parser<A>::CIE_Info cieInfo;
		const char* msg = CFI_Parser<A>::decodeFDE(fAddressSpace, cachedFDE, &fdeInfo, &cieInfo);
		if ( msg == NULL ) {
			typename CFI_Parser<A>::PrologInfo prolog;
			if ( CFI_Parser<A>::parseFDEInstructions(fAddressSpace, fdeInfo, cieInfo, pc, &prolog) ) {
				// save off parsed FDE info
				fInfo.start_ip			= fdeInfo.pcStart;
				fInfo.end_ip			= fdeInfo.pcEnd;
				fInfo.lsda				= fdeInfo.lsda;
				fInfo.handler			= cieInfo.personality;
				fInfo.gp				= prolog.spExtraArgSize;  // some frameless functions need SP altered when resuming in function
				fInfo.flags				= 0;
				fInfo.format			= dwarfEncoding();  
				fInfo.unwind_info		= fdeInfo.fdeStart;
				fInfo.unwind_info_size	= fdeInfo.fdeLength;
				fInfo.extra				= 0;
				return;
			}
		}
	}
	
#if !defined (SUPPORT_REMOTE_UNWINDING)
	// lastly check for old style keymgr registration of dynamically generated FDEs
	
	// acquire exclusive access to libgcc_object_info
	libgcc_object_info* head = (libgcc_object_info*)_keymgr_get_and_lock_processwide_ptr(KEYMGR_GCC3_DW2_OBJ_LIST);
	if ( head != NULL ) {
		// look at each FDE in keymgr
		for (libgcc_object* ob = head->unseen_objects; ob != NULL; ob = ob->next) {
			typename CFI_Parser<A>::FDE_Info fdeInfo;
			typename CFI_Parser<A>::CIE_Info cieInfo;
			const char* msg = CFI_Parser<A>::decodeFDE(fAddressSpace, (pint_t)ob->fde, &fdeInfo, &cieInfo);
			if ( msg == NULL ) {
				// see if this FDE is for a function that includes the pc we are looking for
				if ( (fdeInfo.pcStart <= pc) && (pc < fdeInfo.pcEnd) ) {
					typename CFI_Parser<A>::PrologInfo prolog;
					if ( CFI_Parser<A>::parseFDEInstructions(fAddressSpace, fdeInfo, cieInfo, pc, &prolog) ) {
						// save off parsed FDE info
						fInfo.start_ip			= fdeInfo.pcStart;
						fInfo.end_ip			= fdeInfo.pcEnd;
						fInfo.lsda				= fdeInfo.lsda;
						fInfo.handler			= cieInfo.personality;
						fInfo.gp				= prolog.spExtraArgSize;  // some frameless functions need SP altered when resuming in function
						fInfo.flags				= 0;
						fInfo.format			= dwarfEncoding();  
						fInfo.unwind_info		= fdeInfo.fdeStart;
						fInfo.unwind_info_size	= fdeInfo.fdeLength;
						fInfo.extra				= 0;
						_keymgr_set_and_unlock_processwide_ptr(KEYMGR_GCC3_DW2_OBJ_LIST, head);
						return;
					}
				}
			}
		}
	}
	// release libgcc_object_info 
	_keymgr_set_and_unlock_processwide_ptr(KEYMGR_GCC3_DW2_OBJ_LIST, head);
#endif // !SUPPORT_REMOTE_UNWINDING

#endif // !FOR_DYLD

	// no unwind info, flag that we can't reliable unwind
	fUnwindInfoMissing = true;
}


template <typename A, typename R>
int UnwindCursor<A,R>::step()
{
	// bottom of stack is defined as when no more unwind info
	if ( fUnwindInfoMissing )
			return UNW_STEP_END;

	// apply unwinding to register set
	int result;
	if ( this->mustUseDwarf() )
		result = this->stepWithDwarfFDE();
	else
		result = this->stepWithCompactEncoding();
	
	// update info based on new PC
	if ( result == UNW_STEP_SUCCESS ) {
		this->setInfoBasedOnIPRegister(true);
		if ( fUnwindInfoMissing )
			return UNW_STEP_END;
	}

	return result;
}


template <typename A, typename R>
void UnwindCursor<A,R>::getInfo(unw_proc_info_t* info)
{
	*info = fInfo;
}


template <typename A, typename R>
bool UnwindCursor<A,R>::getFunctionName(char* buf, size_t bufLen, unw_word_t* offset)
{
	return fAddressSpace.findFunctionName(this->getReg(UNW_REG_IP), buf, bufLen, offset);
}

#if defined (SUPPORT_REMOTE_UNWINDING)
template <typename A, typename R>
class RemoteUnwindCursor : UnwindCursor<A,R>
{
public:
    using UnwindCursor<A,R>::getReg;
    using UnwindCursor<A,R>::getFloatReg;

	typedef typename A::pint_t	pint_t;
                        RemoteUnwindCursor(A& as, unw_context_t* regs, void* arg);
    virtual bool        validReg(int);
    virtual int         getReg(int r, uint64_t*);
    virtual int         setReg(int, uint64_t);
    virtual bool        validFloatReg(int);
    virtual int         getFloatReg(int, double*);
    virtual int         setFloatReg(int, double);
    virtual const char* getRegisterName(int);
    virtual int         step();
    virtual void        setRemoteContext(void*);
    virtual bool        remoteUnwindCursor () const {return this->fAddressSpace.getRemoteProcInfo() != NULL; }
    virtual int         endOfPrologueInsns(unw_word_t, unw_word_t, unw_word_t*);
	void		        operator delete(void* p, size_t size) {}
private:
    virtual bool        caller_regno_to_unwind_regno (int, int&);

    bool                fIsLeafFrame;
    bool                fIsFirstFrame;
    void*               fArg;
};

typedef RemoteUnwindCursor<LocalAddressSpace,Registers_x86_64> AbstractRemoteUnwindCursor;

template <typename A, typename R>
RemoteUnwindCursor<A,R>::RemoteUnwindCursor(A& as, unw_context_t* regs, void* arg)
   : UnwindCursor<A,R>::UnwindCursor(regs, as), fIsLeafFrame(false), fIsFirstFrame (false), fArg(arg)
{
    COMPILE_TIME_ASSERT( sizeof(RemoteUnwindCursor<A,R>) < sizeof(unw_cursor_t) );
}

template <typename A, typename R>
bool RemoteUnwindCursor<A,R>::validReg(int r)
{
    int unwind_regno;
    if (!caller_regno_to_unwind_regno(r, unwind_regno))
        return false;
    return UnwindCursor<A,R>::fRegisters.validRegister(unwind_regno);
}

template <typename A, typename R>
int RemoteUnwindCursor<A,R>::getReg(int regNum, uint64_t *valp)
{
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo ();
    if (procinfo == NULL) {
       ABORT("getRemoteReg called with a local unwind, use getReg instead.");
    }

    RemoteRegisterMap *regmap = procinfo->getRegisterMap ();
    int unwind_regno;
    if (regmap->caller_regno_to_unwind_regno (regNum, unwind_regno) == false)
        return UNW_EBADREG;
    regNum = unwind_regno;

    // we always return nonvolatile registers.  If we have the entire register state available
    // for this frame then we can return any register requested.
    if (regmap->nonvolatile_reg_p (regNum) == true || fIsLeafFrame == true) {
        return this->UnwindCursor<A,R>::getReg (unwind_regno, valp);
    }
    return UNW_EREGUNAVAILABLE;
}

template <typename A, typename R>
int RemoteUnwindCursor<A,R>::setReg(int regNum, uint64_t val)
{
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo ();
    if (procinfo == NULL) {
       ABORT("setRemoteReg called with a local unwind, use setReg instead.");
    }

    RemoteRegisterMap *regmap = procinfo->getRegisterMap ();
    int unwind_regno;
    if (regmap->caller_regno_to_unwind_regno (regNum, unwind_regno) == false)
        return UNW_EBADREG;
    regNum = unwind_regno;

    // Only allow the registers to be set if the unwind cursor is pointing to the
    // first frame.  We need to track where registers were retrieved from in memory
    // in every other frame.  Until then, we prohibit register setting in all but
    // the first frame.
    if (fIsFirstFrame) {
        return this->setReg(unwind_regno, val);
    }
    return UNW_EREGUNAVAILABLE;
}

template <typename A, typename R>
bool RemoteUnwindCursor<A,R>::validFloatReg(int r)
{
    int unwind_regno;
    if (!caller_regno_to_unwind_regno(r, unwind_regno))
        return false;
    return UnwindCursor<A,R>::fRegisters.validFloatRegister(unwind_regno);
}

template <typename A, typename R>
int RemoteUnwindCursor<A,R>::getFloatReg(int regNum, double *valp)
{
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo ();
    if (procinfo == NULL) {
       ABORT("getRemoteReg called with a local unwind, use getReg instead.");
    }

    RemoteRegisterMap *regmap = procinfo->getRegisterMap ();
    int unwind_regno;
    if (regmap->caller_regno_to_unwind_regno (regNum, unwind_regno) == false)
        return UNW_EBADREG;
    regNum = unwind_regno;

    // we always return nonvolatile registers.  If we have the entire register state available
    // for this frame then we can return any register requested.
    if (regmap->nonvolatile_reg_p (regNum) == true || fIsLeafFrame == true) {
        return this->UnwindCursor<A,R>::getFloatReg (unwind_regno, valp);
    }
    return UNW_EREGUNAVAILABLE;
}

template <typename A, typename R>
int RemoteUnwindCursor<A,R>::setFloatReg(int regNum, double val)
{
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo ();
    if (procinfo == NULL) {
       ABORT("setRemoteReg called with a local unwind, use setReg instead.");
    }

    RemoteRegisterMap *regmap = procinfo->getRegisterMap ();
    int unwind_regno;
    if (regmap->caller_regno_to_unwind_regno (regNum, unwind_regno) == false)
        return UNW_EBADREG;
    regNum = unwind_regno;

    // Only allow the registers to be set if the unwind cursor is pointing to the
    // first frame.  We need to track where registers were retrieved from in memory
    // in every other frame.  Until then, we prohibit register setting in all but
    // the first frame.
    if (fIsFirstFrame) {
        return this->setFloatReg(unwind_regno, val);
    }
    return UNW_EREGUNAVAILABLE;
}


template <typename A, typename R>
const char* RemoteUnwindCursor<A,R>::getRegisterName(int r) 
{ 
    int t;
    if (!this->caller_regno_to_unwind_regno(r, t))
        return NULL;
    r = t;
	return this->UnwindCursor<A,R>::getRegisterName(r);
}

template <typename A, typename R>
int RemoteUnwindCursor<A,R>::step()
{
    pint_t pc = this->UnwindCursor<A,R>::getReg(UNW_REG_IP);
    pint_t sp = this->UnwindCursor<A,R>::getReg(UNW_REG_SP);
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo();
    bool frame_is_sigtramp = false;
    bool frame_is_inferior_function_call_dummy = false;

    if (procinfo == NULL) {
       ABORT("stepRemote called with local unwind, use step() instead.");
       return UNW_EUNSPEC;
    }
    struct timeval *step_remote = procinfo->timestamp_start();
    procinfo->logVerbose ("stepRemote stepping out of frame with pc value 0x%llx", pc);

    // We'll be off of the first frame once we finish this step.
    fIsFirstFrame = false;

    if (UnwindCursor<A,R>::fAddressSpace.accessors() 
        && UnwindCursor<A,R>::fAddressSpace.accessors()->proc_is_sigtramp != NULL
        && UnwindCursor<A,R>::fAddressSpace.accessors()->proc_is_sigtramp (procinfo->wrap(), pc, fArg)) {
        frame_is_sigtramp = true;
    } 
    if (UnwindCursor<A,R>::fAddressSpace.accessors() 
        && UnwindCursor<A,R>::fAddressSpace.accessors()->proc_is_inferior_function_call != NULL
        && UnwindCursor<A,R>::fAddressSpace.accessors()->proc_is_inferior_function_call (procinfo->wrap(), pc, sp, fArg)) {
        frame_is_inferior_function_call_dummy = true;
    }

    // If the function we're unwinding can't be a leaf function, 
    // use the eh_frame or compact unwind info if possible.
    // The caller should pass couldBeLeafFunc == 0 on the first step of a new context 
    // but we can't trust them in that.

    if ((fIsLeafFrame == false && frame_is_inferior_function_call_dummy == false)
        || frame_is_sigtramp) {
        R saved_registers(UnwindCursor<A,R>::fRegisters);
        this->setInfoBasedOnIPRegister(true);
        // bottom of stack is defined as when no more unwind info
        if ( !UnwindCursor<A,R>::fUnwindInfoMissing ) {
            int result;
            const char *method;
            if ( this->mustUseDwarf() ) {
                result = this->stepWithDwarfFDE();
                method = "dwarf";
            }
            else {
                result = this->stepWithCompactEncoding();
                method = "compact unwind";
            }
            if ( result == UNW_STEP_SUCCESS ) {
                procinfo->logInfo ("Stepped via %s", method);
                procinfo->timestamp_stop (step_remote, "stepRemote");
                if (frame_is_sigtramp)
                    fIsLeafFrame = true;
                return result;
            }
        }
        UnwindCursor<A,R>::fRegisters = saved_registers;
    }

    if (frame_is_sigtramp || frame_is_inferior_function_call_dummy)
        fIsLeafFrame = true;  // this will be true once we complete this stepRemote()
    else
        fIsLeafFrame = false;

    if (frame_is_inferior_function_call_dummy) {
        if (stepOutOfDebuggerDummyFrame (UnwindCursor<A,R>::fAddressSpace, UnwindCursor<A,R>::fRegisters, procinfo, pc, sp, fArg) == UNW_STEP_SUCCESS) {
            procinfo->logInfo ("Stepped via stepOutOfDebuggerDummyFrame");
            procinfo->timestamp_stop (step_remote, "stepRemote");
            return UNW_STEP_SUCCESS;
        }
    }

    // If we haven't already seen this function we'll need to get the function bounds via 
    // eh frame info (if available) - it's the most accurate function bounds in a 
    // stripped binary.  After that we'll ask the driver program (via the get_proc_bounds accessor).

    if (procinfo->haveProfile (pc) == false) {

        uint64_t text_start, text_end, eh_frame_start, eh_frame_len, compact_unwind_start, mh;
        uint64_t start_addr, end_addr;
        if (pc == 0) {
            int ret = stepByArchitectureDefault (UnwindCursor<A,R>::fAddressSpace, UnwindCursor<A,R>::fRegisters, pc);
            procinfo->logInfo ("Stepped via stepByArchitectureDefault");
            procinfo->timestamp_stop (step_remote, "stepRemote");
            return ret;
        }

        // If the address is not contained in any image's address range either we've walked off
        // the stack into random memory or we're backtracing through jit'ed code on the heap.
        // Let's assume the latter and follow the architecture's default stack walking scheme.

        if (!procinfo->getImageAddresses (pc, mh, text_start, text_end, eh_frame_start, eh_frame_len, compact_unwind_start, fArg)) {
            int ret = stepByArchitectureDefault (UnwindCursor<A,R>::fAddressSpace, UnwindCursor<A,R>::fRegisters, pc);
            procinfo->logInfo ("Stepped via stepByArchitectureDefault");
            procinfo->timestamp_stop (step_remote, "stepRemote");
            return ret;
        }
        if (procinfo->haveFuncBounds (mh) == false) {
            struct timeval *get_func_bounds = procinfo->timestamp_start();
            std::vector<FuncBounds> func_bounds;
            // CFI entries are usually around 38 bytes but under-estimate a bit 
            // because we're not distinguishing between CIEs and FDEs.
            if (eh_frame_len > 0)
                func_bounds.reserve (eh_frame_len / 16);
            if (procinfo->getCachingPolicy() != UNW_CACHE_NONE) {    
                // cache the entire eh frame section - we'll need to read the whole
                // thing anyway so we might as well save it.
                uint8_t *eh_buf = (uint8_t *)malloc (eh_frame_len);
                if (UnwindCursor<A,R>::fAddressSpace.getBytes (eh_frame_start, eh_frame_len, eh_buf) == 0)
                {
                    free (eh_buf);
                    return UNW_EUNSPEC;
                }
                RemoteMemoryBlob *ehmem = new RemoteMemoryBlob(eh_buf, free, eh_frame_start, eh_frame_len, mh, NULL);
                if (procinfo->addMemBlob (ehmem) == false)
                    delete ehmem;
            }
            
            if (CFI_Parser<A>::functionFuncBoundsViaFDE(UnwindCursor<A,R>::fAddressSpace, eh_frame_start, eh_frame_len, func_bounds)) {
                procinfo->addFuncBounds(mh, func_bounds);
                procinfo->logVerbose ("Added %d function bounds", (int) func_bounds.size());
                procinfo->timestamp_stop (get_func_bounds, "getting function bounds from EH frame FDEs");
            }
        }
        if (procinfo->findStartAddr (pc, start_addr, end_addr)) {
        // If end_addr is 0, we might be looking at the final function in this binary image
            if (start_addr != 0 && end_addr == 0)
                end_addr = text_end;
            procinfo->logVerbose ("Got function bounds from func bounds vector, 0x%llx-0x%llx", start_addr, end_addr);
        } else {
            if (UnwindCursor<A,R>::fAddressSpace.accessors()->get_proc_bounds (procinfo->wrap(), pc, &start_addr, &end_addr, fArg) != UNW_ESUCCESS) {
                int ret = stepByArchitectureDefault (UnwindCursor<A,R>::fAddressSpace, UnwindCursor<A,R>::fRegisters, pc);
                procinfo->logInfo ("Stepped via stepByArchitectureDefault");
                procinfo->timestamp_stop (step_remote, "stepRemote");
                return ret;
            }
            else {
                procinfo->logVerbose ("Got function bounds from get_proc_bounds callback, 0x%llx-0x%llx", start_addr, end_addr);
            }
        }
        if (start_addr != 0) {
            procinfo->addProfile (UnwindCursor<A,R>::fAddressSpace.accessors(), UnwindCursor<A,R>::fAddressSpace.wrap(), start_addr, end_addr, fArg);
        }
    }

    RemoteUnwindProfile *profile = procinfo->findProfile (pc);
    if (profile == NULL)
      return UNW_ENOINFO;

    int retval = stepWithAssembly (UnwindCursor<A,R>::fAddressSpace, pc, profile, UnwindCursor<A,R>::fRegisters);
    if (retval >= 0) {
        procinfo->logInfo ("Stepped via stepWithAssembly");
        procinfo->timestamp_stop (step_remote, "stepRemote");
        return retval;
    }

    retval = stepByArchitectureDefault (UnwindCursor<A,R>::fAddressSpace, UnwindCursor<A,R>::fRegisters, pc);
    procinfo->logInfo ("Stepped via stepByArchitectureDefault");
    procinfo->timestamp_stop (step_remote, "stepRemote");
    return retval;
}

template <typename A, typename R>
void RemoteUnwindCursor<A,R>::setRemoteContext(void *arg)
{
    // fill in the register state for the currently executing frame.
    getRemoteContext (UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo(), UnwindCursor<A,R>::fRegisters, arg);

    // Flag that this unwind cursor is pointing at the zeroth frame.  We don't
    // want to use compact unwind info / eh frame info to unwind out of this
    // frame. 

    fIsLeafFrame = true;
    fIsFirstFrame = true;
}

// This needs to be done in many of the functions and in libuwind.cxx in one or two
// places so I'm defining a convenience method.
template <typename A, typename R>
bool RemoteUnwindCursor<A,R>::caller_regno_to_unwind_regno (int caller_regno, int& unwind_regno)
{
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo ();
    if (procinfo == NULL) {
        unwind_regno = caller_regno;
        return true;
    }
    if (procinfo->getRegisterMap()->caller_regno_to_unwind_regno (caller_regno, unwind_regno))
        return true;
    return false;
}

template <typename A, typename R>
int RemoteUnwindCursor<A,R>::endOfPrologueInsns (unw_word_t start, unw_word_t end, unw_word_t *endofprologue)
{
    RemoteProcInfo *procinfo = UnwindCursor<A,R>::fAddressSpace.getRemoteProcInfo();
    *endofprologue = start;
    if (procinfo == NULL) {
       ABORT("findEndOfPrologueSetup called with local unwind.");
       return UNW_EUNSPEC;
    }
    if (procinfo->haveProfile (start) == false) {
        uint64_t text_start, text_end, eh_frame_start, eh_frame_len, compact_unwind_start, mh;
        if (!procinfo->getImageAddresses (start, mh, text_start, text_end, eh_frame_start, eh_frame_len, compact_unwind_start, fArg))
            return UNW_EUNSPEC;
        if (end == 0) {
            if (procinfo->haveFuncBounds (mh) == false) {
                std::vector<FuncBounds> func_bounds;
                // CFI entries are usually around 38 bytes but under-estimate a bit 
                // because we're not distinguishing between CIEs and FDEs.
                if (eh_frame_len > 0)
                    func_bounds.reserve (eh_frame_len / 16);
                if (procinfo->getCachingPolicy() != UNW_CACHE_NONE) {
                    // cache the entire eh frame section - we'll need to read the whole
                    // thing anyway so we might as well save it.
                    uint8_t *eh_buf = (uint8_t *)malloc (eh_frame_len);
                    if (UnwindCursor<A,R>::fAddressSpace.getBytes (eh_frame_start, eh_frame_len, eh_buf) == 0)
                    {
                        free (eh_buf);
                        return UNW_EUNSPEC;
                    }
                    RemoteMemoryBlob *ehmem = new RemoteMemoryBlob(eh_buf, free, eh_frame_start, eh_frame_len, mh, NULL);
                    if (procinfo->addMemBlob (ehmem) == false)
                        delete ehmem;
                }
                if (CFI_Parser<A>::functionFuncBoundsViaFDE(UnwindCursor<A,R>::fAddressSpace, eh_frame_start, eh_frame_len, func_bounds)) {
                    procinfo->addFuncBounds(mh, func_bounds);
                }
            }
            uint64_t bounded_start, bounded_end;
            if (procinfo->findStartAddr (start, bounded_start, bounded_end)) {
                end = bounded_end;
            } else {
                if (UnwindCursor<A,R>::fAddressSpace.accessors()->get_proc_bounds (procinfo->wrap(), start, &bounded_start, &bounded_end, fArg) != UNW_ESUCCESS) 
                    if (bounded_end != 0)
                        end = bounded_end;
            }
        }
        if (procinfo->addProfile (UnwindCursor<A,R>::fAddressSpace.accessors(), UnwindCursor<A,R>::fAddressSpace.wrap(), start, end, fArg) == false)
            return UNW_EUNSPEC;
    }
    RemoteUnwindProfile *profile = procinfo->findProfile (start);
    if (profile == NULL)
      return UNW_ENOINFO;
    *endofprologue = profile->fFirstInsnPastPrologue;
    return UNW_ESUCCESS;
}

#endif // SUPPORT_REMOTE_UNWINDING


}; // namespace lldb_private 


#endif // __UNWINDCURSOR_HPP__
