/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- DwarfParser.hpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 
//
// processor specific parsing of dwarf unwind instructions
//

#ifndef __DWARF_PARSER_HPP__
#define __DWARF_PARSER_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "libunwind.h"
#include "dwarf2.h"

#include "AddressSpace.hpp"
#include "RemoteUnwindProfile.h"

namespace lldb_private {


///
/// CFI_Parser does basic parsing of a CFI (Call Frame Information) records.
/// See Dwarf Spec for details: 
///    http://www.linux-foundation.org/spec/booksets/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
///
template <typename A>
class CFI_Parser
{
public:
	typedef typename A::pint_t		pint_t;	

	///
	/// Information encoded in a CIE (Common Information Entry)
	///  
	struct CIE_Info {
		pint_t		cieStart;
		pint_t		cieLength;
		pint_t		cieInstructions;
		uint8_t		pointerEncoding;
		uint8_t		lsdaEncoding;
		uint8_t		personalityEncoding;
		uint8_t		personalityOffsetInCIE;
		pint_t		personality;
		int			codeAlignFactor;
		int			dataAlignFactor;
		bool		isSignalFrame;
		bool		fdesHaveAugmentationData;
	};
	
	///
	/// Information about an FDE (Frame Description Entry)
	///  
	struct FDE_Info {
		pint_t		fdeStart;
		pint_t		fdeLength;
		pint_t		fdeInstructions;
		pint_t		pcStart;
		pint_t		pcEnd;
		pint_t		lsda;
	};

	///
	/// Used by linker when parsing __eh_frame section
	///  
	struct FDE_Reference {
		pint_t		address;
		uint32_t	offsetInFDE;
		uint8_t		encodingOfAddress;
	};
	struct FDE_Atom_Info {
		pint_t			fdeAddress;
		FDE_Reference	function;
		FDE_Reference	cie;
		FDE_Reference	lsda;
	};
	struct CIE_Atom_Info {
		pint_t			cieAddress;
		FDE_Reference	personality;
	};
	
	
	///
	/// Information about a frame layout and registers saved determined 
	/// by "running" the dwarf FDE "instructions"
	///  
	enum { kMaxRegisterNumber = 120 };
	enum RegisterSavedWhere { kRegisterUnused, kRegisterInCFA, kRegisterOffsetFromCFA,
							kRegisterInRegister, kRegisterAtExpression, kRegisterIsExpression } ;
	struct RegisterLocation {
		RegisterSavedWhere	location;
		int64_t				value;
	};
	struct PrologInfo {
		uint32_t			cfaRegister;		
		int32_t				cfaRegisterOffset;	// CFA = (cfaRegister)+cfaRegisterOffset
		int64_t				cfaExpression;		// CFA = expression
		bool				registersInOtherRegisters;
		bool				registerSavedMoreThanOnce;
		bool				cfaOffsetWasNegative;
		uint32_t			spExtraArgSize;
		uint32_t			codeOffsetAtStackDecrement;
		
		RegisterLocation	savedRegisters[kMaxRegisterNumber];	// from where to restore registers
	};

	struct PrologInfoStackEntry {
								PrologInfoStackEntry(PrologInfoStackEntry* n, const PrologInfo& i)
									: next(n), info(i) {}
		PrologInfoStackEntry*	next;
		PrologInfo				info;
	};

	static bool findFDE(A& addressSpace, pint_t pc, pint_t ehSectionStart, uint32_t sectionLength, pint_t fdeHint, FDE_Info* fdeInfo, CIE_Info* cieInfo);

#if defined (SUPPORT_REMOTE_UNWINDING)
    static bool functionFuncBoundsViaFDE(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength, std::vector<FuncBounds> &funcbounds);
#endif

	static const char* decodeFDE(A& addressSpace, pint_t fdeStart, FDE_Info* fdeInfo, CIE_Info* cieInfo);
	static bool parseFDEInstructions(A& addressSpace, const FDE_Info& fdeInfo, const CIE_Info& cieInfo, pint_t upToPC, PrologInfo* results);
	static const char* getCFIs(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength, 
								std::vector<FDE_Atom_Info>& fdes, std::vector<CIE_Atom_Info>& cies);
	static uint32_t getCFICount(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength);

	static const char* parseCIE(A& addressSpace, pint_t cie, CIE_Info* cieInfo);

private:
	static bool parseInstructions(A& addressSpace, pint_t instructions, pint_t instructionsEnd, const CIE_Info& cieInfo, 
								pint_t pcoffset, PrologInfoStackEntry*& rememberStack, PrologInfo* results);

};


///
/// Parse a FDE into a CIE_Info and an FDE_Info 
///  
template <typename A>
const char* CFI_Parser<A>::decodeFDE(A& addressSpace, pint_t fdeStart, FDE_Info* fdeInfo, CIE_Info* cieInfo)
{
	pint_t p = fdeStart;
	uint64_t cfiLength = addressSpace.get32(p);
	p += 4;
	if ( cfiLength == 0xffffffff ) {
		// 0xffffffff means length is really next 8 bytes
		cfiLength = addressSpace.get64(p);
		p += 8;
	}
	if ( cfiLength == 0 ) 
		return "FDE has zero length";	// end marker
	uint32_t ciePointer = addressSpace.get32(p);
	if ( ciePointer == 0 ) 
		return "FDE is really a CIE";	// this is a CIE not an FDE
	pint_t nextCFI = p + cfiLength;
	pint_t cieStart = p-ciePointer;
	const char* err = parseCIE(addressSpace, cieStart, cieInfo);
	if (err != NULL)
		return err;
	p += 4;
	// parse pc begin and range
	pint_t pcStart = addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding);
	pint_t pcRange = addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding & 0x0F);
	// parse rest of info
	fdeInfo->lsda = 0;
	// check for augmentation length
	if ( cieInfo->fdesHaveAugmentationData ) {
		uintptr_t augLen = addressSpace.getULEB128(p, nextCFI);
		pint_t endOfAug = p + augLen;
		if ( cieInfo->lsdaEncoding != 0 ) {
			// peek at value (without indirection).  Zero means no lsda
			pint_t lsdaStart = p;
			if ( addressSpace.getEncodedP(p, nextCFI, cieInfo->lsdaEncoding & 0x0F) != 0 ) {
				// reset pointer and re-parse lsda address
				p = lsdaStart;
				fdeInfo->lsda = addressSpace.getEncodedP(p, nextCFI, cieInfo->lsdaEncoding);
			}
		}
		p = endOfAug;
	}
	fdeInfo->fdeStart = fdeStart;
	fdeInfo->fdeLength = nextCFI - fdeStart;
	fdeInfo->fdeInstructions = p;
	fdeInfo->pcStart = pcStart;
	fdeInfo->pcEnd = pcStart+pcRange;
	return NULL; // success
}


///
/// Scan an eh_frame section to find an FDE for a pc
///  
template <typename A>
bool CFI_Parser<A>::findFDE(A& addressSpace, pint_t pc, pint_t ehSectionStart, uint32_t sectionLength, pint_t fdeHint, FDE_Info* fdeInfo, CIE_Info* cieInfo)
{
	//fprintf(stderr, "findFDE(0x%llX)\n", (long long)pc);
	pint_t p = (fdeHint != 0) ? fdeHint : ehSectionStart;
	const pint_t ehSectionEnd = p + sectionLength;
	while ( p < ehSectionEnd ) {
		pint_t currentCFI = p;
		//fprintf(stderr, "findFDE() CFI at 0x%llX\n", (long long)p);
		uint64_t cfiLength = addressSpace.get32(p);
		p += 4;
		if ( cfiLength == 0xffffffff ) {
			// 0xffffffff means length is really next 8 bytes
			cfiLength = addressSpace.get64(p);
			p += 8;
		}
		if ( cfiLength == 0 ) 
			return false;	// end marker
		uint32_t id = addressSpace.get32(p);
		if ( id == 0 ) {
			// skip over CIEs
			p += cfiLength;
		}
		else {
			// process FDE to see if it covers pc
			pint_t nextCFI = p + cfiLength;
			uint32_t ciePointer = addressSpace.get32(p);
			pint_t cieStart = p-ciePointer;
			// validate pointer to CIE is within section
			if ( (ehSectionStart <= cieStart) && (cieStart < ehSectionEnd) ) {
				if ( parseCIE(addressSpace, cieStart, cieInfo) == NULL ) {
					p += 4;
					// parse pc begin and range
					pint_t pcStart = addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding);
					pint_t pcRange = addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding & 0x0F);
					//fprintf(stderr, "FDE with pcRange [0x%08llX, 0x%08llX)\n",(uint64_t)pcStart, (uint64_t)(pcStart+pcRange));
					// test if pc is within the function this FDE covers
					if ( (pcStart < pc) && (pc <= pcStart+pcRange) ) {
						// parse rest of info
						fdeInfo->lsda = 0;
						// check for augmentation length
						if ( cieInfo->fdesHaveAugmentationData ) {
							uintptr_t augLen = addressSpace.getULEB128(p, nextCFI);
							pint_t endOfAug = p + augLen;
							if ( cieInfo->lsdaEncoding != 0 ) {
								// peek at value (without indirection).  Zero means no lsda
								pint_t lsdaStart = p;
								if ( addressSpace.getEncodedP(p, nextCFI, cieInfo->lsdaEncoding & 0x0F) != 0 ) {
									// reset pointer and re-parse lsda address
									p = lsdaStart;
									fdeInfo->lsda = addressSpace.getEncodedP(p, nextCFI, cieInfo->lsdaEncoding);
								}
							}
							p = endOfAug;
						}
						fdeInfo->fdeStart = currentCFI;
						fdeInfo->fdeLength = nextCFI - currentCFI;
						fdeInfo->fdeInstructions = p;
						fdeInfo->pcStart = pcStart;
						fdeInfo->pcEnd = pcStart+pcRange;
						//fprintf(stderr, "findFDE(pc=0x%llX) found with pcRange [0x%08llX, 0x%08llX)\n",(uint64_t)pc, (uint64_t)pcStart, (uint64_t)(pcStart+pcRange));
						return true;
					}
					else {
						//fprintf(stderr, "findFDE(pc=0x%llX) not found with pcRange [0x%08llX, 0x%08llX)\n",(uint64_t)pc, (uint64_t)pcStart, (uint64_t)(pcStart+pcRange));
						// pc is not in begin/range, skip this FDE
					}
				}
				else {
					// malformed CIE, now augmentation describing pc range encoding
					//fprintf(stderr, "malformed CIE\n");
				}
			}
			else {
				// malformed FDE.  CIE is bad
				//fprintf(stderr, "malformed FDE, cieStart=0x%llX, ehSectionStart=0x%llX, ehSectionEnd=0x%llX\n",
				//	(uint64_t)cieStart, (uint64_t)ehSectionStart, (uint64_t)ehSectionEnd);
			}
			p = nextCFI;
		}
	}
	//fprintf(stderr, "findFDE(pc=0x%llX) not found\n",(uint64_t)pc);
	return false;
}

#if defined (SUPPORT_REMOTE_UNWINDING)
/// Scan an eh_frame section to find all the function start addresses 
/// This is only made for working with libunwind-remote.  It copies
/// the eh_frame section into local memory and steps through it quickly
/// to find the start addresses of the CFIs.
///  
template <typename A>
bool CFI_Parser<A>::functionFuncBoundsViaFDE(A& addressSpace, pint_t ehSectionStart, 
					     uint32_t sectionLength, std::vector<FuncBounds> &funcbounds)
{
	//fprintf(stderr, "functionFuncBoundsViaFDE(0x%llX)\n", (long long)pc);
	pint_t p = ehSectionStart;
	const pint_t ehSectionEnd = p + sectionLength;
    pint_t lastCieSeen = (pint_t) -1;
    CIE_Info cieInfo;
	while ( p < ehSectionEnd ) {
		//fprintf(stderr, "functionFuncBoundsViaFDE() CFI at 0x%llX\n", (long long)p);
		uint64_t cfiLength = addressSpace.get32(p);
		p += 4;
		if ( cfiLength == 0xffffffff ) {
			// 0xffffffff means length is really next 8 bytes
			cfiLength = addressSpace.get64(p);
			p += 8;
		}
		if ( cfiLength == 0 ) 
			return false;	// end marker
		uint32_t id = addressSpace.get32(p);
		if ( id == 0 ) {
			// skip over CIEs
			p += cfiLength;
		}
		else {
			// process FDE to see if it covers pc
			pint_t nextCFI = p + cfiLength;
			uint32_t ciePointer = addressSpace.get32(p);
			pint_t cieStart = p-ciePointer;
			// validate pointer to CIE is within section
			if ( (ehSectionStart <= cieStart) && (cieStart < ehSectionEnd) ) {
                const char *errmsg;
                // don't re-parse the cie if this fde is pointing to one we already parsed
                if (cieStart == lastCieSeen) {
                    errmsg = NULL;
                }
                else {
                    errmsg = parseCIE(addressSpace, cieStart, &cieInfo);
                    if (errmsg == NULL)
                        lastCieSeen = cieStart;
                }
				if ( errmsg == NULL ) {
					p += 4;
					// parse pc begin and range
					pint_t pcStart = addressSpace.getEncodedP(p, nextCFI, cieInfo.pointerEncoding);
					pint_t pcRange = addressSpace.getEncodedP(p, nextCFI, cieInfo.pointerEncoding & 0x0F);
					//fprintf(stderr, "FDE with pcRange [0x%08llX, 0x%08llX)\n",(uint64_t)pcStart, (uint64_t)(pcStart+pcRange));
                    funcbounds.push_back(FuncBounds(pcStart, pcStart + pcRange));
				}
				else {
					// malformed CIE, now augmentation describing pc range encoding
					//fprintf(stderr, "malformed CIE\n");
                    return false;
				}
			}
			else {
				// malformed FDE.  CIE is bad
				//fprintf(stderr, "malformed FDE, cieStart=0x%llX, ehSectionStart=0x%llX, ehSectionEnd=0x%llX\n",
				//	(uint64_t)cieStart, (uint64_t)ehSectionStart, (uint64_t)ehSectionEnd);
                return false;
			}
			p = nextCFI;
		}
	}
	return true;
}
#endif // SUPPORT_REMOTE_UNWINDING



///
/// Extract info from a CIE
///  
template <typename A>
const char* CFI_Parser<A>::parseCIE(A& addressSpace, pint_t cie, CIE_Info* cieInfo)
{
	//fprintf(stderr, "parseCIE(0x%llX)\n", (long long)cie);
	cieInfo->pointerEncoding = 0;
	cieInfo->lsdaEncoding = 0;
	cieInfo->personalityEncoding = 0;
	cieInfo->personalityOffsetInCIE = 0;
	cieInfo->personality = 0;
	cieInfo->codeAlignFactor = 0;
	cieInfo->dataAlignFactor = 0;
	cieInfo->isSignalFrame = false;
	cieInfo->fdesHaveAugmentationData = false;
	cieInfo->cieStart = cie;
	pint_t p = cie;
	uint64_t cieLength = addressSpace.get32(p);
	p += 4;
	pint_t cieContentEnd = p + cieLength;
	if ( cieLength == 0xffffffff ) {
		// 0xffffffff means length is really next 8 bytes
		cieLength = addressSpace.get64(p);
		p += 8;
		cieContentEnd = p + cieLength;
	}
	if ( cieLength == 0 ) 
		return false;	
	// CIE ID is always 0
	if ( addressSpace.get32(p) != 0 ) 
		return "CIE ID is not zero";
	p += 4;
	// Version is always 1 or 3
	uint8_t version = addressSpace.get8(p);
	if ( (version != 1) && (version != 3) )
		return "CIE version is not 1 or 3";
	++p;
	// save start of augmentation string and find end
	pint_t strStart = p;
	while ( addressSpace.get8(p) != 0 )
		++p;
	++p;
	// parse code aligment factor
	cieInfo->codeAlignFactor = addressSpace.getULEB128(p, cieContentEnd);
	// parse data alignment factor
	cieInfo->dataAlignFactor = addressSpace.getSLEB128(p, cieContentEnd);
	// parse return address register
	addressSpace.getULEB128(p, cieContentEnd);
	// parse augmentation data based on augmentation string
	const char* result = NULL;
	if ( addressSpace.get8(strStart) == 'z' ) {
		// parse augmentation data length 
		addressSpace.getULEB128(p, cieContentEnd);
		for (pint_t s=strStart; addressSpace.get8(s) != '\0'; ++s) {
			switch ( addressSpace.get8(s) ) {
				case 'z':
					cieInfo->fdesHaveAugmentationData = true;
					break;
				case 'P':
					cieInfo->personalityEncoding = addressSpace.get8(p);
					++p;
					cieInfo->personalityOffsetInCIE = p-cie;
					cieInfo->personality = addressSpace.getEncodedP(p, cieContentEnd, cieInfo->personalityEncoding);
					break;
				case 'L':
					cieInfo->lsdaEncoding = addressSpace.get8(p);
					++p;
					break;
				case 'R':
					cieInfo->pointerEncoding = addressSpace.get8(p);
					++p;
					break;
				case 'S':
					cieInfo->isSignalFrame = true;
					break;
				default:
					// ignore unknown letters
					break;
			}
		}
	}
	cieInfo->cieLength = cieContentEnd - cieInfo->cieStart;
	cieInfo->cieInstructions = p;
	return result;
}


template <typename A>
uint32_t CFI_Parser<A>::getCFICount(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength)
{
	uint32_t count = 0;
	const pint_t ehSectionEnd = ehSectionStart + sectionLength;
	for (pint_t p=ehSectionStart; p < ehSectionEnd; ) {
		uint64_t cfiLength = addressSpace.get32(p);
		p += 4;
		if ( cfiLength == 0xffffffff ) {
			// 0xffffffff means length is really next 8 bytes
			cfiLength = addressSpace.get64(p);
			p += 8;
		}
		if ( cfiLength == 0 ) 
			return count;	// end marker
		++count;
		p += cfiLength;
	}
	return count;
}



template <typename A>
const char* CFI_Parser<A>::getCFIs(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength, 
								  std::vector<FDE_Atom_Info>& fdes, std::vector<CIE_Atom_Info>& cies)
{
	const pint_t ehSectionEnd = ehSectionStart + sectionLength;
	for (pint_t p=ehSectionStart; p < ehSectionEnd; ) {
		pint_t currentCFI = p;
		uint64_t cfiLength = addressSpace.get32(p);
		p += 4;
		if ( cfiLength == 0xffffffff ) {
			// 0xffffffff means length is really next 8 bytes
			cfiLength = addressSpace.get64(p);
			p += 8;
		}
		if ( cfiLength == 0 ) 
			return NULL;	// end marker
		uint32_t id = addressSpace.get32(p);
		if ( id == 0 ) {
			// is CIE
			CIE_Info cieInfo;
			const char* err = parseCIE(addressSpace, currentCFI, &cieInfo);
			if ( err != NULL ) 
				return err;
			CIE_Atom_Info entry;
			entry.cieAddress = currentCFI;
			entry.personality.address = cieInfo.personality;
			entry.personality.offsetInFDE = cieInfo.personalityOffsetInCIE;
			entry.personality.encodingOfAddress = cieInfo.personalityEncoding;
			cies.push_back(entry);
			p += cfiLength;
		}
		else {
			// is FDE
			FDE_Atom_Info entry;
			entry.fdeAddress = currentCFI;
			entry.function.address = 0;
			entry.cie.address = 0;
			entry.lsda.address = 0;
			pint_t nextCFI = p + cfiLength;
			uint32_t ciePointer = addressSpace.get32(p);
			pint_t cieStart = p-ciePointer;
			// validate pointer to CIE is within section
			if ( (cieStart < ehSectionStart) || (cieStart > ehSectionEnd) )
				return "FDE points to CIE outside __eh_frame section";
			CIE_Info cieInfo;
			const char* err = parseCIE(addressSpace, cieStart, &cieInfo);
			if ( err != NULL ) 
				return err;
			entry.cie.address = cieStart;
			entry.cie.offsetInFDE = p-currentCFI;
			entry.cie.encodingOfAddress = DW_EH_PE_sdata4 | DW_EH_PE_pcrel;
			p += 4;
			// parse pc begin and range
			pint_t offsetOfFunctionAddress = p-currentCFI;
			pint_t pcStart = addressSpace.getEncodedP(p, nextCFI, cieInfo.pointerEncoding);
			pint_t pcRange = addressSpace.getEncodedP(p, nextCFI, cieInfo.pointerEncoding & 0x0F);
			//fprintf(stderr, "FDE with pcRange [0x%08llX, 0x%08llX)\n",(uint64_t)pcStart, (uint64_t)(pcStart+pcRange));
			// test if pc is within the function this FDE covers
			entry.function.address = pcStart;
			entry.function.offsetInFDE = offsetOfFunctionAddress;
			entry.function.encodingOfAddress = cieInfo.pointerEncoding;
			// skip over augmentation length
			if ( cieInfo.fdesHaveAugmentationData ) {
				uintptr_t augLen = addressSpace.getULEB128(p, nextCFI);
				pint_t endOfAug = p + augLen;
				if ( (cieInfo.lsdaEncoding != 0) && (addressSpace.getP(p) != 0) ) {
					pint_t offsetOfLSDAAddress = p-currentCFI;
					entry.lsda.address = addressSpace.getEncodedP(p, nextCFI, cieInfo.lsdaEncoding);
					entry.lsda.offsetInFDE = offsetOfLSDAAddress;
					entry.lsda.encodingOfAddress = cieInfo.lsdaEncoding;
				}
				p = endOfAug;
			}
			fdes.push_back(entry);
			p = nextCFI;
		}
	}
	return NULL; // success
}

	

///
/// "run" the dwarf instructions and create the abstact PrologInfo for an FDE
///  
template <typename A>
bool CFI_Parser<A>::parseFDEInstructions(A& addressSpace, const FDE_Info& fdeInfo, const CIE_Info& cieInfo, pint_t upToPC, PrologInfo* results)
{
	// clear results
	bzero(results, sizeof(PrologInfo));
	PrologInfoStackEntry* rememberStack = NULL;

	// parse CIE then FDE instructions
	return parseInstructions(addressSpace, cieInfo.cieInstructions, cieInfo.cieStart+cieInfo.cieLength, 
						cieInfo, (pint_t)(-1), rememberStack, results)
	    && parseInstructions(addressSpace, fdeInfo.fdeInstructions, fdeInfo.fdeStart+fdeInfo.fdeLength, 
							cieInfo, upToPC-fdeInfo.pcStart, rememberStack, results);
}


///
/// "run" the dwarf instructions
///  
template <typename A>
bool CFI_Parser<A>::parseInstructions(A& addressSpace, pint_t instructions, pint_t instructionsEnd, const CIE_Info& cieInfo,
									pint_t pcoffset, PrologInfoStackEntry*& rememberStack, PrologInfo* results)
{
	const bool logDwarf = false;
	pint_t p = instructions;
	uint32_t codeOffset = 0;
	PrologInfo initialState = *results;
	
	// see Dwarf Spec, section 6.4.2 for details on unwind opcodes
	while ( (p < instructionsEnd) && (codeOffset < pcoffset) ) {
		uint64_t reg;
		uint64_t reg2;
		int64_t offset;
		uint64_t length;
		uint8_t opcode = addressSpace.get8(p);
		uint8_t operand;
		PrologInfoStackEntry* entry;
		++p;
		switch (opcode) {
			case DW_CFA_nop:
				if ( logDwarf ) fprintf(stderr, "DW_CFA_nop\n");
				break;
			case DW_CFA_set_loc:
				codeOffset = addressSpace.getEncodedP(p, instructionsEnd, cieInfo.pointerEncoding);
				if ( logDwarf ) fprintf(stderr, "DW_CFA_set_loc\n");
				break;
			case DW_CFA_advance_loc1:
				codeOffset += (addressSpace.get8(p) * cieInfo.codeAlignFactor);
				p += 1;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_advance_loc1: new offset=%u\n", codeOffset);
				break;
			case DW_CFA_advance_loc2:
				codeOffset += (addressSpace.get16(p) * cieInfo.codeAlignFactor);
				p += 2;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_advance_loc2: new offset=%u\n", codeOffset);
				break;
			case DW_CFA_advance_loc4:
				codeOffset += (addressSpace.get32(p) * cieInfo.codeAlignFactor);
				p += 4;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_advance_loc4: new offset=%u\n", codeOffset);
				break;
			case DW_CFA_offset_extended:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				offset = addressSpace.getULEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_offset_extended dwarf unwind, reg too big\n");
					return false;
				}
				if ( results->savedRegisters[reg].location != kRegisterUnused ) 
					results->registerSavedMoreThanOnce = true;
				results->savedRegisters[reg].location = kRegisterInCFA;
				results->savedRegisters[reg].value = offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_offset_extended(reg=%lld, offset=%lld)\n", reg, offset);
				break;
			case DW_CFA_restore_extended:
				reg = addressSpace.getULEB128(p, instructionsEnd);;
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_restore_extended dwarf unwind, reg too big\n");
					return false;
				}
				results->savedRegisters[reg] = initialState.savedRegisters[reg];
				if ( logDwarf ) fprintf(stderr, "DW_CFA_restore_extended(reg=%lld)\n", reg);
				break;
			case DW_CFA_undefined:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_undefined dwarf unwind, reg too big\n");
					return false;
				}
				results->savedRegisters[reg].location = kRegisterUnused;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_undefined(reg=%lld)\n", reg);
				break;
			case DW_CFA_same_value:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_same_value dwarf unwind, reg too big\n");
					return false;
				}
				if ( logDwarf ) fprintf(stderr, "DW_CFA_same_value(reg=%lld)\n", reg);
				break;
			case DW_CFA_register:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				reg2 = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_register dwarf unwind, reg too big\n");
					return false;
				}
				if ( reg2 > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_register dwarf unwind, reg2 too big\n");
					return false;
				}
				results->savedRegisters[reg].location = kRegisterInRegister;
				results->savedRegisters[reg].value = reg2;
				results->registersInOtherRegisters = true;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_register(reg=%lld, reg2=%lld)\n", reg, reg2);
				break;
			case DW_CFA_remember_state:
				// avoid operator new, because that would be an upward dependency
				entry = (PrologInfoStackEntry*)malloc(sizeof(PrologInfoStackEntry));
				if ( entry != NULL ) {
					entry->next = rememberStack;
					entry->info = *results;
					rememberStack = entry;
				}
				else {
					return false;
				}
				if ( logDwarf ) fprintf(stderr, "DW_CFA_remember_state\n");
				break;
			case DW_CFA_restore_state:
				if ( rememberStack != NULL ) {
					PrologInfoStackEntry* top = rememberStack;
					*results = top->info;
					rememberStack = top->next;
					free((char*)top);
				}
				else {
					return false;
				}
				if ( logDwarf ) fprintf(stderr, "DW_CFA_restore_state\n");
				break;
			case DW_CFA_def_cfa:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				offset = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_def_cfa dwarf unwind, reg too big\n");
					return false;
				}
				results->cfaRegister = reg;
				results->cfaRegisterOffset = offset;
				if ( offset > 0x80000000 ) 
					results->cfaOffsetWasNegative = true;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_def_cfa(reg=%lld, offset=%lld)\n", reg, offset);
				break;
			case DW_CFA_def_cfa_register:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_def_cfa_register dwarf unwind, reg too big\n");
					return false;
				}
				results->cfaRegister = reg;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_def_cfa_register(%lld)\n", reg);
				break;
			case DW_CFA_def_cfa_offset:
				results->cfaRegisterOffset = addressSpace.getULEB128(p, instructionsEnd);
				results->codeOffsetAtStackDecrement = codeOffset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_def_cfa_offset(%d)\n", results->cfaRegisterOffset);
				break;
			case DW_CFA_def_cfa_expression:
				results->cfaRegister = 0;
				results->cfaExpression = p;
				length = addressSpace.getULEB128(p, instructionsEnd);
				p += length;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_def_cfa_expression(expression=0x%llX, length=%llu)\n", 
													results->cfaExpression, length);
				break;
			case DW_CFA_expression:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_expression dwarf unwind, reg too big\n");
					return false;
				}
				results->savedRegisters[reg].location = kRegisterAtExpression;
				results->savedRegisters[reg].value = p;
				length = addressSpace.getULEB128(p, instructionsEnd);
				p += length;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_expression(reg=%lld, expression=0x%llX, length=%llu)\n", 
													reg, results->savedRegisters[reg].value, length);
				break;
			case DW_CFA_offset_extended_sf:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_offset_extended_sf dwarf unwind, reg too big\n");
					return false;
				}
				offset = addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				if ( results->savedRegisters[reg].location != kRegisterUnused ) 
					results->registerSavedMoreThanOnce = true;
				results->savedRegisters[reg].location = kRegisterInCFA;
				results->savedRegisters[reg].value = offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_offset_extended_sf(reg=%lld, offset=%lld)\n", reg, offset);
				break;
			case DW_CFA_def_cfa_sf:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				offset = addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_def_cfa_sf dwarf unwind, reg too big\n");
					return false;
				}
				results->cfaRegister = reg;
				results->cfaRegisterOffset = offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_def_cfa_sf(reg=%lld, offset=%lld)\n", reg, offset);
				break;
			case DW_CFA_def_cfa_offset_sf:
				results->cfaRegisterOffset = addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				results->codeOffsetAtStackDecrement = codeOffset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_def_cfa_offset_sf(%d)\n", results->cfaRegisterOffset);
				break;
			case DW_CFA_val_offset:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				offset = addressSpace.getULEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				results->savedRegisters[reg].location = kRegisterOffsetFromCFA;
				results->savedRegisters[reg].value = offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_val_offset(reg=%lld, offset=%lld\n", reg, offset);
				break;
			case DW_CFA_val_offset_sf:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_val_offset_sf dwarf unwind, reg too big\n");
					return false;
				}
				offset = addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				results->savedRegisters[reg].location = kRegisterOffsetFromCFA;
				results->savedRegisters[reg].value = offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_val_offset_sf(reg=%lld, offset=%lld\n", reg, offset);
				break;
			case DW_CFA_val_expression:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_val_expression dwarf unwind, reg too big\n");
					return false;
				}
				results->savedRegisters[reg].location = kRegisterIsExpression;
				results->savedRegisters[reg].value = p;
				length = addressSpace.getULEB128(p, instructionsEnd);
				p += length;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_val_expression(reg=%lld, expression=0x%llX, length=%lld)\n", 
													reg, results->savedRegisters[reg].value, length);
				break;
			case DW_CFA_GNU_args_size:
				offset = addressSpace.getULEB128(p, instructionsEnd);
				results->spExtraArgSize = offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_GNU_args_size(%lld)\n", offset);
				break;
			case DW_CFA_GNU_negative_offset_extended:
				reg = addressSpace.getULEB128(p, instructionsEnd);
				if ( reg > kMaxRegisterNumber ) {
					fprintf(stderr, "malformed DW_CFA_GNU_negative_offset_extended dwarf unwind, reg too big\n");
					return false;
				}
				offset = addressSpace.getULEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
				if ( results->savedRegisters[reg].location != kRegisterUnused ) 
					results->registerSavedMoreThanOnce = true;
				results->savedRegisters[reg].location = kRegisterInCFA;
				results->savedRegisters[reg].value = -offset;
				if ( logDwarf ) fprintf(stderr, "DW_CFA_GNU_negative_offset_extended(%lld)\n", offset);
				break;
			default:
				operand = opcode & 0x3F;
				switch ( opcode & 0xC0 ) {
					case DW_CFA_offset:
						reg = operand;
						offset = addressSpace.getULEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
						if ( results->savedRegisters[reg].location != kRegisterUnused ) 
							results->registerSavedMoreThanOnce = true;
						results->savedRegisters[reg].location = kRegisterInCFA;
						results->savedRegisters[reg].value = offset;
						if ( logDwarf ) fprintf(stderr, "DW_CFA_offset(reg=%d, offset=%lld)\n", operand, offset);
						break;
					case DW_CFA_advance_loc:
						codeOffset += operand * cieInfo.codeAlignFactor;
						if ( logDwarf ) fprintf(stderr, "DW_CFA_advance_loc: new offset=%u\n", codeOffset);
						break;
					case DW_CFA_restore:
						// <rdar://problem/7503075> Python crashes when handling an exception thrown by an obj-c object
						// libffi uses DW_CFA_restore in the middle of some custom dward, so it is not a good epilog flag
						//return true; // gcc-4.5 starts the epilog with this
						reg = operand;
						results->savedRegisters[reg] = initialState.savedRegisters[reg];
						if ( logDwarf ) fprintf(stderr, "DW_CFA_restore(reg=%lld)\n", reg);
						break;
					default: 
						if ( logDwarf ) fprintf(stderr, "unknown CFA opcode 0x%02X\n", opcode);
						return false;
				}
		}
	}

	return true;
}


} // namespace lldb_private 


#endif // __DWARF_PARSER_HPP__




