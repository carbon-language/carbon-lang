/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- DwarfInstructions.hpp -----------------------------------*- C++ -*-===//
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

#ifndef __DWARF_INSTRUCTIONS_HPP__
#define __DWARF_INSTRUCTIONS_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include <libunwind.h>
#include <mach-o/compact_unwind_encoding.h>

#include "dwarf2.h"
#include "AddressSpace.hpp"
#include "Registers.hpp"
#include "DwarfParser.hpp"
#include "InternalMacros.h"
//#include "CompactUnwinder.hpp"

#define EXTRACT_BITS(value, mask) \
	( (value >> __builtin_ctz(mask)) & (((1 << __builtin_popcount(mask)))-1) )

#define CFI_INVALID_ADDRESS ((pint_t)(-1))

namespace lldb_private {

///
/// Used by linker when parsing __eh_frame section
///  
template <typename A>
struct CFI_Reference {
	typedef typename A::pint_t		pint_t;	
	uint8_t		encodingOfTargetAddress;
	uint32_t	offsetInCFI;
	pint_t		targetAddress;
};
template <typename A>
struct CFI_Atom_Info {
	typedef typename A::pint_t		pint_t;	
	pint_t			address;
	uint32_t		size;
	bool			isCIE;
	union {
		struct {
			CFI_Reference<A>	function;
			CFI_Reference<A>	cie;
			CFI_Reference<A>	lsda;
			uint32_t		compactUnwindInfo;
		}			fdeInfo;
		struct {
			CFI_Reference<A>	personality;
		}			cieInfo;
	} u;
};

typedef void (*WarnFunc)(void* ref, uint64_t funcAddr, const char* msg);  

///
/// DwarfInstructions maps abtract dwarf unwind instructions to a particular architecture
///  
template <typename A, typename R>
class DwarfInstructions
{
public:
	typedef typename A::pint_t		pint_t;	
	typedef typename A::sint_t		sint_t;	

	static const char* parseCFIs(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength, 
						CFI_Atom_Info<A>* infos, uint32_t infosCount, void* ref, WarnFunc warn);


	static compact_unwind_encoding_t createCompactEncodingFromFDE(A& addressSpace, pint_t fdeStart, 
																pint_t* lsda, pint_t* personality,
																char warningBuffer[1024]);

	static int stepWithDwarf(A& addressSpace, pint_t pc, pint_t fdeStart, R& registers);
										
private:

	enum {
		DW_X86_64_RET_ADDR = 16
	};

	enum {
		DW_X86_RET_ADDR = 8
	};

	static pint_t evaluateExpression(pint_t expression, A& addressSpace, const R& registers, pint_t initialStackValue);
	static pint_t getSavedRegister(A& addressSpace, const R& registers, pint_t cfa, 
										const typename CFI_Parser<A>::RegisterLocation& savedReg);
	static double getSavedFloatRegister(A& addressSpace, const R& registers, pint_t cfa, 
										const typename CFI_Parser<A>::RegisterLocation& savedReg);
	static v128 getSavedVectorRegister(A& addressSpace, const R& registers, pint_t cfa, 
										const typename CFI_Parser<A>::RegisterLocation& savedReg);
										
	// x86 specific variants
	static int    lastRestoreReg(const Registers_x86&);
	static bool   isReturnAddressRegister(int regNum, const Registers_x86&);
	static pint_t getCFA(A& addressSpace, const typename CFI_Parser<A>::PrologInfo& prolog, const Registers_x86&);

	static uint32_t getEBPEncodedRegister(uint32_t reg, int32_t regOffsetFromBaseOffset, bool& failure);
	static compact_unwind_encoding_t encodeToUseDwarf(const Registers_x86&);
	static compact_unwind_encoding_t createCompactEncodingFromProlog(A& addressSpace, pint_t funcAddr,
												const Registers_x86&, const typename CFI_Parser<A>::PrologInfo& prolog,
												char warningBuffer[1024]);

	// x86_64 specific variants
	static int    lastRestoreReg(const Registers_x86_64&);
	static bool   isReturnAddressRegister(int regNum, const Registers_x86_64&);
	static pint_t getCFA(A& addressSpace, const typename CFI_Parser<A>::PrologInfo& prolog, const Registers_x86_64&);

	static uint32_t getRBPEncodedRegister(uint32_t reg, int32_t regOffsetFromBaseOffset, bool& failure);
	static compact_unwind_encoding_t encodeToUseDwarf(const Registers_x86_64&);
	static compact_unwind_encoding_t createCompactEncodingFromProlog(A& addressSpace, pint_t funcAddr,
												const Registers_x86_64&, const typename CFI_Parser<A>::PrologInfo& prolog,
												char warningBuffer[1024]);
};


											

template <typename A, typename R>
const char* DwarfInstructions<A,R>::parseCFIs(A& addressSpace, pint_t ehSectionStart, uint32_t sectionLength, 
												CFI_Atom_Info<A>* infos, uint32_t infosCount, void* ref, WarnFunc warn)
{
	typename CFI_Parser<A>::CIE_Info cieInfo;
	CFI_Atom_Info<A>* entry = infos;
	CFI_Atom_Info<A>* end = &infos[infosCount];
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
		if ( entry >= end )
			return "too little space allocated for parseCFIs";
		pint_t nextCFI = p + cfiLength;
		uint32_t id = addressSpace.get32(p);
		if ( id == 0 ) {
			// is CIE
			const char* err = CFI_Parser<A>::parseCIE(addressSpace, currentCFI, &cieInfo);
			if ( err != NULL ) 
				return err;
			entry->address = currentCFI;
			entry->size = nextCFI - currentCFI;
			entry->isCIE = true;
			entry->u.cieInfo.personality.targetAddress = cieInfo.personality;
			entry->u.cieInfo.personality.offsetInCFI = cieInfo.personalityOffsetInCIE;
			entry->u.cieInfo.personality.encodingOfTargetAddress = cieInfo.personalityEncoding;
			++entry;
		}
		else {
			// is FDE
			entry->address = currentCFI;
			entry->size = nextCFI - currentCFI;
			entry->isCIE = false;
			entry->u.fdeInfo.function.targetAddress = CFI_INVALID_ADDRESS;
			entry->u.fdeInfo.cie.targetAddress = CFI_INVALID_ADDRESS;
			entry->u.fdeInfo.lsda.targetAddress = CFI_INVALID_ADDRESS;
			uint32_t ciePointer = addressSpace.get32(p);
			pint_t cieStart = p-ciePointer;
			// validate pointer to CIE is within section
			if ( (cieStart < ehSectionStart) || (cieStart > ehSectionEnd) )
				return "FDE points to CIE outside __eh_frame section";
			// optimize usual case where cie is same for all FDEs
			if ( cieStart != cieInfo.cieStart ) {
				const char* err = CFI_Parser<A>::parseCIE(addressSpace, cieStart, &cieInfo);
				if ( err != NULL ) 
					return err;
			}
			entry->u.fdeInfo.cie.targetAddress = cieStart;
			entry->u.fdeInfo.cie.offsetInCFI = p-currentCFI;
			entry->u.fdeInfo.cie.encodingOfTargetAddress = DW_EH_PE_sdata4 | DW_EH_PE_pcrel;
			p += 4;
			// parse pc begin and range
			pint_t offsetOfFunctionAddress = p-currentCFI;
			pint_t pcStart = addressSpace.getEncodedP(p, nextCFI, cieInfo.pointerEncoding);
			pint_t pcRange = addressSpace.getEncodedP(p, nextCFI, cieInfo.pointerEncoding & 0x0F);
			//fprintf(stderr, "FDE with pcRange [0x%08llX, 0x%08llX)\n",(uint64_t)pcStart, (uint64_t)(pcStart+pcRange));
			// test if pc is within the function this FDE covers
			entry->u.fdeInfo.function.targetAddress = pcStart;
			entry->u.fdeInfo.function.offsetInCFI = offsetOfFunctionAddress;
			entry->u.fdeInfo.function.encodingOfTargetAddress = cieInfo.pointerEncoding;
			// check for augmentation length
			if ( cieInfo.fdesHaveAugmentationData ) {
				uintptr_t augLen = addressSpace.getULEB128(p, nextCFI);
				pint_t endOfAug = p + augLen;
				if ( cieInfo.lsdaEncoding != 0 ) {
					// peek at value (without indirection).  Zero means no lsda
					pint_t lsdaStart = p;
					if ( addressSpace.getEncodedP(p, nextCFI, cieInfo.lsdaEncoding & 0x0F) != 0 ) {
						// reset pointer and re-parse lsda address
						p = lsdaStart;
						pint_t offsetOfLSDAAddress = p-currentCFI;
						entry->u.fdeInfo.lsda.targetAddress = addressSpace.getEncodedP(p, nextCFI, cieInfo.lsdaEncoding);
						entry->u.fdeInfo.lsda.offsetInCFI = offsetOfLSDAAddress;
						entry->u.fdeInfo.lsda.encodingOfTargetAddress = cieInfo.lsdaEncoding;
					}
				}
				p = endOfAug;
			}
			// compute compact unwind encoding
			typename CFI_Parser<A>::FDE_Info fdeInfo;
			fdeInfo.fdeStart = currentCFI;
			fdeInfo.fdeLength = nextCFI - currentCFI;
			fdeInfo.fdeInstructions = p;
			fdeInfo.pcStart = pcStart;
			fdeInfo.pcEnd = pcStart +  pcRange;
			fdeInfo.lsda = entry->u.fdeInfo.lsda.targetAddress;
			typename CFI_Parser<A>::PrologInfo prolog;
			R dummy; // for proper selection of architecture specific functions
			if ( CFI_Parser<A>::parseFDEInstructions(addressSpace, fdeInfo, cieInfo, CFI_INVALID_ADDRESS, &prolog) ) {
				char warningBuffer[1024];
				entry->u.fdeInfo.compactUnwindInfo = createCompactEncodingFromProlog(addressSpace, fdeInfo.pcStart, dummy, prolog, warningBuffer);
				if ( fdeInfo.lsda != CFI_INVALID_ADDRESS ) 
					entry->u.fdeInfo.compactUnwindInfo |= UNWIND_HAS_LSDA;
				if ( warningBuffer[0] != '\0' )
					warn(ref, fdeInfo.pcStart, warningBuffer);
			}
			else {
				warn(ref, CFI_INVALID_ADDRESS, "dwarf unwind instructions could not be parsed");
				entry->u.fdeInfo.compactUnwindInfo = encodeToUseDwarf(dummy);
			}
			++entry;
		}
		p = nextCFI;
	}
	if ( entry != end )
		return "wrong entry count for parseCFIs";
	return NULL; // success
}




template <typename A, typename R>
compact_unwind_encoding_t DwarfInstructions<A,R>::createCompactEncodingFromFDE(A& addressSpace, pint_t fdeStart, 
																		pint_t* lsda, pint_t* personality,
																		char warningBuffer[1024])
{
	typename CFI_Parser<A>::FDE_Info fdeInfo;
	typename CFI_Parser<A>::CIE_Info cieInfo;
	R dummy; // for proper selection of architecture specific functions
	if ( CFI_Parser<A>::decodeFDE(addressSpace, fdeStart, &fdeInfo, &cieInfo) == NULL ) {
		typename CFI_Parser<A>::PrologInfo prolog;
		if ( CFI_Parser<A>::parseFDEInstructions(addressSpace, fdeInfo, cieInfo, CFI_INVALID_ADDRESS, &prolog) ) {
			*lsda = fdeInfo.lsda;
			*personality = cieInfo.personality;
			compact_unwind_encoding_t encoding;
			encoding = createCompactEncodingFromProlog(addressSpace, fdeInfo.pcStart, dummy, prolog, warningBuffer);
			if ( fdeInfo.lsda != 0 ) 
				encoding |= UNWIND_HAS_LSDA;
			return encoding;
		}
		else {
			strcpy(warningBuffer, "dwarf unwind instructions could not be parsed");
			return encodeToUseDwarf(dummy);
		}
	}
	else {
		strcpy(warningBuffer, "dwarf FDE could not be parsed");
		return encodeToUseDwarf(dummy);
	}
}


template <typename A, typename R>
typename A::pint_t DwarfInstructions<A,R>::getSavedRegister(A& addressSpace, const R& registers, pint_t cfa,
													const typename CFI_Parser<A>::RegisterLocation& savedReg)
{
	switch ( savedReg.location ) {
		case CFI_Parser<A>::kRegisterInCFA:
			return addressSpace.getP(cfa + savedReg.value);

		case CFI_Parser<A>::kRegisterAtExpression:
			return addressSpace.getP(evaluateExpression(savedReg.value, addressSpace, registers, cfa));

		case CFI_Parser<A>::kRegisterIsExpression:
			return evaluateExpression(savedReg.value, addressSpace, registers, cfa);

		case CFI_Parser<A>::kRegisterInRegister:
			return registers.getRegister(savedReg.value);

		case CFI_Parser<A>::kRegisterUnused:
		case CFI_Parser<A>::kRegisterOffsetFromCFA:
			// FIX ME
			break;
	}
	ABORT("unsupported restore location for register");
}

template <typename A, typename R>
double DwarfInstructions<A,R>::getSavedFloatRegister(A& addressSpace, const R& registers, pint_t cfa,
													const typename CFI_Parser<A>::RegisterLocation& savedReg)
{
	switch ( savedReg.location ) {
		case CFI_Parser<A>::kRegisterInCFA:
			return addressSpace.getDouble(cfa + savedReg.value);

		case CFI_Parser<A>::kRegisterAtExpression:
			return addressSpace.getDouble(evaluateExpression(savedReg.value, addressSpace, registers, cfa));

		case CFI_Parser<A>::kRegisterIsExpression:
		case CFI_Parser<A>::kRegisterUnused:
		case CFI_Parser<A>::kRegisterOffsetFromCFA:
		case CFI_Parser<A>::kRegisterInRegister:
			// FIX ME
			break;
	}
	ABORT("unsupported restore location for float register");
}

template <typename A, typename R>
v128 DwarfInstructions<A,R>::getSavedVectorRegister(A& addressSpace, const R& registers, pint_t cfa,
													const typename CFI_Parser<A>::RegisterLocation& savedReg)
{
	switch ( savedReg.location ) {
		case CFI_Parser<A>::kRegisterInCFA:
			return addressSpace.getVector(cfa + savedReg.value);

		case CFI_Parser<A>::kRegisterAtExpression:
			return addressSpace.getVector(evaluateExpression(savedReg.value, addressSpace, registers, cfa));

		case CFI_Parser<A>::kRegisterIsExpression:
		case CFI_Parser<A>::kRegisterUnused:
		case CFI_Parser<A>::kRegisterOffsetFromCFA:
		case CFI_Parser<A>::kRegisterInRegister:
			// FIX ME
			break;
	}
	ABORT("unsupported restore location for vector register");
}


template <typename A, typename R>
int DwarfInstructions<A,R>::stepWithDwarf(A& addressSpace, pint_t pc, pint_t fdeStart, R& registers)
{
	//fprintf(stderr, "stepWithDwarf(pc=0x%0llX, fdeStart=0x%0llX)\n", (uint64_t)pc, (uint64_t)fdeStart);
	typename CFI_Parser<A>::FDE_Info fdeInfo;
	typename CFI_Parser<A>::CIE_Info cieInfo;
	if ( CFI_Parser<A>::decodeFDE(addressSpace, fdeStart, &fdeInfo, &cieInfo) == NULL ) {
		typename CFI_Parser<A>::PrologInfo prolog;
		if ( CFI_Parser<A>::parseFDEInstructions(addressSpace, fdeInfo, cieInfo, pc, &prolog) ) {
			R newRegisters = registers;
			
			// get pointer to cfa (architecture specific)
			pint_t cfa = getCFA(addressSpace, prolog, registers);

			// restore registers that dwarf says were saved
			pint_t returnAddress = 0;
			for (int i=0; i <= lastRestoreReg(newRegisters); ++i) {
				if ( prolog.savedRegisters[i].location != CFI_Parser<A>::kRegisterUnused ) {
					if ( registers.validFloatRegister(i) )
						newRegisters.setFloatRegister(i, getSavedFloatRegister(addressSpace, registers, cfa, prolog.savedRegisters[i]));
					else if ( registers.validVectorRegister(i) )
						newRegisters.setVectorRegister(i, getSavedVectorRegister(addressSpace, registers, cfa, prolog.savedRegisters[i]));
					else if ( isReturnAddressRegister(i, registers) )
						returnAddress = getSavedRegister(addressSpace, registers, cfa, prolog.savedRegisters[i]);
					else if ( registers.validRegister(i) )
						newRegisters.setRegister(i, getSavedRegister(addressSpace, registers, cfa, prolog.savedRegisters[i]));
					else
						return UNW_EBADREG;
				}
			}
			
			// by definition the CFA is the stack pointer at the call site, so restoring SP means setting it to CFA
			newRegisters.setSP(cfa);

			// return address is address after call site instruction, so setting IP to that does a return
			newRegisters.setIP(returnAddress);
			
			// do the actual step by replacing the register set with the new ones
			registers = newRegisters;

			return UNW_STEP_SUCCESS;
		}
	}
	return UNW_EBADFRAME;
}



template <typename A, typename R>
typename A::pint_t DwarfInstructions<A,R>::evaluateExpression(pint_t expression, A& addressSpace, 
														const R& registers, pint_t initialStackValue)
{
	const bool log = false;
	pint_t p = expression;
	pint_t expressionEnd = expression+20; // just need something until length is read
	uint64_t length = addressSpace.getULEB128(p, expressionEnd);
	expressionEnd = p + length;
	if (log) fprintf(stderr, "evaluateExpression(): length=%llu\n", length);
	pint_t stack[100];
	pint_t* sp = stack;
	*(++sp) = initialStackValue;
	
	while ( p < expressionEnd ) {
		if (log) {
			for(pint_t* t = sp; t > stack; --t) {
				fprintf(stderr, "sp[] = 0x%llX\n", (uint64_t)(*t));
			}
		}
		uint8_t opcode = addressSpace.get8(p++);
		sint_t svalue;
		pint_t value;
		uint32_t reg;
		switch (opcode) {
			case DW_OP_addr:
				// push immediate address sized value
				value = addressSpace.getP(p);
				p += sizeof(pint_t);
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
		
			case DW_OP_deref:
				// pop stack, dereference, push result
				value = *sp--;
				*(++sp) = addressSpace.getP(value);
				if (log) fprintf(stderr, "dereference 0x%llX\n", (uint64_t)value);
				break;
		
			case DW_OP_const1u:
				// push immediate 1 byte value
				value = addressSpace.get8(p);
				p += 1;
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
				
			case DW_OP_const1s:
				// push immediate 1 byte signed value
				svalue = (int8_t)addressSpace.get8(p);
				p += 1;
				*(++sp) = svalue;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)svalue);
				break;
		
			case DW_OP_const2u:
				// push immediate 2 byte value
				value = addressSpace.get16(p);
				p += 2;
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
				
			case DW_OP_const2s:
				// push immediate 2 byte signed value
				svalue = (int16_t)addressSpace.get16(p);
				p += 2;
				*(++sp) = svalue;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)svalue);
				break;
		
			case DW_OP_const4u:
				// push immediate 4 byte value
				value = addressSpace.get32(p);
				p += 4;
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
				
			case DW_OP_const4s:
				// push immediate 4 byte signed value
				svalue = (int32_t)addressSpace.get32(p);
				p += 4;
				*(++sp) = svalue;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)svalue);
				break;
				
			case DW_OP_const8u:
				// push immediate 8 byte value
				value = addressSpace.get64(p);
				p += 8;
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
				
			case DW_OP_const8s:
				// push immediate 8 byte signed value
				value = (int32_t)addressSpace.get64(p);
				p += 8;
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
		
			case DW_OP_constu:
				// push immediate ULEB128 value
				value = addressSpace.getULEB128(p, expressionEnd);
				*(++sp) = value;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)value);
				break;
				
			case DW_OP_consts:
				// push immediate SLEB128 value
				svalue = addressSpace.getSLEB128(p, expressionEnd);
				*(++sp) = svalue;
				if (log) fprintf(stderr, "push 0x%llX\n", (uint64_t)svalue);
				break;
		
			case DW_OP_dup:
				// push top of stack
				value = *sp;
				*(++sp) = value;
				if (log) fprintf(stderr, "duplicate top of stack\n");
				break;
				
			case DW_OP_drop:
				// pop
				--sp; 
				if (log) fprintf(stderr, "pop top of stack\n");
				break;
				
			case DW_OP_over:
				// dup second
				value = sp[-1];
				*(++sp) = value;
				if (log) fprintf(stderr, "duplicate second in stack\n");
				break;

			case DW_OP_pick:
				// pick from
				reg = addressSpace.get8(p);
				p += 1;
				value = sp[-reg];
				*(++sp) = value;
				if (log) fprintf(stderr, "duplicate %d in stack\n", reg);
				break;

			case DW_OP_swap:
				// swap top two
				value = sp[0];
				sp[0] = sp[-1];
				sp[-1] = value;
				if (log) fprintf(stderr, "swap top of stack\n");
				break;

			case DW_OP_rot:
				// rotate top three
				value = sp[0];
				sp[0] = sp[-1];
				sp[-1] = sp[-2];
				sp[-2] = value;
				if (log) fprintf(stderr, "rotate top three of stack\n");
				break;

			case DW_OP_xderef:
				// pop stack, dereference, push result
				value = *sp--;
				*sp = *((uint64_t*)value);
				if (log) fprintf(stderr, "x-dereference 0x%llX\n", (uint64_t)value);
				break;
			
			case DW_OP_abs:
				svalue = *sp;
				if ( svalue < 0 )
					*sp = -svalue;
				if (log) fprintf(stderr, "abs\n");
				break;
		
			case DW_OP_and:
				value = *sp--;
				*sp &= value;
				if (log) fprintf(stderr, "and\n");
				break;
			
			case DW_OP_div:
				svalue = *sp--;
				*sp = *sp / svalue;
				if (log) fprintf(stderr, "div\n");
				break;
			
			case DW_OP_minus:
				svalue = *sp--;
				*sp = *sp - svalue;
				if (log) fprintf(stderr, "minus\n");
				break;

			case DW_OP_mod:
				svalue = *sp--;
				*sp = *sp % svalue;
				if (log) fprintf(stderr, "module\n");
				break;

			case DW_OP_mul:
				svalue = *sp--;
				*sp = *sp * svalue;
				if (log) fprintf(stderr, "mul\n");
				break;

			case DW_OP_neg:
				*sp =  0 - *sp;
				if (log) fprintf(stderr, "neg\n");
				break;

			case DW_OP_not:
				svalue = *sp;
				*sp =  ~svalue;
				if (log) fprintf(stderr, "not\n");
				break;

			case DW_OP_or:
				value = *sp--;
				*sp |= value;
				if (log) fprintf(stderr, "or\n");
				break;

			case DW_OP_plus:
				value = *sp--;
				*sp += value;
				if (log) fprintf(stderr, "plus\n");
				break;

			case DW_OP_plus_uconst:
				// pop stack, add uelb128 constant, push result
				*sp += addressSpace.getULEB128(p, expressionEnd);
				if (log) fprintf(stderr, "add constant\n");
				break;
		
			case DW_OP_shl:
				value = *sp--;
				*sp = *sp << value;
				if (log) fprintf(stderr, "shift left\n");
				break;
			
			case DW_OP_shr:
				value = *sp--;
				*sp = *sp >> value;
				if (log) fprintf(stderr, "shift left\n");
				break;
				
			case DW_OP_shra:
				value = *sp--;
				svalue = *sp;
				*sp = svalue >> value;
				if (log) fprintf(stderr, "shift left arithmetric\n");
				break;
			
			case DW_OP_xor:
				value = *sp--;
				*sp ^= value;
				if (log) fprintf(stderr, "xor\n");
				break;

			case DW_OP_skip:
				svalue = (int16_t)addressSpace.get16(p);
				p += 2;
				p += svalue;
				if (log) fprintf(stderr, "skip %lld\n", (uint64_t)svalue);
				break;
			
			case DW_OP_bra:
				svalue = (int16_t)addressSpace.get16(p);
				p += 2;
				if ( *sp-- )
					p += svalue;
				if (log) fprintf(stderr, "bra %lld\n", (uint64_t)svalue);
				break;
			
			case DW_OP_eq:
				value = *sp--;
				*sp = (*sp == value);
				if (log) fprintf(stderr, "eq\n");
				break;
			
			case DW_OP_ge:
				value = *sp--;
				*sp = (*sp >= value);
				if (log) fprintf(stderr, "ge\n");
				break;
				
			case DW_OP_gt:
				value = *sp--;
				*sp = (*sp > value);
				if (log) fprintf(stderr, "gt\n");
				break;
				
			case DW_OP_le:
				value = *sp--;
				*sp = (*sp <= value);
				if (log) fprintf(stderr, "le\n");
				break;
				
			case DW_OP_lt:
				value = *sp--;
				*sp = (*sp < value);
				if (log) fprintf(stderr, "lt\n");
				break;
				
			case DW_OP_ne:
				value = *sp--;
				*sp = (*sp != value);
				if (log) fprintf(stderr, "ne\n");
				break;
			
			case DW_OP_lit0:
			case DW_OP_lit1:
			case DW_OP_lit2:
			case DW_OP_lit3:
			case DW_OP_lit4:
			case DW_OP_lit5:
			case DW_OP_lit6:
			case DW_OP_lit7:
			case DW_OP_lit8:
			case DW_OP_lit9:
			case DW_OP_lit10:
			case DW_OP_lit11:
			case DW_OP_lit12:
			case DW_OP_lit13:
			case DW_OP_lit14:
			case DW_OP_lit15:
			case DW_OP_lit16:
			case DW_OP_lit17:
			case DW_OP_lit18:
			case DW_OP_lit19:
			case DW_OP_lit20:
			case DW_OP_lit21:
			case DW_OP_lit22:
			case DW_OP_lit23:
			case DW_OP_lit24:
			case DW_OP_lit25:
			case DW_OP_lit26:
			case DW_OP_lit27:
			case DW_OP_lit28:
			case DW_OP_lit29:
			case DW_OP_lit30:
			case DW_OP_lit31:
				value = opcode - DW_OP_lit0;
				*(++sp) = value;
				if (log) fprintf(stderr, "push literal 0x%llX\n", (uint64_t)value);
				break;
		
			case DW_OP_reg0:
			case DW_OP_reg1:
			case DW_OP_reg2:
			case DW_OP_reg3:
			case DW_OP_reg4:
			case DW_OP_reg5:
			case DW_OP_reg6:
			case DW_OP_reg7:
			case DW_OP_reg8:
			case DW_OP_reg9:
			case DW_OP_reg10:
			case DW_OP_reg11:
			case DW_OP_reg12:
			case DW_OP_reg13:
			case DW_OP_reg14:
			case DW_OP_reg15:
			case DW_OP_reg16:
			case DW_OP_reg17:
			case DW_OP_reg18:
			case DW_OP_reg19:
			case DW_OP_reg20:
			case DW_OP_reg21:
			case DW_OP_reg22:
			case DW_OP_reg23:
			case DW_OP_reg24:
			case DW_OP_reg25:
			case DW_OP_reg26:
			case DW_OP_reg27:
			case DW_OP_reg28:
			case DW_OP_reg29:
			case DW_OP_reg30:
			case DW_OP_reg31:
				reg = opcode - DW_OP_reg0;
				*(++sp) = registers.getRegister(reg);
				if (log) fprintf(stderr, "push reg %d\n", reg);
				break;
		
			case DW_OP_regx:
				reg = addressSpace.getULEB128(p, expressionEnd);
				*(++sp) = registers.getRegister(reg);
				if (log) fprintf(stderr, "push reg %d + 0x%llX\n", reg, (uint64_t)svalue);
				break;			

			case DW_OP_breg0:
			case DW_OP_breg1:
			case DW_OP_breg2:
			case DW_OP_breg3:
			case DW_OP_breg4:
			case DW_OP_breg5:
			case DW_OP_breg6:
			case DW_OP_breg7:
			case DW_OP_breg8:
			case DW_OP_breg9:
			case DW_OP_breg10:
			case DW_OP_breg11:
			case DW_OP_breg12:
			case DW_OP_breg13:
			case DW_OP_breg14:
			case DW_OP_breg15:
			case DW_OP_breg16:
			case DW_OP_breg17:
			case DW_OP_breg18:
			case DW_OP_breg19:
			case DW_OP_breg20:
			case DW_OP_breg21:
			case DW_OP_breg22:
			case DW_OP_breg23:
			case DW_OP_breg24:
			case DW_OP_breg25:
			case DW_OP_breg26:
			case DW_OP_breg27:
			case DW_OP_breg28:
			case DW_OP_breg29:
			case DW_OP_breg30:
			case DW_OP_breg31:
				reg = opcode - DW_OP_breg0;
				svalue = addressSpace.getSLEB128(p, expressionEnd);
				*(++sp) = registers.getRegister(reg) + svalue;
				if (log) fprintf(stderr, "push reg %d + 0x%llX\n", reg, (uint64_t)svalue);
				break;
			
			case DW_OP_bregx:
				reg = addressSpace.getULEB128(p, expressionEnd);
				svalue = addressSpace.getSLEB128(p, expressionEnd);
				*(++sp) = registers.getRegister(reg) + svalue;
				if (log) fprintf(stderr, "push reg %d + 0x%llX\n", reg, (uint64_t)svalue);
				break;
			
			case DW_OP_fbreg:
				ABORT("DW_OP_fbreg not implemented");
				break;
				
			case DW_OP_piece:
				ABORT("DW_OP_piece not implemented");
				break;
				
			case DW_OP_deref_size:
				// pop stack, dereference, push result
				value = *sp--;
				switch ( addressSpace.get8(p++) ) {
					case 1:
						value = addressSpace.get8(value);
						break;
					case 2:
						value = addressSpace.get16(value);
						break;
					case 4:
						value = addressSpace.get32(value);
						break;
					case 8:
						value = addressSpace.get64(value);
						break;
					default:
						ABORT("DW_OP_deref_size with bad size");
				}
				*(++sp) = value;
				if (log) fprintf(stderr, "sized dereference 0x%llX\n", (uint64_t)value);
				break;
			
			case DW_OP_xderef_size:
			case DW_OP_nop:
			case DW_OP_push_object_addres:
			case DW_OP_call2:
			case DW_OP_call4:
			case DW_OP_call_ref:
			default:
				ABORT("dwarf opcode not implemented");
		}
	
	}
	if (log) fprintf(stderr, "expression evaluates to 0x%llX\n", (uint64_t)*sp);
	return *sp;
}



//
//	x86_64 specific functions
//

template <typename A, typename R>
int DwarfInstructions<A,R>::lastRestoreReg(const Registers_x86_64&) 
{
	COMPILE_TIME_ASSERT( (int)CFI_Parser<A>::kMaxRegisterNumber > (int)DW_X86_64_RET_ADDR );
	return DW_X86_64_RET_ADDR; 
}

template <typename A, typename R>
bool DwarfInstructions<A,R>::isReturnAddressRegister(int regNum, const Registers_x86_64&) 
{
	return (regNum == DW_X86_64_RET_ADDR); 
}

template <typename A, typename R>
typename A::pint_t DwarfInstructions<A,R>::getCFA(A& addressSpace, const typename CFI_Parser<A>::PrologInfo& prolog, 
										const Registers_x86_64& registers)
{
	if ( prolog.cfaRegister != 0 )
		return registers.getRegister(prolog.cfaRegister) + prolog.cfaRegisterOffset;
	else if ( prolog.cfaExpression != 0 )
		return evaluateExpression(prolog.cfaExpression, addressSpace, registers, 0);
	else
		ABORT("getCFA(): unknown location for x86_64 cfa");
}



template <typename A, typename R>
compact_unwind_encoding_t DwarfInstructions<A,R>::encodeToUseDwarf(const Registers_x86_64&) 
{
	return UNWIND_X86_64_MODE_DWARF;
}

template <typename A, typename R>
compact_unwind_encoding_t DwarfInstructions<A,R>::encodeToUseDwarf(const Registers_x86&) 
{
	return UNWIND_X86_MODE_DWARF;
}



template <typename A, typename R>
uint32_t DwarfInstructions<A,R>::getRBPEncodedRegister(uint32_t reg, int32_t regOffsetFromBaseOffset, bool& failure)
{
	if ( (regOffsetFromBaseOffset < 0) || (regOffsetFromBaseOffset > 32) ) {
		failure = true;
		return 0;
	}
	unsigned int slotIndex = regOffsetFromBaseOffset/8;
	
	switch ( reg ) {
		case UNW_X86_64_RBX:
			return UNWIND_X86_64_REG_RBX << (slotIndex*3);
		case UNW_X86_64_R12:
			return UNWIND_X86_64_REG_R12 << (slotIndex*3);
		case UNW_X86_64_R13:
			return UNWIND_X86_64_REG_R13 << (slotIndex*3);
		case UNW_X86_64_R14:
			return UNWIND_X86_64_REG_R14 << (slotIndex*3);
		case UNW_X86_64_R15:
			return UNWIND_X86_64_REG_R15 << (slotIndex*3);
	}
	
	// invalid register
	failure = true;
	return 0;
}



template <typename A, typename R>
compact_unwind_encoding_t DwarfInstructions<A,R>::createCompactEncodingFromProlog(A& addressSpace, pint_t funcAddr,
												const Registers_x86_64& r, const typename CFI_Parser<A>::PrologInfo& prolog,
												char warningBuffer[1024])
{
	warningBuffer[0] = '\0';
	
	// don't create compact unwind info for unsupported dwarf kinds
	if ( prolog.registerSavedMoreThanOnce ) {
		strcpy(warningBuffer, "register saved more than once (might be shrink wrap)");
		return UNWIND_X86_64_MODE_DWARF;
	}
	if ( prolog.cfaOffsetWasNegative ) {
		strcpy(warningBuffer, "cfa had negative offset (dwarf might contain epilog)");
		return UNWIND_X86_64_MODE_DWARF;
	}
	if ( prolog.spExtraArgSize != 0 ) {
		strcpy(warningBuffer, "dwarf uses DW_CFA_GNU_args_size");
		return UNWIND_X86_64_MODE_DWARF;
	}
	
	// figure out which kind of frame this function uses
	bool standardRBPframe = ( 
		 (prolog.cfaRegister == UNW_X86_64_RBP) 
	  && (prolog.cfaRegisterOffset == 16)
	  && (prolog.savedRegisters[UNW_X86_64_RBP].location == CFI_Parser<A>::kRegisterInCFA)
	  && (prolog.savedRegisters[UNW_X86_64_RBP].value == -16) );
	bool standardRSPframe = (prolog.cfaRegister == UNW_X86_64_RSP);
	if ( !standardRBPframe && !standardRSPframe ) {
		// no compact encoding for this
		strcpy(warningBuffer, "does not use RBP or RSP based frame");
		return UNWIND_X86_64_MODE_DWARF;
	}
	
	// scan which registers are saved
	int saveRegisterCount = 0;
	bool rbxSaved = false;
	bool r12Saved = false;
	bool r13Saved = false;
	bool r14Saved = false;
	bool r15Saved = false;
	bool rbpSaved = false;
	for (int i=0; i < 64; ++i) {
		if ( prolog.savedRegisters[i].location != CFI_Parser<A>::kRegisterUnused ) {
			if ( prolog.savedRegisters[i].location != CFI_Parser<A>::kRegisterInCFA ) {
				sprintf(warningBuffer, "register %d saved somewhere other that in frame", i);
				return UNWIND_X86_64_MODE_DWARF;
			}
			switch (i) {
				case UNW_X86_64_RBX:
					rbxSaved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_64_R12:
					r12Saved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_64_R13:
					r13Saved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_64_R14:
					r14Saved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_64_R15:
					r15Saved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_64_RBP:
					rbpSaved = true;
					++saveRegisterCount;
					break;
				case DW_X86_64_RET_ADDR:
					break;
				default:
					sprintf(warningBuffer, "non-standard register %d being saved in prolog", i);
					return UNWIND_X86_64_MODE_DWARF;
			}
		}
	}
	const int64_t cfaOffsetRBX = prolog.savedRegisters[UNW_X86_64_RBX].value;
	const int64_t cfaOffsetR12 = prolog.savedRegisters[UNW_X86_64_R12].value;
	const int64_t cfaOffsetR13 = prolog.savedRegisters[UNW_X86_64_R13].value;
	const int64_t cfaOffsetR14 = prolog.savedRegisters[UNW_X86_64_R14].value;
	const int64_t cfaOffsetR15 = prolog.savedRegisters[UNW_X86_64_R15].value;
	const int64_t cfaOffsetRBP = prolog.savedRegisters[UNW_X86_64_RBP].value;
	
	// encode standard RBP frames
	compact_unwind_encoding_t  encoding = 0;
	if ( standardRBPframe ) {
		//		|              |
		//		+--------------+   <- CFA
		//		|   ret addr   |
		//		+--------------+
		//		|     rbp      |
		//		+--------------+   <- rbp
		//		~              ~
		//		+--------------+   
		//		|  saved reg3  |
		//		+--------------+   <- CFA - offset+16
		//		|  saved reg2  |
		//		+--------------+   <- CFA - offset+8
		//		|  saved reg1  |
		//		+--------------+   <- CFA - offset
		//		|              |
		//		+--------------+
		//		|              |
		//						   <- rsp
		//
		encoding = UNWIND_X86_64_MODE_RBP_FRAME;
		
		// find save location of farthest register from rbp
		int furthestCfaOffset = 0;
		if ( rbxSaved & (cfaOffsetRBX < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetRBX;
		if ( r12Saved & (cfaOffsetR12 < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetR12;
		if ( r13Saved & (cfaOffsetR13 < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetR13;
		if ( r14Saved & (cfaOffsetR14 < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetR14;
		if ( r15Saved & (cfaOffsetR15 < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetR15;
		
		if ( furthestCfaOffset == 0 ) {
			// no registers saved, nothing more to encode
			return encoding;
		}
		
		// add stack offset to encoding
		int rbpOffset = furthestCfaOffset + 16;
		int encodedOffset = rbpOffset/(-8);
		if ( encodedOffset > 255 ) {
			strcpy(warningBuffer, "offset of saved registers too far to encode");
			return UNWIND_X86_64_MODE_DWARF;
		}
		encoding |= (encodedOffset << __builtin_ctz(UNWIND_X86_64_RBP_FRAME_OFFSET));
		
		// add register saved from each stack location
		bool encodingFailure = false;
		if ( rbxSaved )
			encoding |= getRBPEncodedRegister(UNW_X86_64_RBX, cfaOffsetRBX - furthestCfaOffset, encodingFailure);
		if ( r12Saved )
			encoding |= getRBPEncodedRegister(UNW_X86_64_R12, cfaOffsetR12 - furthestCfaOffset, encodingFailure);
		if ( r13Saved )
			encoding |= getRBPEncodedRegister(UNW_X86_64_R13, cfaOffsetR13 - furthestCfaOffset, encodingFailure);
		if ( r14Saved )
			encoding |= getRBPEncodedRegister(UNW_X86_64_R14, cfaOffsetR14 - furthestCfaOffset, encodingFailure);
		if ( r15Saved )
			encoding |= getRBPEncodedRegister(UNW_X86_64_R15, cfaOffsetR15 - furthestCfaOffset, encodingFailure);
		
		if ( encodingFailure ){
			strcpy(warningBuffer, "saved registers not contiguous");
			return UNWIND_X86_64_MODE_DWARF;
		}

		return encoding;
	}
	else {
		//		|              |
		//		+--------------+   <- CFA
		//		|   ret addr   |
		//		+--------------+
		//		|  saved reg1  |
		//		+--------------+   <- CFA - 16
		//		|  saved reg2  |
		//		+--------------+   <- CFA - 24
		//		|  saved reg3  |
		//		+--------------+   <- CFA - 32
		//		|  saved reg4  |
		//		+--------------+   <- CFA - 40
		//		|  saved reg5  |
		//		+--------------+   <- CFA - 48
		//		|  saved reg6  |
		//		+--------------+   <- CFA - 56
		//		|              |
		//						   <- esp
		//

		// for RSP based frames we need to encode stack size in unwind info
		encoding = UNWIND_X86_64_MODE_STACK_IMMD;
		uint64_t stackValue = prolog.cfaRegisterOffset / 8;
		uint32_t stackAdjust = 0;
		bool immedStackSize = true;
		const uint32_t stackMaxImmedValue = EXTRACT_BITS(0xFFFFFFFF,UNWIND_X86_64_FRAMELESS_STACK_SIZE);
		if ( stackValue > stackMaxImmedValue ) {
			// stack size is too big to fit as an immediate value, so encode offset of subq instruction in function
			pint_t functionContentAdjustStackIns = funcAddr + prolog.codeOffsetAtStackDecrement - 4;		
			uint32_t stackDecrementInCode = addressSpace.get32(functionContentAdjustStackIns);
			stackAdjust = (prolog.cfaRegisterOffset - stackDecrementInCode)/8;
			stackValue = functionContentAdjustStackIns - funcAddr;
			immedStackSize = false;
			if ( stackAdjust > 7 ) {
				strcpy(warningBuffer, "stack subq instruction is too different from dwarf stack size");
				return UNWIND_X86_64_MODE_DWARF;
			}
			encoding = UNWIND_X86_64_MODE_STACK_IND;
		}	
		
		
		// validate that saved registers are all within 6 slots abutting return address
		int registers[6];
		for (int i=0; i < 6;++i)
			registers[i] = 0;
		if ( r15Saved ) {
			if ( cfaOffsetR15 < -56 ) {
				strcpy(warningBuffer, "r15 is saved too far from return address");
				return UNWIND_X86_64_MODE_DWARF;
			}
			registers[(cfaOffsetR15+56)/8] = UNWIND_X86_64_REG_R15;
		}
		if ( r14Saved ) {
			if ( cfaOffsetR14 < -56 ) {
				strcpy(warningBuffer, "r14 is saved too far from return address");
				return UNWIND_X86_64_MODE_DWARF;
			}
			registers[(cfaOffsetR14+56)/8] = UNWIND_X86_64_REG_R14;
		}
		if ( r13Saved ) {
			if ( cfaOffsetR13 < -56 ) {
				strcpy(warningBuffer, "r13 is saved too far from return address");
				return UNWIND_X86_64_MODE_DWARF;
			}
			registers[(cfaOffsetR13+56)/8] = UNWIND_X86_64_REG_R13;
		}
		if ( r12Saved ) {
			if ( cfaOffsetR12 < -56 ) {
				strcpy(warningBuffer, "r12 is saved too far from return address");
				return UNWIND_X86_64_MODE_DWARF;
			}
			registers[(cfaOffsetR12+56)/8] = UNWIND_X86_64_REG_R12;
		}
		if ( rbxSaved ) {
			if ( cfaOffsetRBX < -56 ) {
				strcpy(warningBuffer, "rbx is saved too far from return address");
				return UNWIND_X86_64_MODE_DWARF;
			}
			registers[(cfaOffsetRBX+56)/8] = UNWIND_X86_64_REG_RBX;
		}
		if ( rbpSaved ) {
			if ( cfaOffsetRBP < -56 ) {
				strcpy(warningBuffer, "rbp is saved too far from return address");
				return UNWIND_X86_64_MODE_DWARF;
			}
			registers[(cfaOffsetRBP+56)/8] = UNWIND_X86_64_REG_RBP;
		}
		
		// validate that saved registers are contiguous and abut return address on stack
		for (int i=0; i < saveRegisterCount; ++i) {
			if ( registers[5-i] == 0 ) {
				strcpy(warningBuffer, "registers not save contiguously in stack");
				return UNWIND_X86_64_MODE_DWARF;
			}
		}
				
		// encode register permutation
		// the 10-bits are encoded differently depending on the number of registers saved
		int renumregs[6];
		for (int i=6-saveRegisterCount; i < 6; ++i) {
			int countless = 0;
			for (int j=6-saveRegisterCount; j < i; ++j) {
				if ( registers[j] < registers[i] )
					++countless;
			}
			renumregs[i] = registers[i] - countless -1;
		}
		uint32_t permutationEncoding = 0;
		switch ( saveRegisterCount ) {
			case 6:
				permutationEncoding |= (120*renumregs[0] + 24*renumregs[1] + 6*renumregs[2] + 2*renumregs[3] + renumregs[4]);
				break;
			case 5:
				permutationEncoding |= (120*renumregs[1] + 24*renumregs[2] + 6*renumregs[3] + 2*renumregs[4] + renumregs[5]);
				break;
			case 4:
				permutationEncoding |= (60*renumregs[2] + 12*renumregs[3] + 3*renumregs[4] + renumregs[5]);
				break;
			case 3:
				permutationEncoding |= (20*renumregs[3] + 4*renumregs[4] + renumregs[5]);
				break;
			case 2:
				permutationEncoding |= (5*renumregs[4] + renumregs[5]);
				break;
			case 1:
				permutationEncoding |= (renumregs[5]);
				break;
		}
		
		encoding |= (stackValue << __builtin_ctz(UNWIND_X86_64_FRAMELESS_STACK_SIZE));
		encoding |= (stackAdjust << __builtin_ctz(UNWIND_X86_64_FRAMELESS_STACK_ADJUST));
		encoding |= (saveRegisterCount << __builtin_ctz(UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT));
		encoding |= (permutationEncoding << __builtin_ctz(UNWIND_X86_64_FRAMELESS_STACK_REG_PERMUTATION));
		return encoding;
	}
}




//
//	x86 specific functions
//
template <typename A, typename R>
int DwarfInstructions<A,R>::lastRestoreReg(const Registers_x86&) 
{
	COMPILE_TIME_ASSERT( (int)CFI_Parser<A>::kMaxRegisterNumber > (int)DW_X86_RET_ADDR );
	return DW_X86_RET_ADDR; 
}

template <typename A, typename R>
bool DwarfInstructions<A,R>::isReturnAddressRegister(int regNum, const Registers_x86&) 
{
	return (regNum == DW_X86_RET_ADDR); 
}

template <typename A, typename R>
typename A::pint_t DwarfInstructions<A,R>::getCFA(A& addressSpace, const typename CFI_Parser<A>::PrologInfo& prolog, 
										const Registers_x86& registers)
{
	if ( prolog.cfaRegister != 0 )
		return registers.getRegister(prolog.cfaRegister) + prolog.cfaRegisterOffset;
	else if ( prolog.cfaExpression != 0 )
		return evaluateExpression(prolog.cfaExpression, addressSpace, registers, 0);
	else
		ABORT("getCFA(): unknown location for x86 cfa");
}





template <typename A, typename R>
uint32_t DwarfInstructions<A,R>::getEBPEncodedRegister(uint32_t reg, int32_t regOffsetFromBaseOffset, bool& failure)
{
	if ( (regOffsetFromBaseOffset < 0) || (regOffsetFromBaseOffset > 16) ) {
		failure = true;
		return 0;
	}
	unsigned int slotIndex = regOffsetFromBaseOffset/4;
	
	switch ( reg ) {
		case UNW_X86_EBX:
			return UNWIND_X86_REG_EBX << (slotIndex*3);
		case UNW_X86_ECX:
			return UNWIND_X86_REG_ECX << (slotIndex*3);
		case UNW_X86_EDX:
			return UNWIND_X86_REG_EDX << (slotIndex*3);
		case UNW_X86_EDI:
			return UNWIND_X86_REG_EDI << (slotIndex*3);
		case UNW_X86_ESI:
			return UNWIND_X86_REG_ESI << (slotIndex*3);
	}
	
	// invalid register
	failure = true;
	return 0;
}

template <typename A, typename R>
compact_unwind_encoding_t DwarfInstructions<A,R>::createCompactEncodingFromProlog(A& addressSpace, pint_t funcAddr,
												const Registers_x86& r, const typename CFI_Parser<A>::PrologInfo& prolog,
												char warningBuffer[1024])
{
	warningBuffer[0] = '\0';
	
	// don't create compact unwind info for unsupported dwarf kinds
	if ( prolog.registerSavedMoreThanOnce ) {
		strcpy(warningBuffer, "register saved more than once (might be shrink wrap)");
		return UNWIND_X86_MODE_DWARF;
	}
	if ( prolog.spExtraArgSize != 0 ) {
		strcpy(warningBuffer, "dwarf uses DW_CFA_GNU_args_size");
		return UNWIND_X86_MODE_DWARF;
	}
	
	// figure out which kind of frame this function uses
	bool standardEBPframe = ( 
		 (prolog.cfaRegister == UNW_X86_EBP) 
	  && (prolog.cfaRegisterOffset == 8)
	  && (prolog.savedRegisters[UNW_X86_EBP].location == CFI_Parser<A>::kRegisterInCFA)
	  && (prolog.savedRegisters[UNW_X86_EBP].value == -8) );
	bool standardESPframe = (prolog.cfaRegister == UNW_X86_ESP);
	if ( !standardEBPframe && !standardESPframe ) {
		// no compact encoding for this
		strcpy(warningBuffer, "does not use EBP or ESP based frame");
		return UNWIND_X86_MODE_DWARF;
	}
	
	// scan which registers are saved
	int saveRegisterCount = 0;
	bool ebxSaved = false;
	bool ecxSaved = false;
	bool edxSaved = false;
	bool esiSaved = false;
	bool ediSaved = false;
	bool ebpSaved = false;
	for (int i=0; i < 64; ++i) {
		if ( prolog.savedRegisters[i].location != CFI_Parser<A>::kRegisterUnused ) {
			if ( prolog.savedRegisters[i].location != CFI_Parser<A>::kRegisterInCFA ) {
				sprintf(warningBuffer, "register %d saved somewhere other that in frame", i);
				return UNWIND_X86_MODE_DWARF;
			}
			switch (i) {
				case UNW_X86_EBX:
					ebxSaved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_ECX:
					ecxSaved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_EDX:
					edxSaved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_ESI:
					esiSaved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_EDI:
					ediSaved = true;
					++saveRegisterCount;
					break;
				case UNW_X86_EBP:
					ebpSaved = true;
					++saveRegisterCount;
					break;
				case DW_X86_RET_ADDR:
					break;
				default:
					sprintf(warningBuffer, "non-standard register %d being saved in prolog", i);
					return UNWIND_X86_MODE_DWARF;
			}
		}
	}
	const int32_t cfaOffsetEBX = prolog.savedRegisters[UNW_X86_EBX].value;
	const int32_t cfaOffsetECX = prolog.savedRegisters[UNW_X86_ECX].value;
	const int32_t cfaOffsetEDX = prolog.savedRegisters[UNW_X86_EDX].value;
	const int32_t cfaOffsetEDI = prolog.savedRegisters[UNW_X86_EDI].value;
	const int32_t cfaOffsetESI = prolog.savedRegisters[UNW_X86_ESI].value;
	const int32_t cfaOffsetEBP = prolog.savedRegisters[UNW_X86_EBP].value;
	
	// encode standard RBP frames
	compact_unwind_encoding_t  encoding = 0;
	if ( standardEBPframe ) {
		//		|              |
		//		+--------------+   <- CFA
		//		|   ret addr   |
		//		+--------------+
		//		|     ebp      |
		//		+--------------+   <- ebp
		//		~              ~
		//		+--------------+   
		//		|  saved reg3  |
		//		+--------------+   <- CFA - offset+8
		//		|  saved reg2  |
		//		+--------------+   <- CFA - offset+e
		//		|  saved reg1  |
		//		+--------------+   <- CFA - offset
		//		|              |
		//		+--------------+
		//		|              |
		//						   <- esp
		//
		encoding = UNWIND_X86_MODE_EBP_FRAME;
		
		// find save location of farthest register from ebp
		int furthestCfaOffset = 0;
		if ( ebxSaved & (cfaOffsetEBX < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetEBX;
		if ( ecxSaved & (cfaOffsetECX < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetECX;
		if ( edxSaved & (cfaOffsetEDX < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetEDX;
		if ( ediSaved & (cfaOffsetEDI < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetEDI;
		if ( esiSaved & (cfaOffsetESI < furthestCfaOffset) )
			furthestCfaOffset = cfaOffsetESI;
		
		if ( furthestCfaOffset == 0 ) {
			// no registers saved, nothing more to encode
			return encoding;
		}
		
		// add stack offset to encoding
		int ebpOffset = furthestCfaOffset + 8;
		int encodedOffset = ebpOffset/(-4);
		if ( encodedOffset > 255 ) {
			strcpy(warningBuffer, "offset of saved registers too far to encode");
			return UNWIND_X86_MODE_DWARF;
		}
		encoding |= (encodedOffset << __builtin_ctz(UNWIND_X86_EBP_FRAME_OFFSET));
		
		// add register saved from each stack location
		bool encodingFailure = false;
		if ( ebxSaved )
			encoding |= getEBPEncodedRegister(UNW_X86_EBX, cfaOffsetEBX - furthestCfaOffset, encodingFailure);
		if ( ecxSaved )
			encoding |= getEBPEncodedRegister(UNW_X86_ECX, cfaOffsetECX - furthestCfaOffset, encodingFailure);
		if ( edxSaved )
			encoding |= getEBPEncodedRegister(UNW_X86_EDX, cfaOffsetEDX - furthestCfaOffset, encodingFailure);
		if ( ediSaved )
			encoding |= getEBPEncodedRegister(UNW_X86_EDI, cfaOffsetEDI - furthestCfaOffset, encodingFailure);
		if ( esiSaved )
			encoding |= getEBPEncodedRegister(UNW_X86_ESI, cfaOffsetESI - furthestCfaOffset, encodingFailure);
		
		if ( encodingFailure ){
			strcpy(warningBuffer, "saved registers not contiguous");
			return UNWIND_X86_MODE_DWARF;
		}

		return encoding;
	}
	else {
		//		|              |
		//		+--------------+   <- CFA
		//		|   ret addr   |
		//		+--------------+
		//		|  saved reg1  |
		//		+--------------+   <- CFA - 8
		//		|  saved reg2  |
		//		+--------------+   <- CFA - 12
		//		|  saved reg3  |
		//		+--------------+   <- CFA - 16
		//		|  saved reg4  |
		//		+--------------+   <- CFA - 20
		//		|  saved reg5  |
		//		+--------------+   <- CFA - 24
		//		|  saved reg6  |
		//		+--------------+   <- CFA - 28
		//		|              |
		//						   <- esp
		//

		// for ESP based frames we need to encode stack size in unwind info
		encoding = UNWIND_X86_MODE_STACK_IMMD;
		uint64_t stackValue = prolog.cfaRegisterOffset / 4;
		uint32_t stackAdjust = 0;
		bool immedStackSize = true;
		const uint32_t stackMaxImmedValue = EXTRACT_BITS(0xFFFFFFFF,UNWIND_X86_FRAMELESS_STACK_SIZE);
		if ( stackValue > stackMaxImmedValue ) {
			// stack size is too big to fit as an immediate value, so encode offset of subq instruction in function
			pint_t functionContentAdjustStackIns = funcAddr + prolog.codeOffsetAtStackDecrement - 4;		
			uint32_t stackDecrementInCode = addressSpace.get32(functionContentAdjustStackIns);
			stackAdjust = (prolog.cfaRegisterOffset - stackDecrementInCode)/4;
			stackValue = functionContentAdjustStackIns - funcAddr;
			immedStackSize = false;
			if ( stackAdjust > 7 ) {
				strcpy(warningBuffer, "stack subq instruction is too different from dwarf stack size");
				return UNWIND_X86_MODE_DWARF;
			}
			encoding = UNWIND_X86_MODE_STACK_IND;
		}	
		
		
		// validate that saved registers are all within 6 slots abutting return address
		int registers[6];
		for (int i=0; i < 6;++i)
			registers[i] = 0;
		if ( ebxSaved ) {
			if ( cfaOffsetEBX < -28 ) {
				strcpy(warningBuffer, "ebx is saved too far from return address");
				return UNWIND_X86_MODE_DWARF;
			}
			registers[(cfaOffsetEBX+28)/4] = UNWIND_X86_REG_EBX;
		}
		if ( ecxSaved ) {
			if ( cfaOffsetECX < -28 ) {
				strcpy(warningBuffer, "ecx is saved too far from return address");
				return UNWIND_X86_MODE_DWARF;
			}
			registers[(cfaOffsetECX+28)/4] = UNWIND_X86_REG_ECX;
		}
		if ( edxSaved ) {
			if ( cfaOffsetEDX < -28 ) {
				strcpy(warningBuffer, "edx is saved too far from return address");
				return UNWIND_X86_MODE_DWARF;
			}
			registers[(cfaOffsetEDX+28)/4] = UNWIND_X86_REG_EDX;
		}
		if ( ediSaved ) {
			if ( cfaOffsetEDI < -28 ) {
				strcpy(warningBuffer, "edi is saved too far from return address");
				return UNWIND_X86_MODE_DWARF;
			}
			registers[(cfaOffsetEDI+28)/4] = UNWIND_X86_REG_EDI;
		}
		if ( esiSaved ) {
			if ( cfaOffsetESI < -28 ) {
				strcpy(warningBuffer, "esi is saved too far from return address");
				return UNWIND_X86_MODE_DWARF;
			}
			registers[(cfaOffsetESI+28)/4] = UNWIND_X86_REG_ESI;
		}
		if ( ebpSaved ) {
			if ( cfaOffsetEBP < -28 ) {
				strcpy(warningBuffer, "ebp is saved too far from return address");
				return UNWIND_X86_MODE_DWARF;
			}
			registers[(cfaOffsetEBP+28)/4] = UNWIND_X86_REG_EBP;
		}
		
		// validate that saved registers are contiguous and abut return address on stack
		for (int i=0; i < saveRegisterCount; ++i) {
			if ( registers[5-i] == 0 ) {
				strcpy(warningBuffer, "registers not save contiguously in stack");
				return UNWIND_X86_MODE_DWARF;
			}
		}
				
		// encode register permutation
		// the 10-bits are encoded differently depending on the number of registers saved
		int renumregs[6];
		for (int i=6-saveRegisterCount; i < 6; ++i) {
			int countless = 0;
			for (int j=6-saveRegisterCount; j < i; ++j) {
				if ( registers[j] < registers[i] )
					++countless;
			}
			renumregs[i] = registers[i] - countless -1;
		}
		uint32_t permutationEncoding = 0;
		switch ( saveRegisterCount ) {
			case 6:
				permutationEncoding |= (120*renumregs[0] + 24*renumregs[1] + 6*renumregs[2] + 2*renumregs[3] + renumregs[4]);
				break;
			case 5:
				permutationEncoding |= (120*renumregs[1] + 24*renumregs[2] + 6*renumregs[3] + 2*renumregs[4] + renumregs[5]);
				break;
			case 4:
				permutationEncoding |= (60*renumregs[2] + 12*renumregs[3] + 3*renumregs[4] + renumregs[5]);
				break;
			case 3:
				permutationEncoding |= (20*renumregs[3] + 4*renumregs[4] + renumregs[5]);
				break;
			case 2:
				permutationEncoding |= (5*renumregs[4] + renumregs[5]);
				break;
			case 1:
				permutationEncoding |= (renumregs[5]);
				break;
		}
		
		encoding |= (stackValue << __builtin_ctz(UNWIND_X86_FRAMELESS_STACK_SIZE));
		encoding |= (stackAdjust << __builtin_ctz(UNWIND_X86_FRAMELESS_STACK_ADJUST));
		encoding |= (saveRegisterCount << __builtin_ctz(UNWIND_X86_FRAMELESS_STACK_REG_COUNT));
		encoding |= (permutationEncoding << __builtin_ctz(UNWIND_X86_FRAMELESS_STACK_REG_PERMUTATION));
		return encoding;
	}
}





} // namespace lldb_private


#endif // __DWARF_INSTRUCTIONS_HPP__




