/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- CompactUnwinder.hpp -------------------------------------*- C++ -*-===//
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

#ifndef __COMPACT_UNWINDER_HPP__
#define __COMPACT_UNWINDER_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <libunwind.h>
#include <mach-o/compact_unwind_encoding.h>

#include "AddressSpace.hpp"
#include "Registers.hpp"



#define EXTRACT_BITS(value, mask) \
	( (value >> __builtin_ctz(mask)) & (((1 << __builtin_popcount(mask)))-1) )

#define SUPPORT_OLD_BINARIES 0 

namespace lldb_private {



///
/// CompactUnwinder_x86 uses a compact unwind info to virtually "step" (aka unwind) by
/// modifying a Registers_x86 register set
///
template <typename A>
class CompactUnwinder_x86
{
public:

	static int stepWithCompactEncoding(compact_unwind_encoding_t info, uint32_t functionStart, A& addressSpace, Registers_x86& registers);
	
private:
	typename A::pint_t		pint_t;
	
	static void frameUnwind(A& addressSpace, Registers_x86& registers);
	static void framelessUnwind(A& addressSpace, typename A::pint_t returnAddressLocation, Registers_x86& registers);
	static int stepWithCompactEncodingEBPFrame(compact_unwind_encoding_t compactEncoding, uint32_t functionStart, A& addressSpace, Registers_x86& registers);
	static int stepWithCompactEncodingFrameless(compact_unwind_encoding_t compactEncoding, uint32_t functionStart, A& addressSpace, Registers_x86& registers, bool indirectStackSize);
#if SUPPORT_OLD_BINARIES
	static int stepWithCompactEncodingCompat(compact_unwind_encoding_t compactEncoding, uint32_t functionStart, A& addressSpace, Registers_x86& registers);
#endif
};



template <typename A>
int CompactUnwinder_x86<A>::stepWithCompactEncoding(compact_unwind_encoding_t compactEncoding, uint32_t functionStart, A& addressSpace, Registers_x86& registers)
{
	//fprintf(stderr, "stepWithCompactEncoding(0x%08X)\n", compactEncoding);
	switch ( compactEncoding & UNWIND_X86_MODE_MASK ) {
#if SUPPORT_OLD_BINARIES
		case UNWIND_X86_MODE_COMPATIBILITY:
			return stepWithCompactEncodingCompat(compactEncoding, functionStart, addressSpace, registers);
#endif
		case UNWIND_X86_MODE_EBP_FRAME:
			return stepWithCompactEncodingEBPFrame(compactEncoding, functionStart, addressSpace, registers);
		case UNWIND_X86_MODE_STACK_IMMD:
			return stepWithCompactEncodingFrameless(compactEncoding, functionStart, addressSpace, registers, false);
		case UNWIND_X86_MODE_STACK_IND:
			return stepWithCompactEncodingFrameless(compactEncoding, functionStart, addressSpace, registers, true);
	}
	ABORT("invalid compact unwind encoding");
}


template <typename A>
int CompactUnwinder_x86<A>::stepWithCompactEncodingEBPFrame(compact_unwind_encoding_t compactEncoding, uint32_t functionStart, 
																	A& addressSpace, Registers_x86& registers)
{
	uint32_t savedRegistersOffset = EXTRACT_BITS(compactEncoding, UNWIND_X86_EBP_FRAME_OFFSET);
	uint32_t savedRegistersLocations = EXTRACT_BITS(compactEncoding, UNWIND_X86_EBP_FRAME_REGISTERS);
	
	uint64_t savedRegisters = registers.getEBP() - 4*savedRegistersOffset;
	for (int i=0; i < 5; ++i) {
		switch (savedRegistersLocations & 0x7) {
			case UNWIND_X86_REG_NONE:
				// no register saved in this slot
				break;
			case UNWIND_X86_REG_EBX:
				registers.setEBX(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_ECX:
				registers.setECX(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_EDX:
				registers.setEDX(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_EDI:
				registers.setEDI(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_ESI:
				registers.setESI(addressSpace.get32(savedRegisters));
				break;
			default:
				DEBUG_MESSAGE("bad register for EBP frame, encoding=%08X for function starting at 0x%X\n", compactEncoding, functionStart);
				ABORT("invalid compact unwind encoding");
		}
		savedRegisters += 4;
		savedRegistersLocations = (savedRegistersLocations >> 3);
	}
	frameUnwind(addressSpace, registers);
	return UNW_STEP_SUCCESS;
}


template <typename A>
int CompactUnwinder_x86<A>::stepWithCompactEncodingFrameless(compact_unwind_encoding_t encoding, uint32_t functionStart, 
																A& addressSpace, Registers_x86& registers, bool indirectStackSize)
{
	uint32_t stackSizeEncoded = EXTRACT_BITS(encoding, UNWIND_X86_FRAMELESS_STACK_SIZE);
	uint32_t stackAdjust = EXTRACT_BITS(encoding, UNWIND_X86_FRAMELESS_STACK_ADJUST);
	uint32_t regCount = EXTRACT_BITS(encoding, UNWIND_X86_FRAMELESS_STACK_REG_COUNT);
	uint32_t permutation = EXTRACT_BITS(encoding, UNWIND_X86_FRAMELESS_STACK_REG_PERMUTATION);
	uint32_t stackSize = stackSizeEncoded*4;
	if ( indirectStackSize ) {
		// stack size is encoded in subl $xxx,%esp instruction
		uint32_t subl = addressSpace.get32(functionStart+stackSizeEncoded);
		stackSize = subl + 4*stackAdjust;
	}
	// decompress permutation
	int permunreg[6];
	switch ( regCount ) {
		case 6:
			permunreg[0] = permutation/120;
			permutation -= (permunreg[0]*120);
			permunreg[1] = permutation/24;
			permutation -= (permunreg[1]*24);
			permunreg[2] = permutation/6;
			permutation -= (permunreg[2]*6);
			permunreg[3] = permutation/2;
			permutation -= (permunreg[3]*2);
			permunreg[4] = permutation;
			permunreg[5] = 0;
			break;
		case 5:
			permunreg[0] = permutation/120;
			permutation -= (permunreg[0]*120);
			permunreg[1] = permutation/24;
			permutation -= (permunreg[1]*24);
			permunreg[2] = permutation/6;
			permutation -= (permunreg[2]*6);
			permunreg[3] = permutation/2;
			permutation -= (permunreg[3]*2);
			permunreg[4] = permutation;
			break;
		case 4:
			permunreg[0] = permutation/60;
			permutation -= (permunreg[0]*60);
			permunreg[1] = permutation/12;
			permutation -= (permunreg[1]*12);
			permunreg[2] = permutation/3;
			permutation -= (permunreg[2]*3);
			permunreg[3] = permutation;
			break;
		case 3:
			permunreg[0] = permutation/20;
			permutation -= (permunreg[0]*20);
			permunreg[1] = permutation/4;
			permutation -= (permunreg[1]*4);
			permunreg[2] = permutation;
			break;
		case 2:
			permunreg[0] = permutation/5;
			permutation -= (permunreg[0]*5);
			permunreg[1] = permutation;
			break;
		case 1:
			permunreg[0] = permutation;
			break;
	}
	// re-number registers back to standard numbers
	int registersSaved[6];
	bool used[7] = { false, false, false, false, false, false, false };
	for (uint32_t i=0; i < regCount; ++i) {
		int renum = 0; 
		for (int u=1; u < 7; ++u) {
			if ( !used[u] ) {
				if ( renum == permunreg[i] ) {
					registersSaved[i] = u;
					used[u] = true;
					break;
				}
				++renum;
			}
		}
	}
	uint64_t savedRegisters = registers.getSP() + stackSize - 4 - 4*regCount;
	for (uint32_t i=0; i < regCount; ++i) {
		switch ( registersSaved[i] ) {
			case UNWIND_X86_REG_EBX:
				registers.setEBX(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_ECX:
				registers.setECX(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_EDX:
				registers.setEDX(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_EDI:
				registers.setEDI(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_ESI:
				registers.setESI(addressSpace.get32(savedRegisters));
				break;
			case UNWIND_X86_REG_EBP:
				registers.setEBP(addressSpace.get32(savedRegisters));
				break;
			default:
				DEBUG_MESSAGE("bad register for frameless, encoding=%08X for function starting at 0x%X\n", encoding, functionStart);
				ABORT("invalid compact unwind encoding");
		}
		savedRegisters += 4;
	}
	framelessUnwind(addressSpace, savedRegisters, registers);
	return UNW_STEP_SUCCESS;
}


#if SUPPORT_OLD_BINARIES
template <typename A>
int CompactUnwinder_x86<A>::stepWithCompactEncodingCompat(compact_unwind_encoding_t compactEncoding, uint32_t functionStart, A& addressSpace, Registers_x86& registers)
{
	//fprintf(stderr, "stepWithCompactEncoding(0x%08X)\n", compactEncoding);
	typename A::pint_t savedRegisters;
	uint32_t stackValue = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_SIZE);
	uint32_t stackSize;
	uint32_t stackAdjust;
	switch (compactEncoding & UNWIND_X86_CASE_MASK ) {
		case UNWIND_X86_UNWIND_INFO_UNSPECIFIED:
			return UNW_ENOINFO;
	
		case UNWIND_X86_EBP_FRAME_NO_REGS:
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
			
		case UNWIND_X86_EBP_FRAME_EBX:
			savedRegisters = registers.getEBP() - 4;
			registers.setEBX(addressSpace.get32(savedRegisters));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_EBP_FRAME_ESI:
			savedRegisters = registers.getEBP() - 4;
			registers.setESI(addressSpace.get32(savedRegisters));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_EBP_FRAME_EDI:
			savedRegisters = registers.getEBP() - 4;
			registers.setEDI(addressSpace.get32(savedRegisters));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_EBP_FRAME_EBX_ESI:
			savedRegisters = registers.getEBP() - 8;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_EBP_FRAME_ESI_EDI:
			savedRegisters = registers.getEBP() - 8;
			registers.setESI(addressSpace.get32(savedRegisters));
			registers.setEDI(addressSpace.get32(savedRegisters+4));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_EBP_FRAME_EBX_ESI_EDI:
			savedRegisters = registers.getEBP() - 12;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			registers.setEDI(addressSpace.get32(savedRegisters+8));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_EBP_FRAME_EBX_EDI:
			savedRegisters = registers.getEBP() - 8;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setEDI(addressSpace.get32(savedRegisters+4));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_NO_REGS:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*0;
			framelessUnwind(addressSpace, savedRegisters+4*0, registers);
			return UNW_STEP_SUCCESS;
				
		case UNWIND_X86_IMM_STK_EBX:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*1;
			registers.setEBX(addressSpace.get32(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+4*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_ESI:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*1;
			registers.setESI(addressSpace.get32(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+4*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_EDI:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*1;
			registers.setEDI(addressSpace.get32(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+4*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_EBX_ESI:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*2;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			framelessUnwind(addressSpace, savedRegisters+4*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_ESI_EDI:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*2;
			registers.setESI(addressSpace.get32(savedRegisters));
			registers.setEDI(addressSpace.get32(savedRegisters+4));
			framelessUnwind(addressSpace, savedRegisters+4*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_ESI_EDI_EBP:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*3;
			registers.setESI(addressSpace.get32(savedRegisters));
			registers.setEDI(addressSpace.get32(savedRegisters+4));
			registers.setEBP(addressSpace.get32(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+4*3, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_EBX_ESI_EDI:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*3;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			registers.setEDI(addressSpace.get32(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+4*3, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IMM_STK_EBX_ESI_EDI_EBP:
			stackSize = stackValue * 4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*4;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			registers.setEDI(addressSpace.get32(savedRegisters+8));
			registers.setEBP(addressSpace.get32(savedRegisters+12));
			framelessUnwind(addressSpace, savedRegisters+4*4, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_IND_STK_NO_REGS:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*0;
			framelessUnwind(addressSpace, savedRegisters+4*0, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_IND_STK_EBX:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*1;
			registers.setEBX(addressSpace.get32(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+4*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IND_STK_ESI:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*1;
			registers.setESI(addressSpace.get32(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+4*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IND_STK_EDI:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*1;
			registers.setEDI(addressSpace.get32(savedRegisters));
			return UNW_STEP_SUCCESS;
			framelessUnwind(addressSpace, savedRegisters+4*1, registers);
		
		case UNWIND_X86_IND_STK_EBX_ESI:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*2;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			framelessUnwind(addressSpace, savedRegisters+4*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IND_STK_ESI_EDI:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*2;
			registers.setESI(addressSpace.get32(savedRegisters));
			registers.setEDI(addressSpace.get32(savedRegisters+4));
			framelessUnwind(addressSpace, savedRegisters+4*2, registers);
			return UNW_STEP_SUCCESS;
			
		case UNWIND_X86_IND_STK_ESI_EDI_EBP:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*3;
			registers.setESI(addressSpace.get32(savedRegisters));
			registers.setEDI(addressSpace.get32(savedRegisters+4));
			registers.setEBP(addressSpace.get32(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+4*3, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_IND_STK_EBX_ESI_EDI:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*3;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			registers.setEDI(addressSpace.get32(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+4*3, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_IND_STK_EBX_ESI_EDI_EBP:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_STACK_ADJUST);
			stackSize += stackAdjust*4;
			savedRegisters = registers.getSP() + stackSize - 4 - 4*4;
			registers.setEBX(addressSpace.get32(savedRegisters));
			registers.setESI(addressSpace.get32(savedRegisters+4));
			registers.setEDI(addressSpace.get32(savedRegisters+8));
			registers.setEBP(addressSpace.get32(savedRegisters+12));
			framelessUnwind(addressSpace, savedRegisters+4*4, registers);
			return UNW_STEP_SUCCESS;
		
		default:
			DEBUG_MESSAGE("unknown compact unwind encoding %08X for function starting at 0x%X\n", 
				compactEncoding & UNWIND_X86_CASE_MASK, functionStart);
			ABORT("unknown compact unwind encoding");
	}
	return UNW_EINVAL;
}
#endif // SUPPORT_OLD_BINARIES



template <typename A>
void CompactUnwinder_x86<A>::frameUnwind(A& addressSpace, Registers_x86& registers)
{
	typename A::pint_t bp = registers.getEBP();
	// ebp points to old ebp
	registers.setEBP(addressSpace.get32(bp));
	// old esp is ebp less saved ebp and return address
	registers.setSP(bp+8);
	// pop return address into eip
	registers.setIP(addressSpace.get32(bp+4));
}

template <typename A>
void CompactUnwinder_x86<A>::framelessUnwind(A& addressSpace, typename A::pint_t returnAddressLocation, Registers_x86& registers)
{
	// return address is on stack after last saved register
	registers.setIP(addressSpace.get32(returnAddressLocation));
	// old esp is before return address
	registers.setSP(returnAddressLocation+4);
}





///
/// CompactUnwinder_x86_64 uses a compact unwind info to virtually "step" (aka unwind) by
/// modifying a Registers_x86_64 register set
///
template <typename A>
class CompactUnwinder_x86_64
{
public:

	static int stepWithCompactEncoding(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, A& addressSpace, Registers_x86_64& registers);
	
private:
	typename A::pint_t		pint_t;
	
	static void frameUnwind(A& addressSpace, Registers_x86_64& registers);
	static void framelessUnwind(A& addressSpace, uint64_t returnAddressLocation, Registers_x86_64& registers);
	static int stepWithCompactEncodingRBPFrame(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, A& addressSpace, Registers_x86_64& registers);
	static int stepWithCompactEncodingFrameless(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, A& addressSpace, Registers_x86_64& registers, bool indirectStackSize);
#if SUPPORT_OLD_BINARIES
	static int stepWithCompactEncodingCompat(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, A& addressSpace, Registers_x86_64& registers);
#endif
};


template <typename A>
int CompactUnwinder_x86_64<A>::stepWithCompactEncoding(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, A& addressSpace, Registers_x86_64& registers)
{
	//fprintf(stderr, "stepWithCompactEncoding(0x%08X)\n", compactEncoding);
	switch ( compactEncoding & UNWIND_X86_64_MODE_MASK ) {
#if SUPPORT_OLD_BINARIES
		case UNWIND_X86_64_MODE_COMPATIBILITY:
			return stepWithCompactEncodingCompat(compactEncoding, functionStart, addressSpace, registers);
#endif
		case UNWIND_X86_64_MODE_RBP_FRAME:
			return stepWithCompactEncodingRBPFrame(compactEncoding, functionStart, addressSpace, registers);
		case UNWIND_X86_64_MODE_STACK_IMMD:
			return stepWithCompactEncodingFrameless(compactEncoding, functionStart, addressSpace, registers, false);
		case UNWIND_X86_64_MODE_STACK_IND:
			return stepWithCompactEncodingFrameless(compactEncoding, functionStart, addressSpace, registers, true);
	}
	ABORT("invalid compact unwind encoding");
}

	
template <typename A>
int CompactUnwinder_x86_64<A>::stepWithCompactEncodingRBPFrame(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, 
																	A& addressSpace, Registers_x86_64& registers)
{
	uint32_t savedRegistersOffset = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_RBP_FRAME_OFFSET);
	uint32_t savedRegistersLocations = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_RBP_FRAME_REGISTERS);
	
	uint64_t savedRegisters = registers.getRBP() - 8*savedRegistersOffset;
	for (int i=0; i < 5; ++i) {
        int readerr = 0;
		switch (savedRegistersLocations & 0x7) {
			case UNWIND_X86_64_REG_NONE:
				// no register saved in this slot
				break;
			case UNWIND_X86_64_REG_RBX:
				registers.setRBX(addressSpace.get64(savedRegisters, readerr));
				break;
			case UNWIND_X86_64_REG_R12:
				registers.setR12(addressSpace.get64(savedRegisters, readerr));
				break;
			case UNWIND_X86_64_REG_R13:
				registers.setR13(addressSpace.get64(savedRegisters, readerr));
				break;
			case UNWIND_X86_64_REG_R14:
				registers.setR14(addressSpace.get64(savedRegisters, readerr));
				break;
			case UNWIND_X86_64_REG_R15:
				registers.setR15(addressSpace.get64(savedRegisters, readerr));
				break;
			default:
				DEBUG_MESSAGE("bad register for RBP frame, encoding=%08X for function starting at 0x%llX\n", compactEncoding, functionStart);
				ABORT("invalid compact unwind encoding");
		}
        // Error reading memory while doing a remote unwind?
        if (readerr)
            return UNW_STEP_END;

		savedRegisters += 8;
		savedRegistersLocations = (savedRegistersLocations >> 3);
	}
	frameUnwind(addressSpace, registers);
	return UNW_STEP_SUCCESS;
}

		
template <typename A>
int CompactUnwinder_x86_64<A>::stepWithCompactEncodingFrameless(compact_unwind_encoding_t encoding, uint64_t functionStart, 
																A& addressSpace, Registers_x86_64& registers, bool indirectStackSize)
{
	uint32_t stackSizeEncoded = EXTRACT_BITS(encoding, UNWIND_X86_64_FRAMELESS_STACK_SIZE);
	uint32_t stackAdjust = EXTRACT_BITS(encoding, UNWIND_X86_64_FRAMELESS_STACK_ADJUST);
	uint32_t regCount = EXTRACT_BITS(encoding, UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT);
	uint32_t permutation = EXTRACT_BITS(encoding, UNWIND_X86_64_FRAMELESS_STACK_REG_PERMUTATION);
	uint32_t stackSize = stackSizeEncoded*8;
	if ( indirectStackSize ) {
		// stack size is encoded in subl $xxx,%esp instruction
		uint32_t subl = addressSpace.get32(functionStart+stackSizeEncoded);
		stackSize = subl + 8*stackAdjust;
	}
	// decompress permutation
	int permunreg[6];
	switch ( regCount ) {
		case 6:
			permunreg[0] = permutation/120;
			permutation -= (permunreg[0]*120);
			permunreg[1] = permutation/24;
			permutation -= (permunreg[1]*24);
			permunreg[2] = permutation/6;
			permutation -= (permunreg[2]*6);
			permunreg[3] = permutation/2;
			permutation -= (permunreg[3]*2);
			permunreg[4] = permutation;
			permunreg[5] = 0;
			break;
		case 5:
			permunreg[0] = permutation/120;
			permutation -= (permunreg[0]*120);
			permunreg[1] = permutation/24;
			permutation -= (permunreg[1]*24);
			permunreg[2] = permutation/6;
			permutation -= (permunreg[2]*6);
			permunreg[3] = permutation/2;
			permutation -= (permunreg[3]*2);
			permunreg[4] = permutation;
			break;
		case 4:
			permunreg[0] = permutation/60;
			permutation -= (permunreg[0]*60);
			permunreg[1] = permutation/12;
			permutation -= (permunreg[1]*12);
			permunreg[2] = permutation/3;
			permutation -= (permunreg[2]*3);
			permunreg[3] = permutation;
			break;
		case 3:
			permunreg[0] = permutation/20;
			permutation -= (permunreg[0]*20);
			permunreg[1] = permutation/4;
			permutation -= (permunreg[1]*4);
			permunreg[2] = permutation;
			break;
		case 2:
			permunreg[0] = permutation/5;
			permutation -= (permunreg[0]*5);
			permunreg[1] = permutation;
			break;
		case 1:
			permunreg[0] = permutation;
			break;
	}
	// re-number registers back to standard numbers
	int registersSaved[6];
	bool used[7] = { false, false, false, false, false, false, false };
	for (uint32_t i=0; i < regCount; ++i) {
		int renum = 0; 
		for (int u=1; u < 7; ++u) {
			if ( !used[u] ) {
				if ( renum == permunreg[i] ) {
					registersSaved[i] = u;
					used[u] = true;
					break;
				}
				++renum;
			}
		}
	}
	uint64_t savedRegisters = registers.getSP() + stackSize - 8 - 8*regCount;
	for (uint32_t i=0; i < regCount; ++i) {
		switch ( registersSaved[i] ) {
			case UNWIND_X86_64_REG_RBX:
				registers.setRBX(addressSpace.get64(savedRegisters));
				break;
			case UNWIND_X86_64_REG_R12:
				registers.setR12(addressSpace.get64(savedRegisters));
				break;
			case UNWIND_X86_64_REG_R13:
				registers.setR13(addressSpace.get64(savedRegisters));
				break;
			case UNWIND_X86_64_REG_R14:
				registers.setR14(addressSpace.get64(savedRegisters));
				break;
			case UNWIND_X86_64_REG_R15:
				registers.setR15(addressSpace.get64(savedRegisters));
				break;
			case UNWIND_X86_64_REG_RBP:
				registers.setRBP(addressSpace.get64(savedRegisters));
				break;
			default:
				DEBUG_MESSAGE("bad register for frameless, encoding=%08X for function starting at 0x%llX\n", encoding, functionStart);
				ABORT("invalid compact unwind encoding");
		}
		savedRegisters += 8;
	}
	framelessUnwind(addressSpace, savedRegisters, registers);
	return UNW_STEP_SUCCESS;
}

#if SUPPORT_OLD_BINARIES
template <typename A>
int CompactUnwinder_x86_64<A>::stepWithCompactEncodingCompat(compact_unwind_encoding_t compactEncoding, uint64_t functionStart, A& addressSpace, Registers_x86_64& registers)
{
	uint64_t savedRegisters;
	uint32_t stackValue = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_SIZE);
	uint64_t stackSize;
	uint32_t stackAdjust;
	
	switch (compactEncoding & UNWIND_X86_64_CASE_MASK ) {
		case UNWIND_X86_64_UNWIND_INFO_UNSPECIFIED:
			return UNW_ENOINFO;
			
		case UNWIND_X86_64_RBP_FRAME_NO_REGS:
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
			
		case UNWIND_X86_64_RBP_FRAME_RBX:
			savedRegisters = registers.getRBP() - 8*1;
			registers.setRBX(addressSpace.get64(savedRegisters));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_RBP_FRAME_RBX_R12:
			savedRegisters = registers.getRBP() - 8*2;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_RBP_FRAME_RBX_R12_R13:
			savedRegisters = registers.getRBP() - 8*3;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_RBP_FRAME_RBX_R12_R13_R14:
			savedRegisters = registers.getRBP() - 8*4;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			registers.setR14(addressSpace.get64(savedRegisters+24));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_RBP_FRAME_RBX_R12_R13_R14_R15:
			savedRegisters = registers.getRBP() - 8*5;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			registers.setR14(addressSpace.get64(savedRegisters+24));
			registers.setR15(addressSpace.get64(savedRegisters+32));
			frameUnwind(addressSpace, registers);
			return UNW_STEP_SUCCESS;
	
		case UNWIND_X86_64_IMM_STK_NO_REGS:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*0;
			framelessUnwind(addressSpace, savedRegisters+8*0, registers);
			return UNW_STEP_SUCCESS;
				
		case UNWIND_X86_64_IMM_STK_RBX:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*1;
			registers.setRBX(addressSpace.get64(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+8*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IMM_STK_RBX_R12:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*2;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+8*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IMM_STK_RBX_RBP:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*2;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+8*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IMM_STK_RBX_R12_R13:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*3;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			framelessUnwind(addressSpace, savedRegisters+8*3, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IMM_STK_RBX_R12_R13_R14:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*4;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			registers.setR14(addressSpace.get64(savedRegisters+24));
			framelessUnwind(addressSpace, savedRegisters+8*4, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IMM_STK_RBX_R12_R13_R14_R15:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*5;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			registers.setR14(addressSpace.get64(savedRegisters+24));
			registers.setR15(addressSpace.get64(savedRegisters+32));
			framelessUnwind(addressSpace, savedRegisters+8*5, registers);
			return UNW_STEP_SUCCESS;
			
		case UNWIND_X86_64_IMM_STK_RBX_RBP_R12_R13_R14_R15:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*6;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			registers.setR13(addressSpace.get64(savedRegisters+24));
			registers.setR14(addressSpace.get64(savedRegisters+32));
			registers.setR15(addressSpace.get64(savedRegisters+40));
			framelessUnwind(addressSpace, savedRegisters+8*6, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_IMM_STK_RBX_RBP_R12:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*3;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			framelessUnwind(addressSpace, savedRegisters+8*3, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_IMM_STK_RBX_RBP_R12_R13:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*4;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			registers.setR13(addressSpace.get64(savedRegisters+24));
			framelessUnwind(addressSpace, savedRegisters+8*4, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_IMM_STK_RBX_RBP_R12_R13_R14:
			stackSize = stackValue * 8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*5;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			registers.setR13(addressSpace.get64(savedRegisters+24));
			registers.setR14(addressSpace.get64(savedRegisters+32));
			framelessUnwind(addressSpace, savedRegisters+8*5, registers);
			return UNW_STEP_SUCCESS;

		case UNWIND_X86_64_IND_STK_NO_REGS:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*0;
			framelessUnwind(addressSpace, savedRegisters+8*0, registers);
			return UNW_STEP_SUCCESS;
				
		case UNWIND_X86_64_IND_STK_RBX:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*1;
			registers.setRBX(addressSpace.get64(savedRegisters));
			framelessUnwind(addressSpace, savedRegisters+8*1, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_R12:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*2;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+8*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_RBP:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*2;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			framelessUnwind(addressSpace, savedRegisters+8*2, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_R12_R13:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*3;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			framelessUnwind(addressSpace, savedRegisters+8*3, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_R12_R13_R14:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*4;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			registers.setR14(addressSpace.get64(savedRegisters+24));
			framelessUnwind(addressSpace, savedRegisters+8*4, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_R12_R13_R14_R15:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*5;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setR12(addressSpace.get64(savedRegisters+8));
			registers.setR13(addressSpace.get64(savedRegisters+16));
			registers.setR14(addressSpace.get64(savedRegisters+24));
			registers.setR15(addressSpace.get64(savedRegisters+32));
			framelessUnwind(addressSpace, savedRegisters+8*5, registers);
			return UNW_STEP_SUCCESS;
			
		case UNWIND_X86_64_IND_STK_RBX_RBP_R12_R13_R14_R15:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*6;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			registers.setR13(addressSpace.get64(savedRegisters+24));
			registers.setR14(addressSpace.get64(savedRegisters+32));
			registers.setR15(addressSpace.get64(savedRegisters+40));
			framelessUnwind(addressSpace, savedRegisters+8*6, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_RBP_R12:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*3;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			framelessUnwind(addressSpace, savedRegisters+8*3, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_RBP_R12_R13:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*4;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			registers.setR13(addressSpace.get64(savedRegisters+24));
			framelessUnwind(addressSpace, savedRegisters+8*4, registers);
			return UNW_STEP_SUCCESS;
		
		case UNWIND_X86_64_IND_STK_RBX_RBP_R12_R13_R14:
			stackSize = addressSpace.get32(functionStart+stackValue);
			stackAdjust = EXTRACT_BITS(compactEncoding, UNWIND_X86_64_STACK_ADJUST);
			stackSize += stackAdjust*8;
			savedRegisters = registers.getSP() + stackSize - 8 - 8*5;
			registers.setRBX(addressSpace.get64(savedRegisters));
			registers.setRBP(addressSpace.get64(savedRegisters+8));
			registers.setR12(addressSpace.get64(savedRegisters+16));
			registers.setR13(addressSpace.get64(savedRegisters+24));
			registers.setR14(addressSpace.get64(savedRegisters+32));
			framelessUnwind(addressSpace, savedRegisters+8*5, registers);
			return UNW_STEP_SUCCESS;
		
		default:
			DEBUG_MESSAGE("unknown compact unwind encoding %08X for function starting at 0x%llX\n", 
				compactEncoding & UNWIND_X86_64_CASE_MASK, functionStart);
			ABORT("unknown compact unwind encoding");
	}
	return UNW_EINVAL;
}
#endif // SUPPORT_OLD_BINARIES


template <typename A>
void CompactUnwinder_x86_64<A>::frameUnwind(A& addressSpace, Registers_x86_64& registers)
{
	uint64_t rbp = registers.getRBP();
	// ebp points to old ebp
	registers.setRBP(addressSpace.get64(rbp));
	// old esp is ebp less saved ebp and return address
	registers.setSP(rbp+16);
	// pop return address into eip
	registers.setIP(addressSpace.get64(rbp+8));
}

template <typename A>
void CompactUnwinder_x86_64<A>::framelessUnwind(A& addressSpace, uint64_t returnAddressLocation, Registers_x86_64& registers)
{
	// return address is on stack after last saved register
	registers.setIP(addressSpace.get64(returnAddressLocation));
	// old esp is before return address
	registers.setSP(returnAddressLocation+8);
}


}; // namespace lldb_private



#endif // __COMPACT_UNWINDER_HPP__




