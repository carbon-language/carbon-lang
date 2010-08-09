/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- Registers.hpp -------------------------------------------*- C++ -*-===//
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

#ifndef __REGISTERS_HPP__
#define __REGISTERS_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <mach-o/getsect.h>
#include <mach/i386/thread_status.h>

#include "libunwind.h"
#include "InternalMacros.h"

namespace lldb_private {


///
/// Registers_x86 holds the register state of a thread in a 32-bit intel process.  
///
class Registers_x86
{
public:	
					Registers_x86();
					Registers_x86(const void* registers);

	bool			validRegister(int num) const;
	uint32_t		getRegister(int num) const;
	void			setRegister(int num, uint32_t value);
	bool			validFloatRegister(int num) const { return false; }
	double			getFloatRegister(int num) const;
	void			setFloatRegister(int num, double value); 
	bool			validVectorRegister(int num) const { return false; }
	v128			getVectorRegister(int num) const;
	void			setVectorRegister(int num, v128 value);
	const char*		getRegisterName(int num);
	void			jumpto() {}
	
	uint32_t		getSP() const			{ return fRegisters.__esp; }
	void			setSP(uint32_t value)	{ fRegisters.__esp = value; }
	uint32_t		getIP()	const			{ return fRegisters.__eip; }
	void			setIP(uint32_t value)	{ fRegisters.__eip = value; }
	uint32_t		getEBP() const			{ return fRegisters.__ebp; }
	void			setEBP(uint32_t value)	{ fRegisters.__ebp = value; }
	uint32_t		getEBX() const			{ return fRegisters.__ebx; }
	void			setEBX(uint32_t value)	{ fRegisters.__ebx = value; }
	uint32_t		getECX() const			{ return fRegisters.__ecx; }
	void			setECX(uint32_t value)	{ fRegisters.__ecx = value; }
	uint32_t		getEDX() const			{ return fRegisters.__edx; }
	void			setEDX(uint32_t value)	{ fRegisters.__edx = value; }
	uint32_t		getESI() const			{ return fRegisters.__esi; }
	void			setESI(uint32_t value)	{ fRegisters.__esi = value; }
	uint32_t		getEDI() const			{ return fRegisters.__edi; }
	void			setEDI(uint32_t value)	{ fRegisters.__edi = value; }
	
private:
	i386_thread_state_t  fRegisters;
};

inline Registers_x86::Registers_x86(const void* registers)
{
	COMPILE_TIME_ASSERT( sizeof(Registers_x86) < sizeof(unw_context_t) );
	fRegisters = *((i386_thread_state_t*)registers); 
}

inline Registers_x86::Registers_x86()
{
	bzero(&fRegisters, sizeof(fRegisters)); 
}


inline bool Registers_x86::validRegister(int regNum) const
{
	if ( regNum == UNW_REG_IP )
		return true;
	if ( regNum == UNW_REG_SP )
		return true;
	if ( regNum < 0 )
		return false;
	if ( regNum > 7 )
		return false;
	return true;
}

inline uint32_t Registers_x86::getRegister(int regNum) const
{
	switch ( regNum ) {
		case UNW_REG_IP:
			return fRegisters.__eip;
		case UNW_REG_SP:
			return fRegisters.__esp;
		case UNW_X86_EAX:
			return fRegisters.__eax;
		case UNW_X86_ECX:
			return fRegisters.__ecx;
		case UNW_X86_EDX:
			return fRegisters.__edx;
		case UNW_X86_EBX:
			return fRegisters.__ebx;
		case UNW_X86_EBP:
			return fRegisters.__ebp;
		case UNW_X86_ESP:
			return fRegisters.__esp;
		case UNW_X86_ESI:
			return fRegisters.__esi;
		case UNW_X86_EDI:
			return fRegisters.__edi;
	}
	ABORT("unsupported x86 register");
}

inline void Registers_x86::setRegister(int regNum, uint32_t value)
{
	switch ( regNum ) {
		case UNW_REG_IP:
			fRegisters.__eip = value;
			return;
		case UNW_REG_SP:
			fRegisters.__esp = value;
			return;
		case UNW_X86_EAX:
			fRegisters.__eax = value;
			return;
		case UNW_X86_ECX:
			fRegisters.__ecx = value;
			return;
		case UNW_X86_EDX:
			fRegisters.__edx = value;
			return;
		case UNW_X86_EBX:
			fRegisters.__ebx = value;
			return;
		case UNW_X86_EBP:
			fRegisters.__ebp = value;
			return;
		case UNW_X86_ESP:
			fRegisters.__esp = value;
			return;
		case UNW_X86_ESI:
			fRegisters.__esi = value;
			return;
		case UNW_X86_EDI:
			fRegisters.__edi = value;
			return;
	}
	ABORT("unsupported x86 register");
}

inline const char* Registers_x86::getRegisterName(int regNum)
{
	switch ( regNum ) {
		case UNW_REG_IP:
			return "ip";
		case UNW_REG_SP:
			return "esp";
		case UNW_X86_EAX:
			return "eax";
		case UNW_X86_ECX:
			return "ecx";
		case UNW_X86_EDX:
			return "edx";
		case UNW_X86_EBX:
			return "ebx";
		case UNW_X86_EBP:
			return "ebp";
		case UNW_X86_ESP:
			return "esp";
		case UNW_X86_ESI:
			return "esi";
		case UNW_X86_EDI:
			return "edi";
		default:
			return "unknown register";
	}
}

inline double Registers_x86::getFloatRegister(int num) const
{
	ABORT("no x86 float registers");
}

inline void Registers_x86::setFloatRegister(int num, double value)
{
	ABORT("no x86 float registers");
}

inline v128 Registers_x86::getVectorRegister(int num) const
{
	ABORT("no x86 vector registers");
}

inline void Registers_x86::setVectorRegister(int num, v128 value)
{
	ABORT("no x86 vector registers");
}




///
/// Registers_x86_64  holds the register state of a thread in a 64-bit intel process.  
///
class Registers_x86_64
{
public:	
					Registers_x86_64();
					Registers_x86_64(const void* registers); 

	bool			validRegister(int num) const;
	uint64_t		getRegister(int num) const;
	void			setRegister(int num, uint64_t value);
	bool			validFloatRegister(int num) const{ return false; }
	double			getFloatRegister(int num) const;
	void			setFloatRegister(int num, double value);
	bool			validVectorRegister(int num) const { return false; }
	v128			getVectorRegister(int num) const;
	void			setVectorRegister(int num, v128 value);
	const char*		getRegisterName(int num);
	void			jumpto() {}
	uint64_t		getSP()	const			{ return fRegisters.__rsp; }
	void			setSP(uint64_t value)	{ fRegisters.__rsp = value; }
	uint64_t		getIP()	const			{ return fRegisters.__rip; }
	void			setIP(uint64_t value)	{ fRegisters.__rip = value; }
	uint64_t		getRBP() const			{ return fRegisters.__rbp; }
	void			setRBP(uint64_t value)	{ fRegisters.__rbp = value; }
	uint64_t		getRBX() const			{ return fRegisters.__rbx; }
	void			setRBX(uint64_t value)	{ fRegisters.__rbx = value; }
	uint64_t		getR12() const			{ return fRegisters.__r12; }
	void			setR12(uint64_t value)	{ fRegisters.__r12 = value; }
	uint64_t		getR13() const			{ return fRegisters.__r13; }
	void			setR13(uint64_t value)	{ fRegisters.__r13 = value; }
	uint64_t		getR14() const			{ return fRegisters.__r14; }
	void			setR14(uint64_t value)	{ fRegisters.__r14 = value; }
	uint64_t		getR15() const			{ return fRegisters.__r15; }
	void			setR15(uint64_t value)	{ fRegisters.__r15 = value; }
private:
	x86_thread_state64_t fRegisters;
};

inline Registers_x86_64::Registers_x86_64(const void* registers)
{
	COMPILE_TIME_ASSERT( sizeof(Registers_x86_64) < sizeof(unw_context_t) );
	fRegisters = *((x86_thread_state64_t*)registers); 
}

inline Registers_x86_64::Registers_x86_64()
{
	bzero(&fRegisters, sizeof(fRegisters)); 
}


inline bool Registers_x86_64::validRegister(int regNum) const
{
	if ( regNum == UNW_REG_IP )
		return true;
	if ( regNum == UNW_REG_SP )
		return true;
	if ( regNum < 0 )
		return false;
	if ( regNum > 15 )
		return false;
	return true;
}

inline uint64_t Registers_x86_64::getRegister(int regNum) const
{
	switch ( regNum ) {
		case UNW_REG_IP:
			return fRegisters.__rip;
		case UNW_REG_SP:
			return fRegisters.__rsp;
		case UNW_X86_64_RAX:
			return fRegisters.__rax;
		case UNW_X86_64_RDX:
			return fRegisters.__rdx;
		case UNW_X86_64_RCX:
			return fRegisters.__rcx;
		case UNW_X86_64_RBX:
			return fRegisters.__rbx;
		case UNW_X86_64_RSI:
			return fRegisters.__rsi;
		case UNW_X86_64_RDI:
			return fRegisters.__rdi;
		case UNW_X86_64_RBP:
			return fRegisters.__rbp;
		case UNW_X86_64_RSP:
			return fRegisters.__rsp;
		case UNW_X86_64_R8:
			return fRegisters.__r8;
		case UNW_X86_64_R9:
			return fRegisters.__r9;
		case UNW_X86_64_R10:
			return fRegisters.__r10;
		case UNW_X86_64_R11:
			return fRegisters.__r11;
		case UNW_X86_64_R12:
			return fRegisters.__r12;
		case UNW_X86_64_R13:
			return fRegisters.__r13;
		case UNW_X86_64_R14:
			return fRegisters.__r14;
		case UNW_X86_64_R15:
			return fRegisters.__r15;
	}
	ABORT("unsupported x86_64 register");
}

inline void Registers_x86_64::setRegister(int regNum, uint64_t value)
{
	switch ( regNum ) {
		case UNW_REG_IP:
			fRegisters.__rip = value;
			return;
		case UNW_REG_SP:
			fRegisters.__rsp = value;
			return;
		case UNW_X86_64_RAX:
			fRegisters.__rax = value;
			return;
		case UNW_X86_64_RDX:
			fRegisters.__rdx = value;
			return;
		case UNW_X86_64_RCX:
			fRegisters.__rcx = value;
			return;
		case UNW_X86_64_RBX:
			fRegisters.__rbx = value;
			return;
		case UNW_X86_64_RSI:
			fRegisters.__rsi = value;
			return;
		case UNW_X86_64_RDI:
			fRegisters.__rdi = value;
			return;
		case UNW_X86_64_RBP:
			fRegisters.__rbp = value;
			return;
		case UNW_X86_64_RSP:
			fRegisters.__rsp = value;
			return;
		case UNW_X86_64_R8:
			fRegisters.__r8 = value;
			return;
		case UNW_X86_64_R9:
			fRegisters.__r9 = value;
			return;
		case UNW_X86_64_R10:
			fRegisters.__r10 = value;
			return;
		case UNW_X86_64_R11:
			fRegisters.__r11 = value;
			return;
		case UNW_X86_64_R12:
			fRegisters.__r12 = value;
			return;
		case UNW_X86_64_R13:
			fRegisters.__r13 = value;
			return;
		case UNW_X86_64_R14:
			fRegisters.__r14 = value;
			return;
		case UNW_X86_64_R15:
			fRegisters.__r15 = value;
			return;
	}
	ABORT("unsupported x86_64 register");
}

inline const char* Registers_x86_64::getRegisterName(int regNum)
{
	switch ( regNum ) {
		case UNW_REG_IP:
			return "rip";
		case UNW_REG_SP:
			return "rsp";
		case UNW_X86_64_RAX:
			return "rax";
		case UNW_X86_64_RDX:
			return "rdx";
		case UNW_X86_64_RCX:
			return "rcx";
		case UNW_X86_64_RBX:
			return "rbx";
		case UNW_X86_64_RSI:
			return "rsi";
		case UNW_X86_64_RDI:
			return "rdi";
		case UNW_X86_64_RBP:
			return "rbp";
		case UNW_X86_64_RSP:
			return "rsp";
		case UNW_X86_64_R8:
			return "r8";
		case UNW_X86_64_R9:
			return "r9";
		case UNW_X86_64_R10:
			return "r10";
		case UNW_X86_64_R11:
			return "r11";
		case UNW_X86_64_R12:
			return "r12";
		case UNW_X86_64_R13:
			return "r13";
		case UNW_X86_64_R14:
			return "r14";
		case UNW_X86_64_R15:
			return "r15";
		default:
			return "unknown register";
	}
}

double Registers_x86_64::getFloatRegister(int num) const
{
	ABORT("no x86_64 float registers");
}

void Registers_x86_64::setFloatRegister(int num, double value)
{
	ABORT("no x86_64 float registers");
}

inline v128 Registers_x86_64::getVectorRegister(int num) const
{
	ABORT("no x86_64 vector registers");
}

inline void Registers_x86_64::setVectorRegister(int num, v128 value)
{
	ABORT("no x86_64 vector registers");
}


} // namespace lldb_private 



#endif // __REGISTERS_HPP__




