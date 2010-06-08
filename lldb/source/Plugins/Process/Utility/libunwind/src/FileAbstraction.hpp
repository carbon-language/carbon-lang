/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- FileAbstraction.hpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __FILE_ABSTRACTION__
#define __FILE_ABSTRACTION__


#include <stdint.h>
#include <string.h>
#include <libkern/OSByteOrder.h>

#ifdef __OPTIMIZE__
#define INLINE	__attribute__((always_inline))
#else
#define INLINE
#endif

//
// This abstraction layer is for use with file formats that have 64-bit/32-bit and Big-Endian/Little-Endian variants
//
// For example: to make a utility that handles 32-bit little enidan files use:  Pointer32<LittleEndian>
//
//
//		get16()			read a 16-bit number from an E endian struct
//		set16()			write a 16-bit number to an E endian struct
//		get32()			read a 32-bit number from an E endian struct
//		set32()			write a 32-bit number to an E endian struct
//		get64()			read a 64-bit number from an E endian struct
//		set64()			write a 64-bit number to an E endian struct
//
//		getBits()		read a bit field from an E endian struct (bitCount=number of bits in field, firstBit=bit index of field)
//		setBits()		write a bit field to an E endian struct (bitCount=number of bits in field, firstBit=bit index of field)
//
//		getBitsRaw()	read a bit field from a struct with native endianness
//		setBitsRaw()	write a bit field from a struct with native endianness
//

class BigEndian
{
public:
	static uint16_t	get16(const uint16_t& from)				INLINE { return OSReadBigInt16(&from, 0); }
	static void		set16(uint16_t& into, uint16_t value)	INLINE { OSWriteBigInt16(&into, 0, value); }
	
	static uint32_t	get32(const uint32_t& from)				INLINE { return OSReadBigInt32(&from, 0); }
	static void		set32(uint32_t& into, uint32_t value)	INLINE { OSWriteBigInt32(&into, 0, value); }
	
	static uint64_t get64(const uint64_t& from)				INLINE { return OSReadBigInt64(&from, 0); }
	static void		set64(uint64_t& into, uint64_t value)	INLINE { OSWriteBigInt64(&into, 0, value); }
	
	static uint32_t	getBits(const uint32_t& from, 
						uint8_t firstBit, uint8_t bitCount)	INLINE { return getBitsRaw(get32(from), firstBit, bitCount); }
	static void		setBits(uint32_t& into, uint32_t value,
						uint8_t firstBit, uint8_t bitCount)	INLINE { uint32_t temp = get32(into); setBitsRaw(temp, value, firstBit, bitCount); set32(into, temp); }

	static uint32_t	getBitsRaw(const uint32_t& from, 
						uint8_t firstBit, uint8_t bitCount)	INLINE { return ((from >> (32-firstBit-bitCount)) & ((1<<bitCount)-1)); }
	static void		setBitsRaw(uint32_t& into, uint32_t value,
						uint8_t firstBit, uint8_t bitCount)	INLINE { uint32_t temp = into; 
																							const uint32_t mask = ((1<<bitCount)-1); 
																							temp &= ~(mask << (32-firstBit-bitCount)); 
																							temp |= ((value & mask) << (32-firstBit-bitCount)); 
																							into = temp; }
	enum { little_endian = 0 };
};


class LittleEndian
{
public:
	static uint16_t	get16(const uint16_t& from)				INLINE { return OSReadLittleInt16(&from, 0); }
	static void		set16(uint16_t& into, uint16_t value)	INLINE { OSWriteLittleInt16(&into, 0, value); }
	
	static uint32_t	get32(const uint32_t& from)				INLINE { return OSReadLittleInt32(&from, 0); }
	static void		set32(uint32_t& into, uint32_t value)	INLINE { OSWriteLittleInt32(&into, 0, value); }
	
	static uint64_t get64(const uint64_t& from)				INLINE { return OSReadLittleInt64(&from, 0); }
	static void		set64(uint64_t& into, uint64_t value)	INLINE { OSWriteLittleInt64(&into, 0, value); }

	static uint32_t	getBits(const uint32_t& from,
						uint8_t firstBit, uint8_t bitCount)	INLINE { return getBitsRaw(get32(from), firstBit, bitCount); }
	static void		setBits(uint32_t& into, uint32_t value,
						uint8_t firstBit, uint8_t bitCount)	INLINE { uint32_t temp = get32(into); setBitsRaw(temp, value, firstBit, bitCount); set32(into, temp); }

	static uint32_t	getBitsRaw(const uint32_t& from,
						uint8_t firstBit, uint8_t bitCount)	INLINE { return ((from >> firstBit) & ((1<<bitCount)-1)); }
	static void		setBitsRaw(uint32_t& into, uint32_t value,
						uint8_t firstBit, uint8_t bitCount)	INLINE {  uint32_t temp = into; 
																							const uint32_t mask = ((1<<bitCount)-1); 
																							temp &= ~(mask << firstBit); 
																							temp |= ((value & mask) << firstBit); 
																							into = temp; }
	enum { little_endian = 1 };
};


template <typename _E>
class Pointer32
{
public:
	typedef uint32_t	uint_t;
	typedef int32_t	    int_t;
	typedef _E			E;
	
	static uint64_t	getP(const uint_t& from)				INLINE { return _E::get32(from); }
	static void		setP(uint_t& into, uint64_t value)		INLINE { _E::set32(into, value); }
};


template <typename _E>
class Pointer64
{
public:
	typedef uint64_t	uint_t;
	typedef int64_t	    int_t;
	typedef _E			E;
	
	static uint64_t	getP(const uint_t& from)				INLINE { return _E::get64(from); }
	static void		setP(uint_t& into, uint64_t value)		INLINE { _E::set64(into, value); }
};






#endif // __FILE_ABSTRACTION__


