//===-- ReaderPrimitives.h - Bytecode file format reading prims -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header defines some basic functions for reading basic primitive types
// from a bytecode stream.
//
//===----------------------------------------------------------------------===//

#ifndef READERPRIMITIVES_H
#define READERPRIMITIVES_H

#include "Support/DataTypes.h"
#include <string>

namespace llvm {

static inline bool read(const unsigned char *&Buf, const unsigned char *EndBuf,
                        unsigned &Result) {
  if (Buf+4 > EndBuf) return true;
#ifdef ENDIAN_LITTLE
  Result = *(unsigned*)Buf;
#else
  Result = Buf[0] | (Buf[1] << 8) | (Buf[2] << 16) | (Buf[3] << 24);
#endif
  Buf += 4;
  return false;
}

static inline bool read(const unsigned char *&Buf, const unsigned char *EndBuf,
                        uint64_t &Result) {
  if (Buf+8 > EndBuf) return true;

#ifdef ENDIAN_LITTLE
  Result = *(uint64_t*)Buf;
#else
  Result = Buf[0] | (Buf[1] << 8) | (Buf[2] << 16) | (Buf[3] << 24) |
    ((uint64_t)(Buf[4] | (Buf[5] << 8) | (Buf[6] << 16) | (Buf[7] << 24)) <<32);
#endif
  Buf += 8;
  return false;
}

static inline bool read(const unsigned char *&Buf, const unsigned char *EndBuf,
                        int &Result) {
  return read(Buf, EndBuf, (unsigned &)Result);
}

static inline bool read(const unsigned char *&Buf, const unsigned char *EndBuf,
                        int64_t &Result) {
  return read(Buf, EndBuf, (uint64_t &)Result);
}


// read_vbr - Read an unsigned integer encoded in variable bitrate format.
//
static inline bool read_vbr(const unsigned char *&Buf, 
                            const unsigned char *EndBuf, unsigned &Result) {
  unsigned Shift = Result = 0;

  do {
    Result |= (unsigned)((*Buf++) & 0x7F) << Shift;
    Shift += 7;
  } while (Buf[-1] & 0x80 && Buf < EndBuf);

  return Buf > EndBuf;
}

static inline bool read_vbr(const unsigned char *&Buf, 
                            const unsigned char *EndBuf, uint64_t &Result) {
  unsigned Shift = 0; Result = 0;

  do {
    Result |= (uint64_t)((*Buf++) & 0x7F) << Shift;
    Shift += 7;
  } while (Buf[-1] & 0x80 && Buf < EndBuf);
  return Buf > EndBuf;
}

// read_vbr (signed) - Read a signed number stored in sign-magnitude format
static inline bool read_vbr(const unsigned char *&Buf, 
                            const unsigned char *EndBuf, int &Result) {
  unsigned R;
  if (read_vbr(Buf, EndBuf, R)) return true;
  if (R & 1)
    Result = -(int)(R >> 1);
  else
    Result =  (int)(R >> 1);
  
  return false;
}


static inline bool read_vbr(const unsigned char *&Buf, 
                            const unsigned char *EndBuf, int64_t &Result) {
  uint64_t R;
  if (read_vbr(Buf, EndBuf, R)) return true;
  if (R & 1)
    Result = -(int64_t)(R >> 1);
  else
    Result =  (int64_t)(R >> 1);
  
  return false;
}

// align32 - Round up to multiple of 32 bits...
static inline bool align32(const unsigned char *&Buf, 
                           const unsigned char *EndBuf) {
  Buf = (const unsigned char *)((unsigned long)(Buf+3) & (~3UL));
  return Buf > EndBuf;
}

static inline bool read(const unsigned char *&Buf, const unsigned char *EndBuf, 
                        std::string &Result, bool Aligned = true) {
  unsigned Size;
  if (read_vbr(Buf, EndBuf, Size)) return true;   // Failure reading size?
  if (Buf+Size > EndBuf) return true;             // Size invalid?

  Result = std::string((char*)Buf, Size);
  Buf += Size;

  if (Aligned)        // If we should stay aligned do so...
    if (align32(Buf, EndBuf)) return true;        // Failure aligning?

  return false;
}

static inline bool input_data(const unsigned char *&Buf,
                              const unsigned char *EndBuf, 
                              void *Ptr, void *End, bool Align = false) {
  unsigned char *Start = (unsigned char *)Ptr;
  unsigned Amount = (unsigned char *)End - Start;
  if (Buf+Amount > EndBuf) return true;
#ifdef ENDIAN_LITTLE
  std::copy(Buf, Buf+Amount, Start);
  Buf += Amount;
#else
  unsigned char *E = (unsigned char *)End;
  while (Ptr != E)
    *--E = *Buf++;
#endif

  if (Align) return align32(Buf, EndBuf);
  return false;
}

} // End llvm namespace

#endif
