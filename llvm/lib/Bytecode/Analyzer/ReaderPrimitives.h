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

  static inline unsigned read(const unsigned char *&Buf,
                              const unsigned char *EndBuf) {
    if (Buf+4 > EndBuf) throw std::string("Ran out of data!");
    Buf += 4;
    return Buf[-4] | (Buf[-3] << 8) | (Buf[-2] << 16) | (Buf[-1] << 24);
  }


  // read_vbr - Read an unsigned integer encoded in variable bitrate format.
  //
  static inline unsigned read_vbr_uint(const unsigned char *&Buf, 
                                       const unsigned char *EndBuf) {
    unsigned Shift = 0;
    unsigned Result = 0;
    
    do {
      if (Buf == EndBuf) throw std::string("Ran out of data!");
      Result |= (unsigned)((*Buf++) & 0x7F) << Shift;
      Shift += 7;
    } while (Buf[-1] & 0x80);
    return Result;
  }
  
  static inline uint64_t read_vbr_uint64(const unsigned char *&Buf, 
                                         const unsigned char *EndBuf) {
    unsigned Shift = 0;
    uint64_t Result = 0;
    
    do {
      if (Buf == EndBuf) throw std::string("Ran out of data!");
      Result |= (uint64_t)((*Buf++) & 0x7F) << Shift;
      Shift += 7;
    } while (Buf[-1] & 0x80);
    return Result;
  }
  
  static inline int64_t read_vbr_int64(const unsigned char *&Buf, 
                                       const unsigned char *EndBuf) {
    uint64_t R = read_vbr_uint64(Buf, EndBuf);
    if (R & 1) {
      if (R != 1)
        return -(int64_t)(R >> 1);
      else   // There is no such thing as -0 with integers.  "-0" really means
             // 0x8000000000000000.
        return 1LL << 63;
    } else
      return  (int64_t)(R >> 1);
  }
  
  // align32 - Round up to multiple of 32 bits...
  static inline void align32(const unsigned char *&Buf, 
                             const unsigned char *EndBuf) {
    Buf = (const unsigned char *)((unsigned long)(Buf+3) & (~3UL));
    if (Buf > EndBuf) throw std::string("Ran out of data!");
  }
  
  static inline std::string read_str(const unsigned char *&Buf,
                                     const unsigned char *EndBuf) {
    unsigned Size = read_vbr_uint(Buf, EndBuf);
    const unsigned char *OldBuf = Buf;
    Buf += Size;
    if (Buf > EndBuf)             // Size invalid?
      throw std::string("Ran out of data reading a string!");
    return std::string((char*)OldBuf, Size);
  }
  
  static inline void input_data(const unsigned char *&Buf,
                                const unsigned char *EndBuf, 
                                void *Ptr, void *End) {
    unsigned char *Start = (unsigned char *)Ptr;
    unsigned Amount = (unsigned char *)End - Start;
    if (Buf+Amount > EndBuf) throw std::string("Ran out of data!");
    std::copy(Buf, Buf+Amount, Start);
    Buf += Amount;
  }
  
} // End llvm namespace

#endif
