//===-- llvm/Bytecode/Primitives.h - Bytecode file format prims -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header defines some basic functions for reading and writing basic 
// primitive types to a bytecode stream.
//
// Using the routines defined in this file does not require linking to any 
// libraries, as all of the services are small self contained units that are to
// be inlined as necessary.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_PRIMITIVES_H
#define LLVM_BYTECODE_PRIMITIVES_H

#include "Support/DataTypes.h"
#include <string>
#include <deque>

namespace llvm {

//===----------------------------------------------------------------------===//
//                             Reading Primitives
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
//                             Writing Primitives
//===----------------------------------------------------------------------===//

// output - If a position is specified, it must be in the valid portion of the
// string... note that this should be inlined always so only the relevant IF 
// body should be included...
//
static inline void output(unsigned i, std::deque<unsigned char> &Out,
                          int pos = -1) {
#ifdef ENDIAN_LITTLE
  if (pos == -1) 
    Out.insert(Out.end(), (unsigned char*)&i, (unsigned char*)&i+4);
  else
    // This cannot use block copy because deques are not guaranteed contiguous!
    std::copy((unsigned char*)&i, 4+(unsigned char*)&i, Out.begin()+pos);
#else
  if (pos == -1) { // Be endian clean, little endian is our friend
    Out.push_back((unsigned char)i); 
    Out.push_back((unsigned char)(i >> 8));
    Out.push_back((unsigned char)(i >> 16));
    Out.push_back((unsigned char)(i >> 24));
  } else {
    Out[pos  ] = (unsigned char)i;
    Out[pos+1] = (unsigned char)(i >> 8);
    Out[pos+2] = (unsigned char)(i >> 16);
    Out[pos+3] = (unsigned char)(i >> 24);
  }
#endif
}

static inline void output(int i, std::deque<unsigned char> &Out) {
  output((unsigned)i, Out);
}

// output_vbr - Output an unsigned value, by using the least number of bytes
// possible.  This is useful because many of our "infinite" values are really
// very small most of the time... but can be large a few times...
//
// Data format used:  If you read a byte with the night bit set, use the low 
// seven bits as data and then read another byte...
//
// Note that using this may cause the output buffer to become unaligned...
//
static inline void output_vbr(uint64_t i, std::deque<unsigned char> &out) {
  while (1) {
    if (i < 0x80) { // done?
      out.push_back((unsigned char)i);   // We know the high bit is clear...
      return;
    }
    
    // Nope, we are bigger than a character, output the next 7 bits and set the
    // high bit to say that there is more coming...
    out.push_back(0x80 | (i & 0x7F));
    i >>= 7;  // Shift out 7 bits now...
  }
}

static inline void output_vbr(unsigned i, std::deque<unsigned char> &out) {
  while (1) {
    if (i < 0x80) { // done?
      out.push_back((unsigned char)i);   // We know the high bit is clear...
      return;
    }
    
    // Nope, we are bigger than a character, output the next 7 bits and set the
    // high bit to say that there is more coming...
    out.push_back(0x80 | (i & 0x7F));
    i >>= 7;  // Shift out 7 bits now...
  }
}

static inline void output_vbr(int64_t i, std::deque<unsigned char> &out) {
  if (i < 0) 
    output_vbr(((uint64_t)(-i) << 1) | 1, out); // Set low order sign bit...
  else
    output_vbr((uint64_t)i << 1, out);          // Low order bit is clear.
}


static inline void output_vbr(int i, std::deque<unsigned char> &out) {
  if (i < 0) 
    output_vbr(((unsigned)(-i) << 1) | 1, out); // Set low order sign bit...
  else
    output_vbr((unsigned)i << 1, out);          // Low order bit is clear.
}

// align32 - emit the minimal number of bytes that will bring us to 32 bit 
// alignment...
//
static inline void align32(std::deque<unsigned char> &Out) {
  int NumPads = (4-(Out.size() & 3)) & 3; // Bytes to get padding to 32 bits
  while (NumPads--) Out.push_back((unsigned char)0xAB);
}

static inline void output(const std::string &s, std::deque<unsigned char> &Out, 
                          bool Aligned = true) {
  unsigned Len = s.length();
  output_vbr(Len, Out);             // Strings may have an arbitrary length...
  Out.insert(Out.end(), s.begin(), s.end());

  if (Aligned)
    align32(Out);                   // Make sure we are now aligned...
}

static inline void output_data(void *Ptr, void *End,
                               std::deque<unsigned char> &Out,
                               bool Align = false) {
#ifdef ENDIAN_LITTLE
  Out.insert(Out.end(), (unsigned char*)Ptr, (unsigned char*)End);
#else
  unsigned char *E = (unsigned char *)End;
  while (Ptr != E)
    Out.push_back(*--E);
#endif

  if (Align) align32(Out);
}

} // End llvm namespace

#endif
