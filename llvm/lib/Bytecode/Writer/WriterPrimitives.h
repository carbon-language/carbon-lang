//===-- WriterPrimitives.h - Bytecode writer file format prims --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header defines some basic functions for writing basic primitive types to
// a bytecode stream.
//
//===----------------------------------------------------------------------===//

#ifndef WRITERPRIMITIVES_H
#define WRITERPRIMITIVES_H

#include "Support/DataTypes.h"
#include <string>
#include <deque>

namespace llvm {

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
