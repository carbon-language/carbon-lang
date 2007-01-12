//===-- ARMCommon.cpp - Define support functions for ARM --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//
#include "ARMCommon.h"

static inline unsigned rotateL(unsigned x, unsigned n){
  return ((x << n) | (x  >> (32 - n)));
}

static inline unsigned rotateR(unsigned x, unsigned n){
  return ((x >> n) | (x  << (32 - n)));
}

// finds the end position of largest sequence of zeros in binary representation
// of 'immediate'.
static int findLargestZeroSequence(unsigned immediate){
  int max_zero_pos = 0;
  int max_zero_length = 0;
  int zero_pos;
  int zero_length;
  int pos = 0;
  int end_pos;

  while ((immediate & 0x3) == 0) {
    immediate = rotateR(immediate, 2);
    pos+=2;
  }
  end_pos = pos+32;

  while (pos<end_pos){
    while (((immediate & 0x3) != 0)&&(pos<end_pos)) {
      immediate = rotateR(immediate, 2);
      pos+=2;
    }
    zero_pos = pos;
    while (((immediate & 0x3) == 0)&&(pos<end_pos)) {
      immediate = rotateR(immediate, 2);
      pos+=2;
    }
    zero_length = pos - zero_pos;
    if (zero_length > max_zero_length){
      max_zero_length = zero_length;
      max_zero_pos = zero_pos % 32;
    }

  }

  return (max_zero_pos + max_zero_length) % 32;
}

std::vector<unsigned> splitImmediate(unsigned immediate){
  std::vector<unsigned> immediatePieces;

  if (immediate == 0){
    immediatePieces.push_back(0);
  } else {
    int start_pos = findLargestZeroSequence(immediate);
    unsigned immediate_tmp = rotateR(immediate, start_pos);
    int pos = 0;
    while (pos < 32){
      while(((immediate_tmp&0x3) == 0)&&(pos<32)){
        immediate_tmp = rotateR(immediate_tmp,2);
        pos+=2;
      }
      if (pos < 32){
        immediatePieces.push_back(rotateL(immediate_tmp&0xFF,
                                          (start_pos + pos) % 32 ));
        immediate_tmp = rotateR(immediate_tmp,8);
        pos+=8;
      }
    }
  }
  return immediatePieces;
}
