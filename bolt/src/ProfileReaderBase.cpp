//===-- ProfileReaderBase.cpp - Base class for profile readers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Interface to be implemented by all profile readers.
//
//===----------------------------------------------------------------------===//

#include "ProfileReaderBase.h"
#include "BinaryFunction.h"

namespace llvm {
namespace bolt {

bool ProfileReaderBase::mayHaveProfileData(const BinaryFunction &BF) {
  return true;
}

}
}
