//==- ProgramPoint.cpp - Program Points for Path-Sensitive Analysis -*- C++ -*-/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface ProgramPoint, which identifies a
//  distinct location in a function.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/ProgramPoint.h"

using namespace clang;

ProgramPointTag::~ProgramPointTag() {}

SimpleProgramPointTag::SimpleProgramPointTag(StringRef description)
  : desc(description) {}

StringRef SimpleProgramPointTag::getTagDescription() const {
  return desc;
}
