//===-- DIContext.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DIContext.h"
#include "DWARFContext.h"
using namespace llvm;

DIContext::~DIContext() {}

DIContext *DIContext::getDWARFContext(const object::ObjectFile &Obj) {
  return new DWARFContextInMemory(Obj);
}
