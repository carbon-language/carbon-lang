//===-- JITEventListener.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITEventListener.h"

using namespace llvm;

// Out-of-line definition of the virtual destructor as this is the key function.
JITEventListener::~JITEventListener() {}
