//===-- GCMetadataPrinter.cpp - Garbage collection infrastructure ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the abstract base class GCMetadataPrinter.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadataPrinter.h"
using namespace llvm;

GCMetadataPrinter::GCMetadataPrinter() {}

GCMetadataPrinter::~GCMetadataPrinter() {}

// Stamp out the registry of GCMetadataPrinter objects so that pulgins can load
// new Strategies.  
DEFINE_REGISTRY(GCMetadataPrinter)
