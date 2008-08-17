//===-- Collector.cpp - Garbage collection infrastructure -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements target- and collector-independent garbage collection
// infrastructure.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCStrategy.h"

using namespace llvm;

GCMetadataPrinter::GCMetadataPrinter() { }

GCMetadataPrinter::~GCMetadataPrinter() { }

void GCMetadataPrinter::beginAssembly(std::ostream &OS, AsmPrinter &AP,
                                      const TargetAsmInfo &TAI) {
  // Default is no action.
}

void GCMetadataPrinter::finishAssembly(std::ostream &OS, AsmPrinter &AP,
                                       const TargetAsmInfo &TAI) {
  // Default is no action.
}
