//===-- CodeGen/AsmPrinter/DwarfTableException.cpp - Dwarf Exception Impl --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a simple implementation of DwarfException that just produces
// the exception table for use with SjLj.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
using namespace llvm;

DwarfSjLjException::DwarfSjLjException(AsmPrinter *A) : DwarfException(A) {
}

DwarfSjLjException::~DwarfSjLjException() {}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfSjLjException::EndModule() {
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void DwarfSjLjException::BeginFunction(const MachineFunction *MF) {
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfSjLjException::EndFunction() {
  // Record if this personality index uses a landing pad.
  bool HasLandingPad = !MMI->getLandingPads().empty();

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  if (HasLandingPad)
    EmitExceptionTable();
}
