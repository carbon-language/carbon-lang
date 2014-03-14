//===- llvm/Support/DiagnosticInfo.cpp - Diagnostic Definitions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the different classes involved in low level diagnostics.
//
// Diagnostics reporting is still done as part of the LLVMContext.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Atomic.h"
#include <string>

using namespace llvm;

int llvm::getNextAvailablePluginDiagnosticKind() {
  static sys::cas_flag PluginKindID = DK_FirstPluginKind;
  return (int)sys::AtomicIncrement(&PluginKindID);
}

DiagnosticInfoInlineAsm::DiagnosticInfoInlineAsm(const Instruction &I,
                                                 const Twine &MsgStr,
                                                 DiagnosticSeverity Severity)
    : DiagnosticInfo(DK_InlineAsm, Severity), LocCookie(0), MsgStr(MsgStr),
      Instr(&I) {
  if (const MDNode *SrcLoc = I.getMetadata("srcloc")) {
    if (SrcLoc->getNumOperands() != 0)
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(SrcLoc->getOperand(0)))
        LocCookie = CI->getZExtValue();
  }
}

void DiagnosticInfoInlineAsm::print(DiagnosticPrinter &DP) const {
  DP << getMsgStr();
  if (getLocCookie())
    DP << " at line " << getLocCookie();
}

void DiagnosticInfoStackSize::print(DiagnosticPrinter &DP) const {
  DP << "stack size limit exceeded (" << getStackSize() << ") in "
     << getFunction();
}

void DiagnosticInfoDebugMetadataVersion::print(DiagnosticPrinter &DP) const {
  DP << "ignoring debug info with an invalid version (" << getMetadataVersion()
     << ") in " << getModule();
}

void DiagnosticInfoSampleProfile::print(DiagnosticPrinter &DP) const {
  if (getFileName() && getLineNum() > 0)
    DP << getFileName() << ":" << getLineNum() << ": ";
  else if (getFileName())
    DP << getFileName() << ": ";
  DP << getMsg();
}
