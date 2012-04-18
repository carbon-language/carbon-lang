//===- Error.cpp - tblgen error handling helper routines --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains error handling helper routines to pretty-print diagnostic
// messages from tblgen.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

SourceMgr SrcMgr;

void PrintWarning(SMLoc WarningLoc, const Twine &Msg) {
  SrcMgr.PrintMessage(WarningLoc, SourceMgr::DK_Warning, Msg);
}

void PrintWarning(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), SourceMgr::DK_Warning, Msg);
}

void PrintWarning(const Twine &Msg) {
  errs() << "error:" << Msg << "\n";
}

void PrintWarning(const TGError &Warning) {
  PrintWarning(Warning.getLoc(), Warning.getMessage());
}

void PrintError(SMLoc ErrorLoc, const Twine &Msg) {
  SrcMgr.PrintMessage(ErrorLoc, SourceMgr::DK_Error, Msg);
}

void PrintError(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), SourceMgr::DK_Error, Msg);
}

void PrintError(const Twine &Msg) {
  errs() << "error:" << Msg << "\n";
}

void PrintError(const TGError &Error) {
  PrintError(Error.getLoc(), Error.getMessage());
}

} // end namespace llvm
