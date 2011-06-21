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

#include "Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

SourceMgr SrcMgr;

void PrintError(SMLoc ErrorLoc, const Twine &Msg) {
  SrcMgr.PrintMessage(ErrorLoc, Msg, "error");
}

void PrintError(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), Msg, "error");
}

void PrintError(const Twine &Msg) {
  errs() << "error:" << Msg << "\n";
}

void PrintError(const TGError &Error) {
  PrintError(Error.getLoc(), Error.getMessage());
}

} // end namespace llvm
