//===- llvm/TableGen/Error.h - tblgen error handling helpers ----*- C++ -*-===//
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

#ifndef LLVM_TABLEGEN_ERROR_H
#define LLVM_TABLEGEN_ERROR_H

#include "llvm/Support/SourceMgr.h"

namespace llvm {

class TGError {
  SMLoc Loc;
  std::string Message;
public:
  TGError(SMLoc loc, const std::string &message) : Loc(loc), Message(message) {}

  SMLoc getLoc() const { return Loc; }
  const std::string &getMessage() const { return Message; }
};

void PrintError(SMLoc ErrorLoc, const Twine &Msg);
void PrintError(const char *Loc, const Twine &Msg);
void PrintError(const Twine &Msg);
void PrintError(const TGError &Error);


extern SourceMgr SrcMgr;


} // end namespace "llvm"

#endif
