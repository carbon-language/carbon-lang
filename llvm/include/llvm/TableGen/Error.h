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
  SmallVector<SMLoc, 4> Locs;
  std::string Message;
public:
  TGError(ArrayRef<SMLoc> locs, const std::string &message)
    : Locs(locs.begin(), locs.end()), Message(message) {}

  ArrayRef<SMLoc> getLoc() const { return Locs; }
  const std::string &getMessage() const { return Message; }
};

void PrintWarning(ArrayRef<SMLoc> WarningLoc, const Twine &Msg);
void PrintWarning(const char *Loc, const Twine &Msg);
void PrintWarning(const Twine &Msg);
void PrintWarning(const TGError &Warning);

void PrintError(ArrayRef<SMLoc> ErrorLoc, const Twine &Msg);
void PrintError(const char *Loc, const Twine &Msg);
void PrintError(const Twine &Msg);
void PrintError(const TGError &Error);

LLVM_ATTRIBUTE_NORETURN void PrintFatalError(const std::string &Msg);
LLVM_ATTRIBUTE_NORETURN void PrintFatalError(ArrayRef<SMLoc> ErrorLoc,
                                             const std::string &Msg);

extern SourceMgr SrcMgr;


} // end namespace "llvm"

#endif
