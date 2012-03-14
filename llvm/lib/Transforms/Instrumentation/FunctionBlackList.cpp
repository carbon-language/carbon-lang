//===-- FunctionBlackList.cpp - blacklist of functions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer 
// or ThreadSanitizer) to avoid instrumenting some functions based on
// user-supplied blacklist.
//
//===----------------------------------------------------------------------===//

#include "FunctionBlackList.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Function.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

namespace llvm {

FunctionBlackList::FunctionBlackList(const std::string &Path) {
  Functions = NULL;
  const char *kFunPrefix = "fun:";
  if (!Path.size()) return;
  std::string Fun;

  OwningPtr<MemoryBuffer> File;
  if (error_code EC = MemoryBuffer::getFile(Path.c_str(), File)) {
    report_fatal_error("Can't open blacklist file " + Path + ": " +
                       EC.message());
  }
  MemoryBuffer *Buff = File.take();
  const char *Data = Buff->getBufferStart();
  size_t DataLen = Buff->getBufferSize();
  SmallVector<StringRef, 16> Lines;
  SplitString(StringRef(Data, DataLen), Lines, "\n\r");
  for (size_t i = 0, numLines = Lines.size(); i < numLines; i++) {
    if (Lines[i].startswith(kFunPrefix)) {
      std::string ThisFunc = Lines[i].substr(strlen(kFunPrefix));
      std::string ThisFuncRE;
      // add ThisFunc replacing * with .*
      for (size_t j = 0, n = ThisFunc.size(); j < n; j++) {
        if (ThisFunc[j] == '*')
          ThisFuncRE += '.';
        ThisFuncRE += ThisFunc[j];
      }
      // Check that the regexp is valid.
      Regex CheckRE(ThisFuncRE);
      std::string Error;
      if (!CheckRE.isValid(Error))
        report_fatal_error("malformed blacklist regex: " + ThisFunc +
                           ": " + Error);
      // Append to the final regexp.
      if (Fun.size())
        Fun += "|";
      Fun += ThisFuncRE;
    }
  }
  if (Fun.size()) {
    Functions = new Regex(Fun);
  }
}

bool FunctionBlackList::isIn(const Function &F) {
  if (Functions) {
    bool Res = Functions->match(F.getName());
    return Res;
  }
  return false;
}

}  // namespace llvm
