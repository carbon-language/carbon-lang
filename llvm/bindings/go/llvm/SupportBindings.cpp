//===- SupportBindings.cpp - Additional bindings for support --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines additional C bindings for the support component.
//
//===----------------------------------------------------------------------===//

#include "SupportBindings.h"
#include "llvm/Support/DynamicLibrary.h"
#include <stdlib.h>
#include <string.h>

void LLVMLoadLibraryPermanently2(const char *Filename, char **ErrMsg) {
  std::string ErrMsgStr;
  if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Filename, &ErrMsgStr)) {
    *ErrMsg = static_cast<char *>(malloc(ErrMsgStr.size() + 1));
    memcpy(static_cast<void *>(*ErrMsg),
           static_cast<const void *>(ErrMsgStr.c_str()), ErrMsgStr.size() + 1);
  }
}
