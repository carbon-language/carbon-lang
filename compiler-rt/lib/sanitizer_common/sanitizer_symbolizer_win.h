//===-- sanitizer_symbolizer_win.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Header file for the Windows symbolizer tool.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SYMBOLIZER_WIN_H
#define SANITIZER_SYMBOLIZER_WIN_H

#include "sanitizer_symbolizer_internal.h"

namespace __sanitizer {

class WinSymbolizerTool : public SymbolizerTool {
 public:
  bool SymbolizePC(uptr addr, SymbolizedStack *stack) override;
  bool SymbolizeData(uptr addr, DataInfo *info) override {
    return false;
  }
  const char *Demangle(const char *name) override;
};

}  // namespace __sanitizer

#endif  // SANITIZER_SYMBOLIZER_WIN_H
