//== ProgramState_Fwd.h - Incomplete declarations of ProgramState -*- C++ -*--=/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PROGRAMSTATE_FWD_H
#define LLVM_CLANG_PROGRAMSTATE_FWD_H

namespace clang {
namespace ento {
  class ProgramState;
  class ProgramStateManager;
  typedef const ProgramState* ProgramStateRef;
}
}

#endif

