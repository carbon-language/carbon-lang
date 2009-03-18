//===--- Compilation.cpp - Compilation Task Implementation --------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/ToolChain.h"

using namespace clang::driver;

Compilation::Compilation(ToolChain &_DefaultToolChain,
                         ArgList *_Args) 
  : DefaultToolChain(_DefaultToolChain), Args(_Args) {
}

Compilation::~Compilation() {  
  delete Args;
  
  // Free any derived arg lists.
  for (llvm::DenseMap<const ToolChain*, ArgList*>::iterator 
         it = TCArgs.begin(), ie = TCArgs.end(); it != ie; ++it) {
    ArgList *A = it->second;
    if (A != Args)
      delete Args;
  }

  // Free the actions, if built.
  for (ActionList::iterator it = Actions.begin(), ie = Actions.end(); 
       it != ie; ++it)
    delete *it;
}

const ArgList &Compilation::getArgsForToolChain(const ToolChain *TC) {
  if (!TC)
    TC = &DefaultToolChain;

  ArgList *&Entry = TCArgs[TC];
  if (!Entry)
    Entry = TC->TranslateArgs(*Args);

  return *Entry;
}

int Compilation::Execute() const {
  return 0;
}
