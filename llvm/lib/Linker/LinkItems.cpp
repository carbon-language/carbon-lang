//===- lib/Linker/LinkItems.cpp - Link LLVM objects and libraries ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bytecode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"

using namespace llvm;

// LinkItems - preserve link order for an arbitrary set of linkage items.
bool
Linker::LinkInItems(const ItemList& Items) {
  // For each linkage item ...
  for (ItemList::const_iterator I = Items.begin(), E = Items.end(); 
       I != E; ++I) {
    if (I->second) {
      // Link in the library suggested.
      if (LinkInLibrary(I->first))
        return true;
    } else {
      if (LinkInFile(sys::Path(I->first)))
        return true;
    }
  }

  // At this point we have processed all the link items provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  const Module::LibraryListType& DepLibs = Composite->getLibraries();
  for (Module::LibraryListType::const_iterator I = DepLibs.begin(), 
      E = DepLibs.end(); I != E; ++I) {
    if(LinkInLibrary(*I))
      return true;
  }

  return false;
}
