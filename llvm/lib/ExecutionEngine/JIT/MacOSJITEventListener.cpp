//===-- MacOSJITEventListener.cpp - Save symbol table for OSX perf tools --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITEventListener object that records JITted functions to
// a global __jitSymbolTable linked list.  Apple's performance tools use this to
// determine a symbol name and accurate code range for a PC value.  Because
// performance tools are generally asynchronous, the code below is written with
// the hope that it could be interrupted at any time and have useful answers.
// However, we don't go crazy with atomic operations, we just do a "reasonable
// effort".
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "macos-jit-event-listener"
#include "llvm/Function.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include <stddef.h>
using namespace llvm;

#ifdef __APPLE__
#define ENABLE_JIT_SYMBOL_TABLE 0
#endif

#if ENABLE_JIT_SYMBOL_TABLE

namespace {

/// JITSymbolEntry - Each function that is JIT compiled results in one of these
/// being added to an array of symbols.  This indicates the name of the function
/// as well as the address range it occupies.  This allows the client to map
/// from a PC value to the name of the function.
struct JITSymbolEntry {
  const char *FnName;   // FnName - a strdup'd string.
  void *FnStart;
  intptr_t FnSize;
};


struct JITSymbolTable {
  /// NextPtr - This forms a linked list of JitSymbolTable entries.  This
  /// pointer is not used right now, but might be used in the future.  Consider
  /// it reserved for future use.
  JITSymbolTable *NextPtr;
  
  /// Symbols - This is an array of JitSymbolEntry entries.  Only the first
  /// 'NumSymbols' symbols are valid.
  JITSymbolEntry *Symbols;
  
  /// NumSymbols - This indicates the number entries in the Symbols array that
  /// are valid.
  unsigned NumSymbols;
  
  /// NumAllocated - This indicates the amount of space we have in the Symbols
  /// array.  This is a private field that should not be read by external tools.
  unsigned NumAllocated;
};

class MacOSJITEventListener : public JITEventListener {
public:
  virtual void NotifyFunctionEmitted(const Function &F,
                                     void *FnStart, size_t FnSize,
                                     const EmittedFunctionDetails &Details);
  virtual void NotifyFreeingMachineCode(const Function &F, void *OldPtr);
};

}  // anonymous namespace.

// This is a public symbol so the performance tools can find it.
JITSymbolTable *__jitSymbolTable;

namespace llvm {
JITEventListener *createMacOSJITEventListener() {
  return new MacOSJITEventListener;
}
}

// Adds the just-emitted function to the symbol table.
void MacOSJITEventListener::NotifyFunctionEmitted(
    const Function &F, void *FnStart, size_t FnSize,
    const EmittedFunctionDetails &) {
  assert(F.hasName() && FnStart != 0 && "Bad symbol to add");
  JITSymbolTable **SymTabPtrPtr = 0;
  SymTabPtrPtr = &__jitSymbolTable;

  // If this is the first entry in the symbol table, add the JITSymbolTable
  // index.
  if (*SymTabPtrPtr == 0) {
    JITSymbolTable *New = new JITSymbolTable();
    New->NextPtr = 0;
    New->Symbols = 0;
    New->NumSymbols = 0;
    New->NumAllocated = 0;
    *SymTabPtrPtr = New;
  }

  JITSymbolTable *SymTabPtr = *SymTabPtrPtr;

  // If we have space in the table, reallocate the table.
  if (SymTabPtr->NumSymbols >= SymTabPtr->NumAllocated) {
    // If we don't have space, reallocate the table.
    unsigned NewSize = std::max(64U, SymTabPtr->NumAllocated*2);
    JITSymbolEntry *NewSymbols = new JITSymbolEntry[NewSize];
    JITSymbolEntry *OldSymbols = SymTabPtr->Symbols;

    // Copy the old entries over.
    memcpy(NewSymbols, OldSymbols, SymTabPtr->NumSymbols*sizeof(OldSymbols[0]));

    // Swap the new symbols in, delete the old ones.
    SymTabPtr->Symbols = NewSymbols;
    SymTabPtr->NumAllocated = NewSize;
    delete [] OldSymbols;
  }

  // Otherwise, we have enough space, just tack it onto the end of the array.
  JITSymbolEntry &Entry = SymTabPtr->Symbols[SymTabPtr->NumSymbols];
  Entry.FnName = strdup(F.getName().data());
  Entry.FnStart = FnStart;
  Entry.FnSize = FnSize;
  ++SymTabPtr->NumSymbols;
}

// Removes the to-be-deleted function from the symbol table.
void MacOSJITEventListener::NotifyFreeingMachineCode(
    const Function &, void *FnStart) {
  assert(FnStart && "Invalid function pointer");
  JITSymbolTable **SymTabPtrPtr = 0;
  SymTabPtrPtr = &__jitSymbolTable;

  JITSymbolTable *SymTabPtr = *SymTabPtrPtr;
  JITSymbolEntry *Symbols = SymTabPtr->Symbols;

  // Scan the table to find its index.  The table is not sorted, so do a linear
  // scan.
  unsigned Index;
  for (Index = 0; Symbols[Index].FnStart != FnStart; ++Index)
    assert(Index != SymTabPtr->NumSymbols && "Didn't find function!");

  // Once we have an index, we know to nuke this entry, overwrite it with the
  // entry at the end of the array, making the last entry redundant.
  const char *OldName = Symbols[Index].FnName;
  Symbols[Index] = Symbols[SymTabPtr->NumSymbols-1];
  free((void*)OldName);

  // Drop the number of symbols in the table.
  --SymTabPtr->NumSymbols;

  // Finally, if we deleted the final symbol, deallocate the table itself.
  if (SymTabPtr->NumSymbols != 0)
    return;

  *SymTabPtrPtr = 0;
  delete [] Symbols;
  delete SymTabPtr;
}

#else  // !ENABLE_JIT_SYMBOL_TABLE

namespace llvm {
// By defining this to return NULL, we can let clients call it unconditionally,
// even if they aren't on an Apple system.
JITEventListener *createMacOSJITEventListener() {
  return NULL;
}
}  // namespace llvm

#endif  // ENABLE_JIT_SYMBOL_TABLE
