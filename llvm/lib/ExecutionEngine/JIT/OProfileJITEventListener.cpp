//===-- OProfileJITEventListener.cpp - Tell OProfile about JITted code ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITEventListener object that calls into OProfile to tell
// it about JITted functions.  For now, we only record function names and sizes,
// but eventually we'll also record line number information.
//
// See http://oprofile.sourceforge.net/doc/devel/jit-interface.html for the
// definition of the interface we're using.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "oprofile-jit-event-listener"
#include "llvm/Function.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Errno.h"
#include "llvm/Config/config.h"
#include <stddef.h>
using namespace llvm;

#if USE_OPROFILE

#include <opagent.h>

namespace {

class OProfileJITEventListener : public JITEventListener {
  op_agent_t Agent;
public:
  OProfileJITEventListener();
  ~OProfileJITEventListener();

  virtual void NotifyFunctionEmitted(const Function &F,
                                     void *FnStart, size_t FnSize,
                                     const EmittedFunctionDetails &Details);
  virtual void NotifyFreeingMachineCode(const Function &F, void *OldPtr);
};

OProfileJITEventListener::OProfileJITEventListener()
    : Agent(op_open_agent()) {
  if (Agent == NULL) {
    const std::string err_str = sys::StrError();
    DEBUG(errs() << "Failed to connect to OProfile agent: " << err_str << "\n");
  } else {
    DEBUG(errs() << "Connected to OProfile agent.\n");
  }
}

OProfileJITEventListener::~OProfileJITEventListener() {
  if (Agent != NULL) {
    if (op_close_agent(Agent) == -1) {
      const std::string err_str = sys::StrError();
      DEBUG(errs() << "Failed to disconnect from OProfile agent: "
                   << err_str << "\n");
    } else {
      DEBUG(errs() << "Disconnected from OProfile agent.\n");
    }
  }
}

class FilenameCache {
  // Holds the filename of each CompileUnit, so that we can pass the
  // pointer into oprofile.  These char*s are freed in the destructor.
  DenseMap<MDNode*, char*> Filenames;

 public:
  const char *getFilename(MDNode *CompileUnit) {
    char *&Filename = Filenames[CompileUnit];
    if (Filename == NULL) {
      DICompileUnit CU(CompileUnit);
      Filename = strdup(CU.getFilename());
    }
    return Filename;
  }
  ~FilenameCache() {
    for (DenseMap<MDNode*, char*>::iterator
             I = Filenames.begin(), E = Filenames.end(); I != E; ++I) {
      free(I->second);
    }
  }
};

static debug_line_info LineStartToOProfileFormat(
    const MachineFunction &MF, FilenameCache &Filenames,
    uintptr_t Address, DebugLoc Loc) {
  debug_line_info Result;
  Result.vma = Address;
  const DebugLocTuple &tuple = MF.getDebugLocTuple(Loc);
  Result.lineno = tuple.Line;
  Result.filename = Filenames.getFilename(tuple.CompileUnit);
  DEBUG(errs() << "Mapping " << reinterpret_cast<void*>(Result.vma) << " to "
               << Result.filename << ":" << Result.lineno << "\n");
  return Result;
}

// Adds the just-emitted function to the symbol table.
void OProfileJITEventListener::NotifyFunctionEmitted(
    const Function &F, void *FnStart, size_t FnSize,
    const EmittedFunctionDetails &Details) {
  assert(F.hasName() && FnStart != 0 && "Bad symbol to add");
  if (op_write_native_code(Agent, F.getName().data(),
                           reinterpret_cast<uint64_t>(FnStart),
                           FnStart, FnSize) == -1) {
    DEBUG(errs() << "Failed to tell OProfile about native function " 
          << F.getName() << " at [" 
          << FnStart << "-" << ((char*)FnStart + FnSize) << "]\n");
    return;
  }

  // Now we convert the line number information from the address/DebugLoc format
  // in Details to the address/filename/lineno format that OProfile expects.
  // OProfile 0.9.4 (and maybe later versions) has a bug that causes it to
  // ignore line numbers for addresses above 4G.
  FilenameCache Filenames;
  std::vector<debug_line_info> LineInfo;
  LineInfo.reserve(1 + Details.LineStarts.size());
  if (!Details.MF->getDefaultDebugLoc().isUnknown()) {
    LineInfo.push_back(LineStartToOProfileFormat(
        *Details.MF, Filenames,
        reinterpret_cast<uintptr_t>(FnStart),
        Details.MF->getDefaultDebugLoc()));
  }
  for (std::vector<EmittedFunctionDetails::LineStart>::const_iterator
           I = Details.LineStarts.begin(), E = Details.LineStarts.end();
       I != E; ++I) {
    LineInfo.push_back(LineStartToOProfileFormat(
        *Details.MF, Filenames, I->Address, I->Loc));
  }
  if (!LineInfo.empty()) {
    if (op_write_debug_line_info(Agent, FnStart,
                                 LineInfo.size(), &*LineInfo.begin()) == -1) {
      DEBUG(errs() 
            << "Failed to tell OProfile about line numbers for native function "
            << F.getName() << " at [" 
            << FnStart << "-" << ((char*)FnStart + FnSize) << "]\n");
    }
  }
}

// Removes the to-be-deleted function from the symbol table.
void OProfileJITEventListener::NotifyFreeingMachineCode(
    const Function &F, void *FnStart) {
  assert(FnStart && "Invalid function pointer");
  if (op_unload_native_code(Agent, reinterpret_cast<uint64_t>(FnStart)) == -1) {
    DEBUG(errs() << "Failed to tell OProfile about unload of native function "
                 << F.getName() << " at " << FnStart << "\n");
  }
}

}  // anonymous namespace.

namespace llvm {
JITEventListener *createOProfileJITEventListener() {
  return new OProfileJITEventListener;
}
}

#else  // USE_OPROFILE

namespace llvm {
// By defining this to return NULL, we can let clients call it unconditionally,
// even if they haven't configured with the OProfile libraries.
JITEventListener *createOProfileJITEventListener() {
  return NULL;
}
}  // namespace llvm

#endif  // USE_OPROFILE
