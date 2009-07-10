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
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Support/Debug.h"
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
    DOUT << "Failed to connect to OProfile agent: " << err_str << "\n";
  } else {
    DOUT << "Connected to OProfile agent.\n";
  }
}

OProfileJITEventListener::~OProfileJITEventListener() {
  if (Agent != NULL) {
    if (op_close_agent(Agent) == -1) {
      const std::string err_str = sys::StrError();
      DOUT << "Failed to disconnect from OProfile agent: " << err_str << "\n";
    } else {
      DOUT << "Disconnected from OProfile agent.\n";
    }
  }
}

// Adds the just-emitted function to the symbol table.
void OProfileJITEventListener::NotifyFunctionEmitted(
    const Function &F, void *FnStart, size_t FnSize,
    const EmittedFunctionDetails &) {
  const char *const FnName = F.getNameStart();
  assert(FnName != 0 && FnStart != 0 && "Bad symbol to add");
  if (op_write_native_code(Agent, FnName,
                           reinterpret_cast<uint64_t>(FnStart),
                           FnStart, FnSize) == -1) {
    DOUT << "Failed to tell OProfile about native function " << FnName
         << " at [" << FnStart << "-" << ((char*)FnStart + FnSize) << "]\n";
  }
}

// Removes the to-be-deleted function from the symbol table.
void OProfileJITEventListener::NotifyFreeingMachineCode(
    const Function &F, void *FnStart) {
  assert(FnStart && "Invalid function pointer");
  if (op_unload_native_code(Agent, reinterpret_cast<uint64_t>(FnStart)) == -1) {
    DOUT << "Failed to tell OProfile about unload of native function "
         << F.getName() << " at " << FnStart << "\n";
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
