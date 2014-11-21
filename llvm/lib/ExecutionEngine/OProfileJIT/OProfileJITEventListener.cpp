//===-- OProfileJITEventListener.cpp - Tell OProfile about JITted code ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITEventListener object that uses OProfileWrapper to tell
// oprofile about JITted functions, including source line information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/JITEventListener.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ExecutionEngine/OProfileWrapper.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Errno.h"
#include "EventListenerCommon.h"

#include <dirent.h>
#include <fcntl.h>

using namespace llvm;
using namespace llvm::jitprofiling;

#define DEBUG_TYPE "oprofile-jit-event-listener"

namespace {

class OProfileJITEventListener : public JITEventListener {
  OProfileWrapper& Wrapper;

  void initialize();

public:
  OProfileJITEventListener(OProfileWrapper& LibraryWrapper)
  : Wrapper(LibraryWrapper) {
    initialize();
  }

  ~OProfileJITEventListener();

  virtual void NotifyObjectEmitted(const ObjectImage &Obj);

  virtual void NotifyFreeingObject(const ObjectImage &Obj);
};

void OProfileJITEventListener::initialize() {
  if (!Wrapper.op_open_agent()) {
    const std::string err_str = sys::StrError();
    DEBUG(dbgs() << "Failed to connect to OProfile agent: " << err_str << "\n");
  } else {
    DEBUG(dbgs() << "Connected to OProfile agent.\n");
  }
}

OProfileJITEventListener::~OProfileJITEventListener() {
  if (Wrapper.isAgentAvailable()) {
    if (Wrapper.op_close_agent() == -1) {
      const std::string err_str = sys::StrError();
      DEBUG(dbgs() << "Failed to disconnect from OProfile agent: "
                   << err_str << "\n");
    } else {
      DEBUG(dbgs() << "Disconnected from OProfile agent.\n");
    }
  }
}

void OProfileJITEventListener::NotifyObjectEmitted(const ObjectImage &Obj) {
  if (!Wrapper.isAgentAvailable()) {
    return;
  }

  // Use symbol info to iterate functions in the object.
  for (object::symbol_iterator I = Obj.begin_symbols(), E = Obj.end_symbols();
       I != E; ++I) {
    object::SymbolRef::Type SymType;
    if (I->getType(SymType)) continue;
    if (SymType == object::SymbolRef::ST_Function) {
      StringRef  Name;
      uint64_t   Addr;
      uint64_t   Size;
      if (I->getName(Name)) continue;
      if (I->getAddress(Addr)) continue;
      if (I->getSize(Size)) continue;

      if (Wrapper.op_write_native_code(Name.data(), Addr, (void*)Addr, Size)
                        == -1) {
        DEBUG(dbgs() << "Failed to tell OProfile about native function "
          << Name << " at ["
          << (void*)Addr << "-" << ((char*)Addr + Size) << "]\n");
        continue;
      }
      // TODO: support line number info (similar to IntelJITEventListener.cpp)
    }
  }
}

void OProfileJITEventListener::NotifyFreeingObject(const ObjectImage &Obj) {
  if (!Wrapper.isAgentAvailable()) {
    return;
  }

  // Use symbol info to iterate functions in the object.
  for (object::symbol_iterator I = Obj.begin_symbols(), E = Obj.end_symbols();
       I != E; ++I) {
    object::SymbolRef::Type SymType;
    if (I->getType(SymType)) continue;
    if (SymType == object::SymbolRef::ST_Function) {
      uint64_t   Addr;
      if (I->getAddress(Addr)) continue;

      if (Wrapper.op_unload_native_code(Addr) == -1) {
        DEBUG(dbgs()
          << "Failed to tell OProfile about unload of native function at "
          << (void*)Addr << "\n");
        continue;
      }
    }
  }
}

}  // anonymous namespace.

namespace llvm {
JITEventListener *JITEventListener::createOProfileJITEventListener() {
  static std::unique_ptr<OProfileWrapper> JITProfilingWrapper(
      new OProfileWrapper);
  return new OProfileJITEventListener(*JITProfilingWrapper);
}

// for testing
JITEventListener *JITEventListener::createOProfileJITEventListener(
                                      OProfileWrapper* TestImpl) {
  return new OProfileJITEventListener(*TestImpl);
}

} // namespace llvm

