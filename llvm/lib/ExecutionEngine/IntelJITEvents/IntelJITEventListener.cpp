//===-- IntelJITEventListener.cpp - Tell Intel profiler about JITed code --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITEventListener object to tell Intel(R) VTune(TM)
// Amplifier XE 2011 about JITted functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/JITEventListener.h"

#define DEBUG_TYPE "amplifier-jit-event-listener"
#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/ValueHandle.h"
#include "EventListenerCommon.h"
#include "IntelJITEventsWrapper.h"

using namespace llvm;
using namespace llvm::jitprofiling;

namespace {

class IntelJITEventListener : public JITEventListener {
  typedef DenseMap<void*, unsigned int> MethodIDMap;

  OwningPtr<IntelJITEventsWrapper> Wrapper;
  MethodIDMap MethodIDs;
  FilenameCache Filenames;

  typedef SmallVector<const void *, 64> MethodAddressVector;
  typedef DenseMap<const void *, MethodAddressVector>  ObjectMap;

  ObjectMap  LoadedObjectMap;

public:
  IntelJITEventListener(IntelJITEventsWrapper* libraryWrapper) {
      Wrapper.reset(libraryWrapper);
  }

  ~IntelJITEventListener() {
  }

  virtual void NotifyFunctionEmitted(const Function &F,
                                     void *FnStart, size_t FnSize,
                                     const EmittedFunctionDetails &Details);

  virtual void NotifyFreeingMachineCode(void *OldPtr);

  virtual void NotifyObjectEmitted(const ObjectImage &Obj);

  virtual void NotifyFreeingObject(const ObjectImage &Obj);
};

static LineNumberInfo LineStartToIntelJITFormat(
    uintptr_t StartAddress,
    uintptr_t Address,
    DebugLoc Loc) {
  LineNumberInfo Result;

  Result.Offset = Address - StartAddress;
  Result.LineNumber = Loc.getLine();

  return Result;
}

static iJIT_Method_Load FunctionDescToIntelJITFormat(
    IntelJITEventsWrapper& Wrapper,
    const char* FnName,
    uintptr_t FnStart,
    size_t FnSize) {
  iJIT_Method_Load Result;
  memset(&Result, 0, sizeof(iJIT_Method_Load));

  Result.method_id = Wrapper.iJIT_GetNewMethodID();
  Result.method_name = const_cast<char*>(FnName);
  Result.method_load_address = reinterpret_cast<void*>(FnStart);
  Result.method_size = FnSize;

  Result.class_id = 0;
  Result.class_file_name = NULL;
  Result.user_data = NULL;
  Result.user_data_size = 0;
  Result.env = iJDE_JittingAPI;

  return Result;
}

// Adds the just-emitted function to the symbol table.
void IntelJITEventListener::NotifyFunctionEmitted(
    const Function &F, void *FnStart, size_t FnSize,
    const EmittedFunctionDetails &Details) {
  iJIT_Method_Load FunctionMessage = FunctionDescToIntelJITFormat(*Wrapper,
                                      F.getName().data(),
                                      reinterpret_cast<uint64_t>(FnStart),
                                      FnSize);

  std::vector<LineNumberInfo> LineInfo;

  if (!Details.LineStarts.empty()) {
    // Now convert the line number information from the address/DebugLoc
    // format in Details to the offset/lineno in Intel JIT API format.

    LineInfo.reserve(Details.LineStarts.size() + 1);

    DebugLoc FirstLoc = Details.LineStarts[0].Loc;
    assert(!FirstLoc.isUnknown()
           && "LineStarts should not contain unknown DebugLocs");

    MDNode *FirstLocScope = FirstLoc.getScope(F.getContext());
    DISubprogram FunctionDI = getDISubprogram(FirstLocScope);
    if (FunctionDI.Verify()) {
      FunctionMessage.source_file_name = const_cast<char*>(
                                          Filenames.getFullPath(FirstLocScope));

      LineNumberInfo FirstLine;
      FirstLine.Offset = 0;
      FirstLine.LineNumber = FunctionDI.getLineNumber();
      LineInfo.push_back(FirstLine);
    }

    for (std::vector<EmittedFunctionDetails::LineStart>::const_iterator I =
          Details.LineStarts.begin(), E = Details.LineStarts.end();
          I != E; ++I) {
      // This implementation ignores the DebugLoc filename because the Intel
      // JIT API does not support multiple source files associated with a single
      // JIT function
      LineInfo.push_back(LineStartToIntelJITFormat(
                          reinterpret_cast<uintptr_t>(FnStart),
                          I->Address,
                          I->Loc));

      // If we have no file name yet for the function, use the filename from
      // the first instruction that has one
      if (FunctionMessage.source_file_name == 0) {
        MDNode *scope = I->Loc.getScope(
          Details.MF->getFunction()->getContext());
        FunctionMessage.source_file_name = const_cast<char*>(
                                                  Filenames.getFullPath(scope));
      }
    }

    FunctionMessage.line_number_size = LineInfo.size();
    FunctionMessage.line_number_table = &*LineInfo.begin();
  } else {
    FunctionMessage.line_number_size = 0;
    FunctionMessage.line_number_table = 0;
  }

  Wrapper->iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED,
                            &FunctionMessage);
  MethodIDs[FnStart] = FunctionMessage.method_id;
}

void IntelJITEventListener::NotifyFreeingMachineCode(void *FnStart) {
  MethodIDMap::iterator I = MethodIDs.find(FnStart);
  if (I != MethodIDs.end()) {
    Wrapper->iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_UNLOAD_START, &I->second);
    MethodIDs.erase(I);
  }
}

void IntelJITEventListener::NotifyObjectEmitted(const ObjectImage &Obj) {
  // Get the address of the object image for use as a unique identifier
  const void* ObjData = Obj.getData().data();
  MethodAddressVector Functions;

  // Use symbol info to iterate functions in the object.
  error_code ec;
  for (object::symbol_iterator I = Obj.begin_symbols(),
                               E = Obj.end_symbols();
                        I != E && !ec;
                        I.increment(ec)) {
    object::SymbolRef::Type SymType;
    if (I->getType(SymType)) continue;
    if (SymType == object::SymbolRef::ST_Function) {
      StringRef Name;
      uint64_t  Addr;
      uint64_t  Size;
      if (I->getName(Name)) continue;
      if (I->getAddress(Addr)) continue;
      if (I->getSize(Size)) continue;

      // Record this address in a local vector
      Functions.push_back((void*)Addr);

      // Build the function loaded notification message
      iJIT_Method_Load FunctionMessage = FunctionDescToIntelJITFormat(*Wrapper,
                                           Name.data(),
                                           Addr,
                                           Size);

      // FIXME: Try to find line info for this function in the DWARF sections.
      FunctionMessage.source_file_name = 0;
      FunctionMessage.line_number_size = 0;
      FunctionMessage.line_number_table = 0;

      Wrapper->iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED,
                                &FunctionMessage);
      MethodIDs[(void*)Addr] = FunctionMessage.method_id;
    }
  }

  // To support object unload notification, we need to keep a list of
  // registered function addresses for each loaded object.  We will
  // use the MethodIDs map to get the registered ID for each function.
  LoadedObjectMap[ObjData] = Functions;
}

void IntelJITEventListener::NotifyFreeingObject(const ObjectImage &Obj) {
  // Get the address of the object image for use as a unique identifier
  const void* ObjData = Obj.getData().data();

  // Get the object's function list from LoadedObjectMap
  ObjectMap::iterator OI = LoadedObjectMap.find(ObjData);
  if (OI == LoadedObjectMap.end())
    return;
  MethodAddressVector& Functions = OI->second;

  // Walk the function list, unregistering each function
  for (MethodAddressVector::iterator FI = Functions.begin(),
                                     FE = Functions.end();
       FI != FE;
       ++FI) {
    void* FnStart = const_cast<void*>(*FI);
    MethodIDMap::iterator MI = MethodIDs.find(FnStart);
    if (MI != MethodIDs.end()) {
      Wrapper->iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_UNLOAD_START,
                                &MI->second);
      MethodIDs.erase(MI);
    }
  }

  // Erase the object from LoadedObjectMap
  LoadedObjectMap.erase(OI);
}

}  // anonymous namespace.

namespace llvm {
JITEventListener *JITEventListener::createIntelJITEventListener() {
  return new IntelJITEventListener(new IntelJITEventsWrapper);
}

// for testing
JITEventListener *JITEventListener::createIntelJITEventListener(
                                      IntelJITEventsWrapper* TestImpl) {
  return new IntelJITEventListener(TestImpl);
}

} // namespace llvm

