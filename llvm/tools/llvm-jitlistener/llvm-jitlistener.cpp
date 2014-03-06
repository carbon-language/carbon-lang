//===-- llvm-jitlistener.cpp - Utility for testing MCJIT event listener ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a used by lit tests to verify the MCJIT JITEventListener
// interface.  It registers a mock JIT event listener, generates a module from
// an input IR file and dumps the reported event information to stdout.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "../../lib/ExecutionEngine/IntelJITEvents/IntelJITEventsWrapper.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include <string>

using namespace llvm;

namespace {

typedef std::vector<std::pair<std::string, unsigned int> > SourceLocations;
typedef std::map<uint64_t, SourceLocations> NativeCodeMap;

NativeCodeMap  ReportedDebugFuncs;

int NotifyEvent(iJIT_JVM_EVENT EventType, void *EventSpecificData) {
  switch (EventType) {
    case iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED: {
      if (!EventSpecificData) {
        errs() <<
          "Error: The JIT event listener did not provide a event data.";
        return -1;
      }
      iJIT_Method_Load* msg = static_cast<iJIT_Method_Load*>(EventSpecificData);

      ReportedDebugFuncs[msg->method_id];

      outs() << "Method load [" << msg->method_id << "]: " << msg->method_name
             << ", Size = " << msg->method_size << "\n";

      for(unsigned int i = 0; i < msg->line_number_size; ++i) {
        if (!msg->line_number_table) {
          errs() << "A function with a non-zero line count had no line table.";
          return -1;
        }
        std::pair<std::string, unsigned int> loc(
          std::string(msg->source_file_name),
          msg->line_number_table[i].LineNumber);
        ReportedDebugFuncs[msg->method_id].push_back(loc);
        outs() << "  Line info @ " << msg->line_number_table[i].Offset
               << ": " << msg->source_file_name
               << ", line " << msg->line_number_table[i].LineNumber << "\n";
      }
      outs() << "\n";
    }
    break;
    case iJVM_EVENT_TYPE_METHOD_UNLOAD_START: {
      if (!EventSpecificData) {
        errs() <<
          "Error: The JIT event listener did not provide a event data.";
        return -1;
      }
      unsigned int UnloadId
        = *reinterpret_cast<unsigned int*>(EventSpecificData);
      assert(1 == ReportedDebugFuncs.erase(UnloadId));
      outs() << "Method unload [" << UnloadId << "]\n";
    }
    break;
    default:
      break;
  }
  return 0;
}

iJIT_IsProfilingActiveFlags IsProfilingActive(void) {
  // for testing, pretend we have an Intel Parallel Amplifier XE 2011
  // instance attached
  return iJIT_SAMPLING_ON;
}

unsigned int GetNewMethodID(void) {
  static unsigned int id = 0;
  return ++id;
}

class JitEventListenerTest {
protected:
  void InitEE(const std::string &IRFile) {
    LLVMContext &Context = getGlobalContext();

    // If we have a native target, initialize it to ensure it is linked in and
    // usable by the JIT.
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    // Parse the bitcode...
    SMDiagnostic Err;
    TheModule = ParseIRFile(IRFile, Err, Context);
    if (!TheModule) {
      errs() << Err.getMessage();
      return;
    }

    // FIXME: This is using the default legacy JITMemoryManager because it
    // supports poison memory.  At some point, we'll need to update this to
    // use an MCJIT-specific memory manager.  It might be nice to have the
    // poison memory option there too.
    JITMemoryManager *MemMgr = JITMemoryManager::CreateDefaultMemManager();
    if (!MemMgr) {
      errs() << "Unable to create memory manager.";
      return;
    }

    // Tell the memory manager to poison freed memory so that accessing freed
    // memory is more easily tested.
    MemMgr->setPoisonMemory(true);

    // Override the triple to generate ELF on Windows since that's supported
    Triple Tuple(TheModule->getTargetTriple());
    if (Tuple.getTriple().empty())
      Tuple.setTriple(sys::getProcessTriple());

    if (Tuple.isOSWindows() && Triple::ELF != Tuple.getEnvironment()) {
      Tuple.setEnvironment(Triple::ELF);
      TheModule->setTargetTriple(Tuple.getTriple());
    }

    // Compile the IR
    std::string Error;
    TheJIT.reset(EngineBuilder(TheModule)
      .setEngineKind(EngineKind::JIT)
      .setErrorStr(&Error)
      .setJITMemoryManager(MemMgr)
      .setUseMCJIT(true)
      .create());
    if (Error.empty() == false)
      errs() << Error;
  }

  void DestroyEE() {
    TheJIT.reset();
  }

  LLVMContext Context; // Global ownership
  Module *TheModule; // Owned by ExecutionEngine.
  JITMemoryManager *JMM; // Owned by ExecutionEngine.
  std::unique_ptr<ExecutionEngine> TheJIT;

public:
  void ProcessInput(const std::string &Filename) {
    InitEE(Filename);

    std::unique_ptr<llvm::JITEventListener> Listener(
        JITEventListener::createIntelJITEventListener(new IntelJITEventsWrapper(
            NotifyEvent, 0, IsProfilingActive, 0, 0, GetNewMethodID)));

    TheJIT->RegisterJITEventListener(Listener.get());

    TheJIT->finalizeObject();

    // Destroy the JIT engine instead of unregistering to get unload events.
    DestroyEE();
  }
};



} // end anonymous namespace

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input IR file>"),
               cl::Required);

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm jit event listener test utility\n");

  JitEventListenerTest Test;

  Test.ProcessInput(InputFilename);

  return 0;
}
