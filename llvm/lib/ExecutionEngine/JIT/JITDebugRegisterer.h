//===-- JITDebugRegisterer.h - Register debug symbols for JIT -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITDebugRegisterer object that is used by the JIT to
// register debug info with debuggers like GDB.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_JIT_DEBUGREGISTERER_H
#define LLVM_EXECUTION_ENGINE_JIT_DEBUGREGISTERER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/DataTypes.h"
#include <string>

// This must be kept in sync with gdb/gdb/jit.h .
extern "C" {

  typedef enum {
    JIT_NOACTION = 0,
    JIT_REGISTER_FN,
    JIT_UNREGISTER_FN
  } jit_actions_t;

  struct jit_code_entry {
    struct jit_code_entry *next_entry;
    struct jit_code_entry *prev_entry;
    const char *symfile_addr;
    uint64_t symfile_size;
  };

  struct jit_descriptor {
    uint32_t version;
    // This should be jit_actions_t, but we want to be specific about the
    // bit-width.
    uint32_t action_flag;
    struct jit_code_entry *relevant_entry;
    struct jit_code_entry *first_entry;
  };

}

namespace llvm {

class ELFSection;
class Function;
class TargetMachine;


/// This class encapsulates information we want to send to the debugger.
///
struct DebugInfo {
  uint8_t *FnStart;
  uint8_t *FnEnd;
  uint8_t *EhStart;
  uint8_t *EhEnd;

  DebugInfo() : FnStart(0), FnEnd(0), EhStart(0), EhEnd(0) {}
};

typedef DenseMap< const Function*, std::pair<std::string, jit_code_entry*> >
  RegisteredFunctionsMap;

/// This class registers debug info for JITed code with an attached debugger.
/// Without proper debug info, GDB can't do things like source level debugging
/// or even produce a proper stack trace on linux-x86_64.  To use this class,
/// whenever a function is JITed, create a DebugInfo struct and pass it to the
/// RegisterFunction method.  The method will then do whatever is necessary to
/// inform the debugger about the JITed function.
class JITDebugRegisterer {

  TargetMachine &TM;

  /// FnMap - A map of functions that have been registered to the associated
  /// temporary files.  Used for cleanup.
  RegisteredFunctionsMap FnMap;

  /// MakeELF - Builds the ELF file in memory and returns a std::string that
  /// contains the ELF.
  std::string MakeELF(const Function *F, DebugInfo &I);

public:
  JITDebugRegisterer(TargetMachine &tm);

  /// ~JITDebugRegisterer - Unregisters all code and frees symbol files.
  ///
  ~JITDebugRegisterer();

  /// RegisterFunction - Register debug info for the given function with an
  /// attached debugger.  Clients must call UnregisterFunction on all
  /// registered functions before deleting them to free the associated symbol
  /// file and unregister it from the debugger.
  void RegisterFunction(const Function *F, DebugInfo &I);

  /// UnregisterFunction - Unregister the debug info for the given function
  /// from the debugger and free associated memory.
  void UnregisterFunction(const Function *F);

private:
  /// UnregisterFunctionInternal - Unregister the debug info for the given
  /// function from the debugger and delete any temporary files.  The private
  /// version of this method does not remove the function from FnMap so that it
  /// can be called while iterating over FnMap.
  void UnregisterFunctionInternal(RegisteredFunctionsMap::iterator I);

};

} // end namespace llvm

#endif // LLVM_EXECUTION_ENGINE_JIT_DEBUGREGISTERER_H
