//===-- JITDebugRegisterer.cpp - Register debug symbols for JIT -----------===//
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

#include "JITDebugRegisterer.h"
#include "../../CodeGen/ELF.h"
#include "../../CodeGen/ELFWriter.h"
#include "llvm/LLVMContext.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"
#include <string>

namespace llvm {

// This must be kept in sync with gdb/gdb/jit.h .
extern "C" {

  // Debuggers puts a breakpoint in this function.
  LLVM_ATTRIBUTE_NOINLINE void __jit_debug_register_code() { }

  // We put information about the JITed function in this global, which the
  // debugger reads.  Make sure to specify the version statically, because the
  // debugger checks the version before we can set it during runtime.
  struct jit_descriptor __jit_debug_descriptor = { 1, 0, 0, 0 };

}

namespace {

  /// JITDebugLock - Used to serialize all code registration events, since they
  /// modify global variables.
  sys::Mutex JITDebugLock;

}

JITDebugRegisterer::JITDebugRegisterer(TargetMachine &tm) : TM(tm), FnMap() { }

JITDebugRegisterer::~JITDebugRegisterer() {
  // Free all ELF memory.
  for (RegisteredFunctionsMap::iterator I = FnMap.begin(), E = FnMap.end();
       I != E; ++I) {
    // Call the private method that doesn't update the map so our iterator
    // doesn't break.
    UnregisterFunctionInternal(I);
  }
  FnMap.clear();
}

std::string JITDebugRegisterer::MakeELF(const Function *F, DebugInfo &I) {
  // Stack allocate an empty module with an empty LLVMContext for the ELFWriter
  // API.  We don't use the real module because then the ELFWriter would write
  // out unnecessary GlobalValues during finalization.
  LLVMContext Context;
  Module M("", Context);

  // Make a buffer for the ELF in memory.
  std::string Buffer;
  raw_string_ostream O(Buffer);
  ELFWriter EW(O, TM);
  EW.doInitialization(M);

  // Copy the binary into the .text section.  This isn't necessary, but it's
  // useful to be able to disassemble the ELF by hand.
  ELFSection &Text = EW.getTextSection(const_cast<Function *>(F));
  Text.Addr = (uint64_t)I.FnStart;
  // TODO: We could eliminate this copy if we somehow used a pointer/size pair
  // instead of a vector.
  Text.getData().assign(I.FnStart, I.FnEnd);

  // Copy the exception handling call frame information into the .eh_frame
  // section.  This allows GDB to get a good stack trace, particularly on
  // linux x86_64.  Mark this as a PROGBITS section that needs to be loaded
  // into memory at runtime.
  ELFSection &EH = EW.getSection(".eh_frame", ELF::SHT_PROGBITS,
                                 ELF::SHF_ALLOC);
  // Pointers in the DWARF EH info are all relative to the EH frame start,
  // which is stored here.
  EH.Addr = (uint64_t)I.EhStart;
  // TODO: We could eliminate this copy if we somehow used a pointer/size pair
  // instead of a vector.
  EH.getData().assign(I.EhStart, I.EhEnd);

  // Add this single function to the symbol table, so the debugger prints the
  // name instead of '???'.  We give the symbol default global visibility.
  ELFSym *FnSym = ELFSym::getGV(F,
                                ELF::STB_GLOBAL,
                                ELF::STT_FUNC,
                                ELF::STV_DEFAULT);
  FnSym->SectionIdx = Text.SectionIdx;
  FnSym->Size = I.FnEnd - I.FnStart;
  FnSym->Value = 0;  // Offset from start of section.
  EW.SymbolList.push_back(FnSym);

  EW.doFinalization(M);
  O.flush();

  // When trying to debug why GDB isn't getting the debug info right, it's
  // awfully helpful to write the object file to disk so that it can be
  // inspected with readelf and objdump.
  if (JITEmitDebugInfoToDisk) {
    std::string Filename;
    raw_string_ostream O2(Filename);
    O2 << "/tmp/llvm_function_" << I.FnStart << "_" << F->getName() << ".o";
    O2.flush();
    std::string Errors;
    raw_fd_ostream O3(Filename.c_str(), Errors);
    O3 << Buffer;
    O3.close();
  }

  return Buffer;
}

void JITDebugRegisterer::RegisterFunction(const Function *F, DebugInfo &I) {
  // TODO: Support non-ELF platforms.
  if (!TM.getELFWriterInfo())
    return;

  std::string Buffer = MakeELF(F, I);

  jit_code_entry *JITCodeEntry = new jit_code_entry();
  JITCodeEntry->symfile_addr = Buffer.c_str();
  JITCodeEntry->symfile_size = Buffer.size();

  // Add a mapping from F to the entry and buffer, so we can delete this
  // info later.
  FnMap[F] = std::make_pair(Buffer, JITCodeEntry);

  // Acquire the lock and do the registration.
  {
    MutexGuard locked(JITDebugLock);
    __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;

    // Insert this entry at the head of the list.
    JITCodeEntry->prev_entry = NULL;
    jit_code_entry *NextEntry = __jit_debug_descriptor.first_entry;
    JITCodeEntry->next_entry = NextEntry;
    if (NextEntry != NULL) {
      NextEntry->prev_entry = JITCodeEntry;
    }
    __jit_debug_descriptor.first_entry = JITCodeEntry;
    __jit_debug_descriptor.relevant_entry = JITCodeEntry;
    __jit_debug_register_code();
  }
}

void JITDebugRegisterer::UnregisterFunctionInternal(
    RegisteredFunctionsMap::iterator I) {
  jit_code_entry *&JITCodeEntry = I->second.second;

  // Acquire the lock and do the unregistration.
  {
    MutexGuard locked(JITDebugLock);
    __jit_debug_descriptor.action_flag = JIT_UNREGISTER_FN;

    // Remove the jit_code_entry from the linked list.
    jit_code_entry *PrevEntry = JITCodeEntry->prev_entry;
    jit_code_entry *NextEntry = JITCodeEntry->next_entry;
    if (NextEntry) {
      NextEntry->prev_entry = PrevEntry;
    }
    if (PrevEntry) {
      PrevEntry->next_entry = NextEntry;
    } else {
      assert(__jit_debug_descriptor.first_entry == JITCodeEntry);
      __jit_debug_descriptor.first_entry = NextEntry;
    }

    // Tell GDB which entry we removed, and unregister the code.
    __jit_debug_descriptor.relevant_entry = JITCodeEntry;
    __jit_debug_register_code();
  }

  delete JITCodeEntry;
  JITCodeEntry = NULL;

  // Free the ELF file in memory.
  std::string &Buffer = I->second.first;
  Buffer.clear();
}

void JITDebugRegisterer::UnregisterFunction(const Function *F) {
  // TODO: Support non-ELF platforms.
  if (!TM.getELFWriterInfo())
    return;

  RegisteredFunctionsMap::iterator I = FnMap.find(F);
  if (I == FnMap.end()) return;
  UnregisterFunctionInternal(I);
  FnMap.erase(I);
}

} // end namespace llvm
