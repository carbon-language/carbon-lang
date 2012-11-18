//===- RecordingMemoryManager.cpp - Recording memory manager --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This memory manager allocates local storage and keeps a record of each
// allocation. Iterators are provided for all data and code allocations.
//
//===----------------------------------------------------------------------===//

#include "RecordingMemoryManager.h"
using namespace llvm;

uint8_t *RecordingMemoryManager::
allocateCodeSection(uintptr_t Size, unsigned Alignment, unsigned SectionID) {
  // The recording memory manager is just a local copy of the remote target.
  // The alignment requirement is just stored here for later use. Regular
  // heap storage is sufficient here.
  void *Addr = malloc(Size);
  assert(Addr && "malloc() failure!");
  sys::MemoryBlock Block(Addr, Size);
  AllocatedCodeMem.push_back(Allocation(Block, Alignment));
  return (uint8_t*)Addr;
}

uint8_t *RecordingMemoryManager::
allocateDataSection(uintptr_t Size, unsigned Alignment,
                    unsigned SectionID, bool IsReadOnly) {
  // The recording memory manager is just a local copy of the remote target.
  // The alignment requirement is just stored here for later use. Regular
  // heap storage is sufficient here.
  void *Addr = malloc(Size);
  assert(Addr && "malloc() failure!");
  sys::MemoryBlock Block(Addr, Size);
  AllocatedDataMem.push_back(Allocation(Block, Alignment));
  return (uint8_t*)Addr;
}
void RecordingMemoryManager::setMemoryWritable() { llvm_unreachable("Unexpected!"); }
void RecordingMemoryManager::setMemoryExecutable() { llvm_unreachable("Unexpected!"); }
void RecordingMemoryManager::setPoisonMemory(bool poison) { llvm_unreachable("Unexpected!"); }
void RecordingMemoryManager::AllocateGOT() { llvm_unreachable("Unexpected!"); }
uint8_t *RecordingMemoryManager::getGOTBase() const {
  llvm_unreachable("Unexpected!");
  return 0;
}
uint8_t *RecordingMemoryManager::startFunctionBody(const Function *F, uintptr_t &ActualSize){
  llvm_unreachable("Unexpected!");
  return 0;
}
uint8_t *RecordingMemoryManager::allocateStub(const GlobalValue* F, unsigned StubSize,
                                              unsigned Alignment) {
  llvm_unreachable("Unexpected!");
  return 0;
}
void RecordingMemoryManager::endFunctionBody(const Function *F, uint8_t *FunctionStart,
                                             uint8_t *FunctionEnd) {
  llvm_unreachable("Unexpected!");
}
uint8_t *RecordingMemoryManager::allocateSpace(intptr_t Size, unsigned Alignment) {
  llvm_unreachable("Unexpected!");
  return 0;
}
uint8_t *RecordingMemoryManager::allocateGlobal(uintptr_t Size, unsigned Alignment) {
  llvm_unreachable("Unexpected!");
  return 0;
}
void RecordingMemoryManager::deallocateFunctionBody(void *Body) {
  llvm_unreachable("Unexpected!");
}
uint8_t* RecordingMemoryManager::startExceptionTable(const Function* F, uintptr_t &ActualSize) {
  llvm_unreachable("Unexpected!");
  return 0;
}
void RecordingMemoryManager::endExceptionTable(const Function *F, uint8_t *TableStart,
                                               uint8_t *TableEnd, uint8_t* FrameRegister) {
  llvm_unreachable("Unexpected!");
}
void RecordingMemoryManager::deallocateExceptionTable(void *ET) {
  llvm_unreachable("Unexpected!");
}

static int jit_noop() {
  return 0;
}

void *RecordingMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                        bool AbortOnFailure) {
  // We should not invoke parent's ctors/dtors from generated main()!
  // On Mingw and Cygwin, the symbol __main is resolved to
  // callee's(eg. tools/lli) one, to invoke wrong duplicated ctors
  // (and register wrong callee's dtors with atexit(3)).
  // We expect ExecutionEngine::runStaticConstructorsDestructors()
  // is called before ExecutionEngine::runFunctionAsMain() is called.
  if (Name == "__main") return (void*)(intptr_t)&jit_noop;

  return NULL;
}
