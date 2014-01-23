//=- ChildTarget.inc - Child process for external JIT execution for Windows -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Non-implementation of the Windows-specific parts of the ChildTarget class
// which executes JITed code in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

LLIChildTarget::~LLIChildTarget() {
}

// The RemoteTargetExternal implementation should prevent us from ever getting
// here on Windows, but nothing prevents a user from running this directly.
void LLIChildTarget::initializeConnection() {
  assert(0 && "lli-child-target is not implemented for Windows");
}

int LLIChildTarget::WriteBytes(const void *Data, size_t Size) {
  return 0;
}

int LLIChildTarget::ReadBytes(void *Data, size_t Size) {
  return 0;
}

uint64_t LLIChildTarget::allocate(uint32_t Alignment, uint32_t Size) {
  return 0;
}
