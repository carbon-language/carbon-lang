//= RemoteTargetExternal.inc - LLVM out-of-process JIT execution for Windows =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of the Windows-specific parts of the RemoteTargetExternal class
// which is meant to execute JITed code in a separate process from where it was
// built.  To support this functionality on Windows, implement these functions.
//
//===----------------------------------------------------------------------===//

namespace llvm {

bool RemoteTargetExternal::create() {
  return false;
}

bool RemoteTargetExternal::WriteBytes(const void *Data, size_t Size) {
  return false;
}

bool RemoteTargetExternal::ReadBytes(void *Data, size_t Size) {
  return false;
}

void RemoteTargetExternal::Wait() {
}

RemoteTargetExternal::~RemoteTargetExternal() {
}

} // namespace llvm
