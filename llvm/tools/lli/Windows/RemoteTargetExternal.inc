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

void RemoteTargetExternal::create() {
}

int RemoteTargetExternal::WriteBytes(const void *Data, size_t Size) {
  return 0;
}

int RemoteTargetExternal::ReadBytes(void *Data, size_t Size) {
  return 0;
}

void RemoteTargetExternal::Wait() {
}

RemoteTargetExternal::~RemoteTargetExternal() {
}

} // namespace llvm
