//===- ChildTarget.inc - Child process for external JIT execution for Unix -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Unix-specific parts of the ChildTarget class
// which executes JITed code in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

namespace {

struct ConnectionData_t {
  int InputPipe;
  int OutputPipe;

  ConnectionData_t(int in, int out) : InputPipe(in), OutputPipe(out) {}
};

} // namespace

LLIChildTarget::~LLIChildTarget() {
  delete static_cast<ConnectionData_t *>(ConnectionData);
}

// OS-specific methods
void LLIChildTarget::initializeConnection() {
  // Store the parent ends of the pipes
  ConnectionData = (void*)new ConnectionData_t(STDIN_FILENO, STDOUT_FILENO);
}

int LLIChildTarget::WriteBytes(const void *Data, size_t Size) {
  return write(((ConnectionData_t*)ConnectionData)->OutputPipe, Data, Size);
}

int LLIChildTarget::ReadBytes(void *Data, size_t Size) {
  return read(((ConnectionData_t*)ConnectionData)->InputPipe, Data, Size);
}
