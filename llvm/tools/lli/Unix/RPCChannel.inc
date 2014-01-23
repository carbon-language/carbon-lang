//=- RPCChannel.inc - LLVM out-of-process JIT execution for Unix --=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Unix-specific parts of the RPCChannel class
// which executes JITed code in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct ConnectionData_t {
  int InputPipe;
  int OutputPipe;

  ConnectionData_t(int in, int out) : InputPipe(in), OutputPipe(out) {}
};

} // namespace

namespace llvm {

bool RPCChannel::createServer() {
  int PipeFD[2][2];
  pid_t ChildPID;

  // Create two pipes.
  if (pipe(PipeFD[0]) != 0 || pipe(PipeFD[1]) != 0)
    perror("Error creating pipe: ");

  ChildPID = fork();

  if (ChildPID == 0) {
    // In the child...

    // Close the parent ends of the pipes
    close(PipeFD[0][1]);
    close(PipeFD[1][0]);

    // Use our pipes as stdin and stdout
    if (PipeFD[0][0] != STDIN_FILENO) {
      dup2(PipeFD[0][0], STDIN_FILENO);
      close(PipeFD[0][0]);
    }
    if (PipeFD[1][1] != STDOUT_FILENO) {
      dup2(PipeFD[1][1], STDOUT_FILENO);
      close(PipeFD[1][1]);
    }

    // Execute the child process.
    char *args[1] = { NULL };
    int rc = execv(ChildName.c_str(), args);
    if (rc != 0)
      perror("Error executing child process: ");
  } else {
    // In the parent...

    // Close the child ends of the pipes
    close(PipeFD[0][0]);
    close(PipeFD[1][1]);

    // Store the parent ends of the pipes
    ConnectionData = (void *)new ConnectionData_t(PipeFD[1][0], PipeFD[0][1]);
    return true;
  }
  return false;
}

bool RPCChannel::createClient() {
  // Store the parent ends of the pipes
  ConnectionData = (void *)new ConnectionData_t(STDIN_FILENO, STDOUT_FILENO);
  return true;
}

void RPCChannel::ReportError(int rc, size_t Size, std::string &ErrorMsg) {
  if (rc == -1) {
    if (errno == EPIPE)
      ErrorMsg += "pipe closed";
    else if (errno == EINTR)
      ErrorMsg += "interrupted";
    else
      ErrorMsg += "file descriptor error";
  } else {
    char Number[10] = { 0 };
    ErrorMsg += "Expecting ";
    sprintf(Number, "%d", (uint32_t)Size);
    ErrorMsg += Number;
    ErrorMsg += " bytes, Got ";
    sprintf(Number, "%d", rc);
    ErrorMsg += Number;
  }
}

int RPCChannel::WriteBytes(const void *Data, size_t Size) {
  return write(((ConnectionData_t *)ConnectionData)->OutputPipe, Data, Size);
}

int RPCChannel::ReadBytes(void *Data, size_t Size) {
  return read(((ConnectionData_t *)ConnectionData)->InputPipe, Data, Size);
}

void RPCChannel::Wait() { wait(NULL); }

RPCChannel::~RPCChannel() {
  delete static_cast<ConnectionData_t *>(ConnectionData);
}

} // namespace llvm
