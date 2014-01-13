//===---- RemoteTargetExternal.cpp - LLVM out-of-process JIT execution ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the RemoteTargetExternal class which executes JITed code
// in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "RemoteTarget.h"
#include "RemoteTargetExternal.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

bool RemoteTargetExternal::allocateSpace(size_t Size, unsigned Alignment,
                                 uint64_t &Address) {
  SendAllocateSpace(Alignment, Size);
  Receive(LLI_AllocationResult, Address);
  return false;
}

bool RemoteTargetExternal::loadData(uint64_t Address, const void *Data, size_t Size) {
  SendLoadSection(Address, Data, (uint32_t)Size, false);
  Receive(LLI_LoadComplete);
  return false;
}

bool RemoteTargetExternal::loadCode(uint64_t Address, const void *Data, size_t Size) {
  SendLoadSection(Address, Data, (uint32_t)Size, true);
  Receive(LLI_LoadComplete);
  return false;
}

bool RemoteTargetExternal::executeCode(uint64_t Address, int &RetVal) {
  SendExecute(Address);

  Receive(LLI_ExecutionResult, RetVal);
  return false;
}

void RemoteTargetExternal::stop() {
  SendTerminate();
  Wait();
}

void RemoteTargetExternal::SendAllocateSpace(uint32_t Alignment, uint32_t Size) {
  int rc;
  (void)rc;
  uint32_t MsgType = (uint32_t)LLI_AllocateSpace;
  rc = WriteBytes(&MsgType, 4);
  assert(rc == 4 && "Error writing message type.");

  uint32_t DataSize = 8;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4 && "Error writing data size.");

  rc = WriteBytes(&Alignment, 4);
  assert(rc == 4 && "Error writing alignment data.");

  rc = WriteBytes(&Size, 4);
  assert(rc == 4 && "Error writing size data.");
}

void RemoteTargetExternal::SendLoadSection(uint64_t Addr,
                                       const void *Data,
                                       uint32_t Size,
                                       bool IsCode) {
  int rc;
  (void)rc;
  uint32_t MsgType = IsCode ? LLI_LoadCodeSection : LLI_LoadDataSection;
  rc = WriteBytes(&MsgType, 4);
  assert(rc == 4 && "Error writing message type.");

  uint32_t DataSize = Size + 8;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4 && "Error writing data size.");

  rc = WriteBytes(&Addr, 8);
  assert(rc == 8 && "Error writing data.");

  rc = WriteBytes(Data, Size);
  assert(rc == (int)Size && "Error writing data.");
}

void RemoteTargetExternal::SendExecute(uint64_t Addr) {
  int rc;
  (void)rc;
  uint32_t MsgType = (uint32_t)LLI_Execute;
  rc = WriteBytes(&MsgType, 4);
  assert(rc == 4 && "Error writing message type.");

  uint32_t DataSize = 8;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4 && "Error writing data size.");

  rc = WriteBytes(&Addr, 8);
  assert(rc == 8 && "Error writing data.");
}

void RemoteTargetExternal::SendTerminate() {
  int rc;
  (void)rc;
  uint32_t MsgType = (uint32_t)LLI_Terminate;
  rc = WriteBytes(&MsgType, 4);
  assert(rc == 4 && "Error writing message type.");

  // No data or data size is sent with Terminate
}


void RemoteTargetExternal::Receive(LLIMessageType ExpectedMsgType) {
  int rc;
  (void)rc;
  uint32_t MsgType;
  rc = ReadBytes(&MsgType, 4);
  assert(rc == 4 && "Error reading message type.");
  assert(MsgType == (uint32_t)ExpectedMsgType && "Error: received unexpected message type.");

  uint32_t DataSize;
  rc = ReadBytes(&DataSize, 4);
  assert(rc == 4 && "Error reading data size.");
  assert(DataSize == 0 && "Error: unexpected data size.");
}

void RemoteTargetExternal::Receive(LLIMessageType ExpectedMsgType, int &Data) {
  uint64_t Temp;
  Receive(ExpectedMsgType, Temp);
  Data = (int)(int64_t)Temp;
}

void RemoteTargetExternal::Receive(LLIMessageType ExpectedMsgType, uint64_t &Data) {
  int rc;
  (void)rc;
  uint32_t MsgType;
  rc = ReadBytes(&MsgType, 4);
  assert(rc == 4 && "Error reading message type.");
  assert(MsgType == (uint32_t)ExpectedMsgType && "Error: received unexpected message type.");

  uint32_t DataSize;
  rc = ReadBytes(&DataSize, 4);
  assert(rc == 4 && "Error reading data size.");
  assert(DataSize == 8 && "Error: unexpected data size.");

  rc = ReadBytes(&Data, 8);
  assert(DataSize == 8 && "Error: unexpected data.");
}

#ifdef LLVM_ON_UNIX
#include "Unix/RemoteTargetExternal.inc"
#endif

#ifdef LLVM_ON_WIN32
#include "Windows/RemoteTargetExternal.inc"
#endif
