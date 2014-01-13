#include "llvm/Config/config.h"
#include "../RemoteTargetMessage.h"
#include <assert.h>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

using namespace llvm;

class LLIChildTarget {
public:
  ~LLIChildTarget(); // OS-specific destructor
  void initialize();
  LLIMessageType waitForIncomingMessage();
  void handleMessage(LLIMessageType messageType);

private:
  // Incoming message handlers
  void handleAllocateSpace();
  void handleLoadSection(bool IsCode);
  void handleExecute();
  void handleTerminate();

  // Outgoing message handlers
  void sendChildActive();
  void sendAllocationResult(uint64_t Addr);
  void sendLoadComplete();
  void sendExecutionComplete(uint64_t Result);

  // OS-specific functions
  void initializeConnection();
  int WriteBytes(const void *Data, size_t Size);
  int ReadBytes(void *Data, size_t Size);
  uint64_t allocate(uint32_t Alignment, uint32_t Size);
  void makeSectionExecutable(uint64_t Addr, uint32_t Size);
  void InvalidateInstructionCache(const void *Addr, size_t Len);
  void releaseMemory(uint64_t Addr, uint32_t Size);

  // Store a map of allocated buffers to sizes.
  typedef std::map<uint64_t, uint32_t> AllocMapType;
  AllocMapType m_AllocatedBufferMap;

  // Communication handles (OS-specific)
  void *ConnectionData;
};

int main() {
  LLIChildTarget  ThisChild;
  ThisChild.initialize();
  LLIMessageType MsgType;
  do {
    MsgType = ThisChild.waitForIncomingMessage();
    ThisChild.handleMessage(MsgType);
  } while (MsgType != LLI_Terminate &&
           MsgType != LLI_Error);
  return 0;
}

// Public methods
void LLIChildTarget::initialize() {
  initializeConnection();
  sendChildActive();
}

LLIMessageType LLIChildTarget::waitForIncomingMessage() {
  int32_t MsgType = -1;
  if (ReadBytes(&MsgType, 4) > 0)
    return (LLIMessageType)MsgType;
  return LLI_Error;
}

void LLIChildTarget::handleMessage(LLIMessageType messageType) {
  switch (messageType) {
    case LLI_AllocateSpace:
      handleAllocateSpace();
      break;
    case LLI_LoadCodeSection:
      handleLoadSection(true);
      break;
    case LLI_LoadDataSection:
      handleLoadSection(false);
      break;
    case LLI_Execute:
      handleExecute();
      break;
    case LLI_Terminate:
      handleTerminate();
      break;
    default:
      // FIXME: Handle error!
      break;
  }
}

// Incoming message handlers
void LLIChildTarget::handleAllocateSpace() {
  // Read and verify the message data size.
  uint32_t DataSize;
  int rc = ReadBytes(&DataSize, 4);
  (void)rc;
  assert(rc == 4);
  assert(DataSize == 8);

  // Read the message arguments.
  uint32_t Alignment;
  uint32_t AllocSize;
  rc = ReadBytes(&Alignment, 4);
  assert(rc == 4);
  rc = ReadBytes(&AllocSize, 4);
  assert(rc == 4);

  // Allocate the memory.
  uint64_t Addr = allocate(Alignment, AllocSize);

  // Send AllocationResult message.
  sendAllocationResult(Addr);
}

void LLIChildTarget::handleLoadSection(bool IsCode) {
  // Read the message data size.
  uint32_t DataSize;
  int rc = ReadBytes(&DataSize, 4);
  assert(rc == 4);

  // Read the target load address.
  uint64_t Addr;
  rc = ReadBytes(&Addr, 8);
  assert(rc == 8);

  size_t BufferSize = DataSize - 8;

  // FIXME: Verify that this is in allocated space

  // Read section data into previously allocated buffer
  rc = ReadBytes((void*)Addr, DataSize - 8);
  assert(rc == (int)(BufferSize));

  // If IsCode, mark memory executable
  if (IsCode)
    makeSectionExecutable(Addr, BufferSize);

  // Send MarkLoadComplete message.
  sendLoadComplete();
}

void LLIChildTarget::handleExecute() {
  // Read the message data size.
  uint32_t DataSize;
  int rc = ReadBytes(&DataSize, 4);
  assert(rc == 4);
  assert(DataSize == 8);

  // Read the target address.
  uint64_t Addr;
  rc = ReadBytes(&Addr, 8);
  assert(rc == 8);

  // Call function
  int Result;
  int (*fn)(void) = (int(*)(void))Addr;
  Result = fn();

  // Send ExecutionResult message.
  sendExecutionComplete((int64_t)Result);
}

void LLIChildTarget::handleTerminate() {
  // Release all allocated memory
  AllocMapType::iterator Begin = m_AllocatedBufferMap.begin();
  AllocMapType::iterator End = m_AllocatedBufferMap.end();
  for (AllocMapType::iterator It = Begin; It != End; ++It) {
    releaseMemory(It->first, It->second);
  }
  m_AllocatedBufferMap.clear();
}

// Outgoing message handlers
void LLIChildTarget::sendChildActive() {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_ChildActive;
  int rc = WriteBytes(&MsgType, 4);
  assert(rc == 4);

  // Write the data size.
  uint32_t DataSize = 0;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);
}

void LLIChildTarget::sendAllocationResult(uint64_t Addr) {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_AllocationResult;
  int rc = WriteBytes(&MsgType, 4);
  assert(rc == 4);

  // Write the data size.
  uint32_t DataSize = 8;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);

  // Write the allocated address.
  rc = WriteBytes(&Addr, 8);
  assert(rc == 8);
}

void LLIChildTarget::sendLoadComplete() {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_LoadComplete;
  int rc = WriteBytes(&MsgType, 4);
  assert(rc == 4);

  // Write the data size.
  uint32_t DataSize = 0;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);
}

void LLIChildTarget::sendExecutionComplete(uint64_t Result) {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_ExecutionResult;
  int rc = WriteBytes(&MsgType, 4);
  assert(rc == 4);


  // Write the data size.
  uint32_t DataSize = 8;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);

  // Write the result.
  rc = WriteBytes(&Result, 8);
  assert(rc == 8);
}

#ifdef LLVM_ON_UNIX
#include "Unix/ChildTarget.inc"
#endif

#ifdef LLVM_ON_WIN32
#include "Windows/ChildTarget.inc"
#endif
