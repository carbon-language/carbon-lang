//===---- RemoteTargetMessage.h - LLI out-of-process message protocol -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of the LLIMessageType enum which is used for communication with a
// child process for remote execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLI_REMOTETARGETMESSAGE_H
#define LLI_REMOTETARGETMESSAGE_H

namespace llvm {

// LLI messages from parent-to-child or vice versa follow an exceedingly simple
// protocol where the first four bytes represent the message type, the next
// four bytes represent the size of data for the command and following bytes
// represent the actual data.
//
// The protocol is not intended to be robust, secure or fault-tolerant.  It is
// only here for testing purposes and is therefore intended to be the simplest
// implementation that will work.  It is assumed that the parent and child
// share characteristics like endianness.

enum LLIMessageType {
  LLI_Error = -1,
  LLI_ChildActive = 0,        // Data = not used
  LLI_AllocateSpace,          // Data = struct { uint_32t Align, uint_32t Size }
  LLI_AllocationResult,       // Data = uint64_t AllocAddress (in Child memory space)
  LLI_LoadCodeSection,        // Data = uint32_t Addr, followed by section contests
  LLI_LoadDataSection,        // Data = uint32_t Addr, followed by section contents
  LLI_LoadComplete,           // Data = not used
  LLI_Execute,                // Data = Address of function to execute
  LLI_ExecutionResult,        // Data = uint64_t Result
  LLI_Terminate               // Data = not used
};

} // end namespace llvm

#endif
