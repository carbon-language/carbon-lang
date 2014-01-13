//===----- RemoteTargetExternal.h - LLVM out-of-process JIT execution -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of the RemoteTargetExternal class which executes JITed code in a
// separate process from where it was built.
//
//===----------------------------------------------------------------------===//

#ifndef LLI_REMOTETARGETEXTERNAL_H
#define LLI_REMOTETARGETEXTERNAL_H

#include "RemoteTarget.h"
#include "RemoteTargetMessage.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Memory.h"
#include <stdlib.h>
#include <string>

namespace llvm {

class RemoteTargetExternal : public RemoteTarget {
public:
  /// Allocate space in the remote target address space.
  ///
  /// @param      Size      Amount of space, in bytes, to allocate.
  /// @param      Alignment Required minimum alignment for allocated space.
  /// @param[out] Address   Remote address of the allocated memory.
  ///
  /// @returns False on success. On failure, ErrorMsg is updated with
  ///          descriptive text of the encountered error.
  virtual bool allocateSpace(size_t Size,
                             unsigned Alignment,
                             uint64_t &Address);

  /// Load data into the target address space.
  ///
  /// @param      Address   Destination address in the target process.
  /// @param      Data      Source address in the host process.
  /// @param      Size      Number of bytes to copy.
  ///
  /// @returns False on success. On failure, ErrorMsg is updated with
  ///          descriptive text of the encountered error.
  virtual bool loadData(uint64_t Address, const void *Data, size_t Size);

  /// Load code into the target address space and prepare it for execution.
  ///
  /// @param      Address   Destination address in the target process.
  /// @param      Data      Source address in the host process.
  /// @param      Size      Number of bytes to copy.
  ///
  /// @returns False on success. On failure, ErrorMsg is updated with
  ///          descriptive text of the encountered error.
  virtual bool loadCode(uint64_t Address, const void *Data, size_t Size);

  /// Execute code in the target process. The called function is required
  /// to be of signature int "(*)(void)".
  ///
  /// @param      Address   Address of the loaded function in the target
  ///                       process.
  /// @param[out] RetVal    The integer return value of the called function.
  ///
  /// @returns False on success. On failure, ErrorMsg is updated with
  ///          descriptive text of the encountered error.
  virtual bool executeCode(uint64_t Address, int &RetVal);

  /// Minimum alignment for memory permissions. Used to seperate code and
  /// data regions to make sure data doesn't get marked as code or vice
  /// versa.
  ///
  /// @returns Page alignment return value. Default of 4k.
  virtual unsigned getPageAlignment() { return 4096; }

  /// Start the remote process.
  virtual void create();

  /// Terminate the remote process.
  virtual void stop();

  RemoteTargetExternal(std::string &Name) : RemoteTarget(), ChildName(Name) {}
  virtual ~RemoteTargetExternal();

private:
  std::string ChildName;

  // This will get filled in as a point to an OS-specific structure.
  void *ConnectionData;

  void SendAllocateSpace(uint32_t Alignment, uint32_t Size);
  void SendLoadSection(uint64_t Addr,
                       const void *Data,
                       uint32_t Size,
                       bool IsCode);
  void SendExecute(uint64_t Addr);
  void SendTerminate();

  void Receive(LLIMessageType Msg);
  void Receive(LLIMessageType Msg, int &Data);
  void Receive(LLIMessageType Msg, uint64_t &Data);

  int WriteBytes(const void *Data, size_t Size);
  int ReadBytes(void *Data, size_t Size);
  void Wait();
};

} // end namespace llvm

#endif // LLI_REMOTETARGETEXTERNAL_H
