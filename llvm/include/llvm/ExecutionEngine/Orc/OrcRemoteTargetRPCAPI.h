//===- OrcRemoteTargetRPCAPI.h - Orc Remote-target RPC API ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Orc remote-target RPC API. It should not be used
// directly, but is used by the RemoteTargetClient and RemoteTargetServer
// classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETRPCAPI_H
#define LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETRPCAPI_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/RPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/RawByteChannel.h"

namespace llvm {
namespace orc {

namespace remote {

/// Template error for missing resources.
template <typename ResourceIdT>
class ResourceNotFound
  : public ErrorInfo<ResourceNotFound<ResourceIdT>> {
public:
  static char ID;

  ResourceNotFound(ResourceIdT ResourceId,
                   std::string ResourceDescription = "")
    : ResourceId(std::move(ResourceId)),
      ResourceDescription(std::move(ResourceDescription)) {}

  std::error_code convertToErrorCode() const override {
    return orcError(OrcErrorCode::UnknownResourceHandle);
  }

  void log(raw_ostream &OS) const override {
    OS << (ResourceDescription.empty()
             ? "Remote resource with id "
               : ResourceDescription)
       << " " << ResourceId << " not found";
  }

private:
  ResourceIdT ResourceId;
  std::string ResourceDescription;
};

template <typename ResourceIdT>
char ResourceNotFound<ResourceIdT>::ID = 0;

class DirectBufferWriter {
public:
  DirectBufferWriter() = default;
  DirectBufferWriter(const char *Src, JITTargetAddress Dst, uint64_t Size)
      : Src(Src), Dst(Dst), Size(Size) {}

  const char *getSrc() const { return Src; }
  JITTargetAddress getDst() const { return Dst; }
  uint64_t getSize() const { return Size; }

private:
  const char *Src;
  JITTargetAddress Dst;
  uint64_t Size;
};

} // end namespace remote

namespace shared {

template <> class SerializationTypeName<JITSymbolFlags> {
public:
  static const char *getName() { return "JITSymbolFlags"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, JITSymbolFlags> {
public:

  static Error serialize(ChannelT &C, const JITSymbolFlags &Flags) {
    return serializeSeq(C, Flags.getRawFlagsValue(), Flags.getTargetFlags());
  }

  static Error deserialize(ChannelT &C, JITSymbolFlags &Flags) {
    JITSymbolFlags::UnderlyingType JITFlags;
    JITSymbolFlags::TargetFlagsType TargetFlags;
    if (auto Err = deserializeSeq(C, JITFlags, TargetFlags))
      return Err;
    Flags = JITSymbolFlags(static_cast<JITSymbolFlags::FlagNames>(JITFlags),
                           TargetFlags);
    return Error::success();
  }
};

template <> class SerializationTypeName<remote::DirectBufferWriter> {
public:
  static const char *getName() { return "DirectBufferWriter"; }
};

template <typename ChannelT>
class SerializationTraits<
    ChannelT, remote::DirectBufferWriter, remote::DirectBufferWriter,
    std::enable_if_t<std::is_base_of<RawByteChannel, ChannelT>::value>> {
public:
  static Error serialize(ChannelT &C, const remote::DirectBufferWriter &DBW) {
    if (auto EC = serializeSeq(C, DBW.getDst()))
      return EC;
    if (auto EC = serializeSeq(C, DBW.getSize()))
      return EC;
    return C.appendBytes(DBW.getSrc(), DBW.getSize());
  }

  static Error deserialize(ChannelT &C, remote::DirectBufferWriter &DBW) {
    JITTargetAddress Dst;
    if (auto EC = deserializeSeq(C, Dst))
      return EC;
    uint64_t Size;
    if (auto EC = deserializeSeq(C, Size))
      return EC;
    char *Addr = reinterpret_cast<char *>(static_cast<uintptr_t>(Dst));

    DBW = remote::DirectBufferWriter(nullptr, Dst, Size);

    return C.readBytes(Addr, Size);
  }
};

} // end namespace shared

namespace remote {

class ResourceIdMgr {
public:
  using ResourceId = uint64_t;
  static const ResourceId InvalidId = ~0U;

  ResourceIdMgr() = default;
  explicit ResourceIdMgr(ResourceId FirstValidId)
    : NextId(std::move(FirstValidId)) {}

  ResourceId getNext() {
    if (!FreeIds.empty()) {
      ResourceId I = FreeIds.back();
      FreeIds.pop_back();
      return I;
    }
    assert(NextId + 1 != ~0ULL && "All ids allocated");
    return NextId++;
  }

  void release(ResourceId I) { FreeIds.push_back(I); }

private:
  ResourceId NextId = 1;
  std::vector<ResourceId> FreeIds;
};

/// Registers EH frames on the remote.
namespace eh {

  /// Registers EH frames on the remote.
class RegisterEHFrames
    : public shared::RPCFunction<RegisterEHFrames,
                                 void(JITTargetAddress Addr, uint32_t Size)> {
public:
  static const char *getName() { return "RegisterEHFrames"; }
};

  /// Deregisters EH frames on the remote.
class DeregisterEHFrames
    : public shared::RPCFunction<DeregisterEHFrames,
                                 void(JITTargetAddress Addr, uint32_t Size)> {
public:
  static const char *getName() { return "DeregisterEHFrames"; }
};

} // end namespace eh

/// RPC functions for executing remote code.
namespace exec {

  /// Call an 'int32_t()'-type function on the remote, returns the called
  /// function's return value.
class CallIntVoid
    : public shared::RPCFunction<CallIntVoid, int32_t(JITTargetAddress Addr)> {
public:
  static const char *getName() { return "CallIntVoid"; }
};

  /// Call an 'int32_t(int32_t)'-type function on the remote, returns the called
  /// function's return value.
class CallIntInt
    : public shared::RPCFunction<CallIntInt,
                                 int32_t(JITTargetAddress Addr, int)> {
public:
  static const char *getName() { return "CallIntInt"; }
};

  /// Call an 'int32_t(int32_t, char**)'-type function on the remote, returns the
  /// called function's return value.
class CallMain
    : public shared::RPCFunction<CallMain,
                                 int32_t(JITTargetAddress Addr,
                                         std::vector<std::string> Args)> {
public:
  static const char *getName() { return "CallMain"; }
};

  /// Calls a 'void()'-type function on the remote, returns when the called
  /// function completes.
class CallVoidVoid
    : public shared::RPCFunction<CallVoidVoid, void(JITTargetAddress FnAddr)> {
public:
  static const char *getName() { return "CallVoidVoid"; }
};

} // end namespace exec

/// RPC functions for remote memory management / inspection / modification.
namespace mem {

  /// Creates a memory allocator on the remote.
class CreateRemoteAllocator
    : public shared::RPCFunction<CreateRemoteAllocator,
                                 void(ResourceIdMgr::ResourceId AllocatorID)> {
public:
  static const char *getName() { return "CreateRemoteAllocator"; }
};

  /// Destroys a remote allocator, freeing any memory allocated by it.
class DestroyRemoteAllocator
    : public shared::RPCFunction<DestroyRemoteAllocator,
                                 void(ResourceIdMgr::ResourceId AllocatorID)> {
public:
  static const char *getName() { return "DestroyRemoteAllocator"; }
};

  /// Read a remote memory block.
class ReadMem
    : public shared::RPCFunction<
          ReadMem, std::vector<uint8_t>(JITTargetAddress Src, uint64_t Size)> {
public:
  static const char *getName() { return "ReadMem"; }
};

  /// Reserve a block of memory on the remote via the given allocator.
class ReserveMem
    : public shared::RPCFunction<
          ReserveMem, JITTargetAddress(ResourceIdMgr::ResourceId AllocID,
                                       uint64_t Size, uint32_t Align)> {
public:
  static const char *getName() { return "ReserveMem"; }
};

  /// Set the memory protection on a memory block.
class SetProtections
    : public shared::RPCFunction<
          SetProtections, void(ResourceIdMgr::ResourceId AllocID,
                               JITTargetAddress Dst, uint32_t ProtFlags)> {
public:
  static const char *getName() { return "SetProtections"; }
};

  /// Write to a remote memory block.
class WriteMem
    : public shared::RPCFunction<WriteMem,
                                 void(remote::DirectBufferWriter DB)> {
public:
  static const char *getName() { return "WriteMem"; }
};

  /// Write to a remote pointer.
class WritePtr
    : public shared::RPCFunction<WritePtr, void(JITTargetAddress Dst,
                                                JITTargetAddress Val)> {
public:
  static const char *getName() { return "WritePtr"; }
};

} // end namespace mem

/// RPC functions for remote stub and trampoline management.
namespace stubs {

  /// Creates an indirect stub owner on the remote.
class CreateIndirectStubsOwner
    : public shared::RPCFunction<CreateIndirectStubsOwner,
                                 void(ResourceIdMgr::ResourceId StubOwnerID)> {
public:
  static const char *getName() { return "CreateIndirectStubsOwner"; }
};

  /// RPC function for destroying an indirect stubs owner.
class DestroyIndirectStubsOwner
    : public shared::RPCFunction<DestroyIndirectStubsOwner,
                                 void(ResourceIdMgr::ResourceId StubsOwnerID)> {
public:
  static const char *getName() { return "DestroyIndirectStubsOwner"; }
};

  /// EmitIndirectStubs result is (StubsBase, PtrsBase, NumStubsEmitted).
class EmitIndirectStubs
    : public shared::RPCFunction<
          EmitIndirectStubs,
          std::tuple<JITTargetAddress, JITTargetAddress, uint32_t>(
              ResourceIdMgr::ResourceId StubsOwnerID,
              uint32_t NumStubsRequired)> {
public:
  static const char *getName() { return "EmitIndirectStubs"; }
};

  /// RPC function to emit the resolver block and return its address.
class EmitResolverBlock
    : public shared::RPCFunction<EmitResolverBlock, void()> {
public:
  static const char *getName() { return "EmitResolverBlock"; }
};

  /// EmitTrampolineBlock result is (BlockAddr, NumTrampolines).
class EmitTrampolineBlock
    : public shared::RPCFunction<EmitTrampolineBlock,
                                 std::tuple<JITTargetAddress, uint32_t>()> {
public:
  static const char *getName() { return "EmitTrampolineBlock"; }
};

} // end namespace stubs

/// Miscelaneous RPC functions for dealing with remotes.
namespace utils {

  /// GetRemoteInfo result is (Triple, PointerSize, PageSize, TrampolineSize,
  ///                          IndirectStubsSize).
class GetRemoteInfo
    : public shared::RPCFunction<
          GetRemoteInfo,
          std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t>()> {
public:
  static const char *getName() { return "GetRemoteInfo"; }
};

  /// Get the address of a remote symbol.
class GetSymbolAddress
    : public shared::RPCFunction<GetSymbolAddress,
                                 JITTargetAddress(std::string SymbolName)> {
public:
  static const char *getName() { return "GetSymbolAddress"; }
};

  /// Request that the host execute a compile callback.
class RequestCompile
    : public shared::RPCFunction<
          RequestCompile, JITTargetAddress(JITTargetAddress TrampolineAddr)> {
public:
  static const char *getName() { return "RequestCompile"; }
};

  /// Notify the remote and terminate the session.
class TerminateSession : public shared::RPCFunction<TerminateSession, void()> {
public:
  static const char *getName() { return "TerminateSession"; }
};

} // namespace utils

class OrcRemoteTargetRPCAPI
    : public shared::SingleThreadedRPCEndpoint<shared::RawByteChannel> {
public:
  // FIXME: Remove constructors once MSVC supports synthesizing move-ops.
  OrcRemoteTargetRPCAPI(shared::RawByteChannel &C)
      : shared::SingleThreadedRPCEndpoint<shared::RawByteChannel>(C, true) {}
};

} // end namespace remote

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETRPCAPI_H
