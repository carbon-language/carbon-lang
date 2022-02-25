//===- EPCGenericMemoryAccessTest.cpp -- Tests for EPCGenericMemoryAccess -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

template <typename WriteT, typename SPSWriteT>
llvm::orc::shared::detail::CWrapperFunctionResult
testWriteUInts(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSWriteT>)>::handle(
             ArgData, ArgSize,
             [](std::vector<WriteT> Ws) {
               for (auto &W : Ws)
                 *jitTargetAddressToPointer<decltype(W.Value) *>(W.Address) =
                     W.Value;
             })
      .release();
}

llvm::orc::shared::detail::CWrapperFunctionResult
testWriteBuffers(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSMemoryAccessBufferWrite>)>::handle(
             ArgData, ArgSize,
             [](std::vector<tpctypes::BufferWrite> Ws) {
               for (auto &W : Ws)
                 memcpy(jitTargetAddressToPointer<char *>(W.Address),
                        W.Buffer.data(), W.Buffer.size());
             })
      .release();
}

TEST(EPCGenericMemoryAccessTest, MemWrites) {
  auto SelfEPC = cantFail(SelfExecutorProcessControl::Create());

  EPCGenericMemoryAccess::FuncAddrs FAs;
  FAs.WriteUInt8s = ExecutorAddress::fromPtr(
      &testWriteUInts<tpctypes::UInt8Write, SPSMemoryAccessUInt8Write>);
  FAs.WriteUInt16s = ExecutorAddress::fromPtr(
      &testWriteUInts<tpctypes::UInt16Write, SPSMemoryAccessUInt16Write>);
  FAs.WriteUInt32s = ExecutorAddress::fromPtr(
      &testWriteUInts<tpctypes::UInt32Write, SPSMemoryAccessUInt32Write>);
  FAs.WriteUInt64s = ExecutorAddress::fromPtr(
      &testWriteUInts<tpctypes::UInt64Write, SPSMemoryAccessUInt64Write>);
  FAs.WriteBuffers = ExecutorAddress::fromPtr(&testWriteBuffers);

  auto MemAccess = std::make_unique<EPCGenericMemoryAccess>(*SelfEPC, FAs);

  uint8_t Test_UInt8_1 = 0;
  uint8_t Test_UInt8_2 = 0;
  uint16_t Test_UInt16 = 0;
  uint32_t Test_UInt32 = 0;
  uint64_t Test_UInt64 = 0;
  char Test_Buffer[21];

  auto Err1 = MemAccess->writeUInt8s(
      {{pointerToJITTargetAddress(&Test_UInt8_1), 1},
       {pointerToJITTargetAddress(&Test_UInt8_2), 0xFE}});

  EXPECT_THAT_ERROR(std::move(Err1), Succeeded());
  EXPECT_EQ(Test_UInt8_1, 1U);
  EXPECT_EQ(Test_UInt8_2, 0xFE);

  auto Err2 =
      MemAccess->writeUInt16s({{pointerToJITTargetAddress(&Test_UInt16), 1}});
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());
  EXPECT_EQ(Test_UInt16, 1U);

  auto Err3 =
      MemAccess->writeUInt32s({{pointerToJITTargetAddress(&Test_UInt32), 1}});
  EXPECT_THAT_ERROR(std::move(Err3), Succeeded());
  EXPECT_EQ(Test_UInt32, 1U);

  auto Err4 =
      MemAccess->writeUInt64s({{pointerToJITTargetAddress(&Test_UInt64), 1}});
  EXPECT_THAT_ERROR(std::move(Err4), Succeeded());
  EXPECT_EQ(Test_UInt64, 1U);

  StringRef TestMsg("test-message");
  auto Err5 = MemAccess->writeBuffers(
      {{pointerToJITTargetAddress(&Test_Buffer), TestMsg}});
  EXPECT_THAT_ERROR(std::move(Err5), Succeeded());
  EXPECT_EQ(StringRef(Test_Buffer, TestMsg.size()), TestMsg);
}

} // namespace
