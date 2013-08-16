//===-- SystemZSelectionDAGInfo.cpp - SystemZ SelectionDAG Info -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-selectiondag-info"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"

using namespace llvm;

SystemZSelectionDAGInfo::
SystemZSelectionDAGInfo(const SystemZTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

SystemZSelectionDAGInfo::~SystemZSelectionDAGInfo() {
}

SDValue SystemZSelectionDAGInfo::
EmitTargetCodeForMemcpy(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Dst, SDValue Src, SDValue Size, unsigned Align,
                        bool IsVolatile, bool AlwaysInline,
                        MachinePointerInfo DstPtrInfo,
                        MachinePointerInfo SrcPtrInfo) const {
  if (IsVolatile)
    return SDValue();

  if (ConstantSDNode *CSize = dyn_cast<ConstantSDNode>(Size)) {
    uint64_t Bytes = CSize->getZExtValue();
    if (Bytes >= 1 && Bytes <= 0x100) {
      // A single MVC.
      return DAG.getNode(SystemZISD::MVC, DL, MVT::Other,
                         Chain, Dst, Src, Size);
    }
  }
  return SDValue();
}

// Handle a memset of 1, 2, 4 or 8 bytes with the operands given by
// Chain, Dst, ByteVal and Size.  These cases are expected to use
// MVI, MVHHI, MVHI and MVGHI respectively.
static SDValue memsetStore(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                           SDValue Dst, uint64_t ByteVal, uint64_t Size,
                           unsigned Align,
                           MachinePointerInfo DstPtrInfo) {
  uint64_t StoreVal = ByteVal;
  for (unsigned I = 1; I < Size; ++I)
    StoreVal |= ByteVal << (I * 8);
  return DAG.getStore(Chain, DL,
                      DAG.getConstant(StoreVal, MVT::getIntegerVT(Size * 8)),
                      Dst, DstPtrInfo, false, false, Align);
}

SDValue SystemZSelectionDAGInfo::
EmitTargetCodeForMemset(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Dst, SDValue Byte, SDValue Size,
                        unsigned Align, bool IsVolatile,
                        MachinePointerInfo DstPtrInfo) const {
  EVT DstVT = Dst.getValueType();

  if (IsVolatile)
    return SDValue();

  if (ConstantSDNode *CSize = dyn_cast<ConstantSDNode>(Size)) {
    uint64_t Bytes = CSize->getZExtValue();
    if (Bytes == 0)
      return SDValue();
    if (ConstantSDNode *CByte = dyn_cast<ConstantSDNode>(Byte)) {
      // Handle cases that can be done using at most two of
      // MVI, MVHI, MVHHI and MVGHI.  The latter two can only be
      // used if ByteVal is all zeros or all ones; in other casees,
      // we can move at most 2 halfwords.
      uint64_t ByteVal = CByte->getZExtValue();
      if (ByteVal == 0 || ByteVal == 255 ?
          Bytes <= 16 && CountPopulation_64(Bytes) <= 2 :
          Bytes <= 4) {
        unsigned Size1 = Bytes == 16 ? 8 : 1 << findLastSet(Bytes);
        unsigned Size2 = Bytes - Size1;
        SDValue Chain1 = memsetStore(DAG, DL, Chain, Dst, ByteVal, Size1,
                                     Align, DstPtrInfo);
        if (Size2 == 0)
          return Chain1;
        Dst = DAG.getNode(ISD::ADD, DL, DstVT, Dst,
                          DAG.getConstant(Size1, DstVT));
        DstPtrInfo = DstPtrInfo.getWithOffset(Size1);
        SDValue Chain2 = memsetStore(DAG, DL, Chain, Dst, ByteVal, Size2,
                                     std::min(Align, Size1), DstPtrInfo);
        return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chain1, Chain2);
      }
    } else {
      // Handle one and two bytes using STC.
      if (Bytes <= 2) {
        SDValue Chain1 = DAG.getStore(Chain, DL, Byte, Dst, DstPtrInfo,
                                      false, false, Align);
        if (Bytes == 1)
          return Chain1;
        SDValue Dst2 = DAG.getNode(ISD::ADD, DL, DstVT, Dst,
                                   DAG.getConstant(1, DstVT));
        SDValue Chain2 = DAG.getStore(Chain, DL, Byte, Dst2,
                                      DstPtrInfo.getWithOffset(1),
                                      false, false, 1);
        return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chain1, Chain2);
      }
    }
    assert(Bytes >= 2 && "Should have dealt with 0- and 1-byte cases already");
    if (Bytes <= 0x101) {
      // Copy the byte to the first location and then use MVC to copy
      // it to the rest.
      Chain = DAG.getStore(Chain, DL, Byte, Dst, DstPtrInfo,
                           false, false, Align);
      SDValue Dst2 = DAG.getNode(ISD::ADD, DL, DstVT, Dst,
                                 DAG.getConstant(1, DstVT));
      return DAG.getNode(SystemZISD::MVC, DL, MVT::Other, Chain, Dst2, Dst,
                         DAG.getConstant(Bytes - 1, MVT::i32));
    }
  }
  return SDValue();
}

// Convert the current CC value into an integer that is 0 if CC == 0,
// less than zero if CC == 1 and greater than zero if CC >= 2.
// The sequence starts with IPM, which puts CC into bits 29 and 28
// of an integer and clears bits 30 and 31.
static SDValue addIPMSequence(SDLoc DL, SDValue Glue, SelectionDAG &DAG) {
  SDValue IPM = DAG.getNode(SystemZISD::IPM, DL, MVT::i32, Glue);
  SDValue SRL = DAG.getNode(ISD::SRL, DL, MVT::i32, IPM,
                            DAG.getConstant(28, MVT::i32));
  SDValue ROTL = DAG.getNode(ISD::ROTL, DL, MVT::i32, SRL,
                             DAG.getConstant(31, MVT::i32));
  return ROTL;
}

std::pair<SDValue, SDValue> SystemZSelectionDAGInfo::
EmitTargetCodeForMemcmp(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Src1, SDValue Src2, SDValue Size,
                        MachinePointerInfo Op1PtrInfo,
                        MachinePointerInfo Op2PtrInfo) const {
  if (ConstantSDNode *CSize = dyn_cast<ConstantSDNode>(Size)) {
    uint64_t Bytes = CSize->getZExtValue();
    if (Bytes >= 1 && Bytes <= 0x100) {
      // A single CLC.
      SDVTList VTs = DAG.getVTList(MVT::Other, MVT::Glue);
      Chain = DAG.getNode(SystemZISD::CLC, DL, VTs, Chain,
                          Src1, Src2, Size);
      SDValue Glue = Chain.getValue(1);
      return std::make_pair(addIPMSequence(DL, Glue, DAG), Chain);
    }
  }
  return std::make_pair(SDValue(), SDValue());
}

std::pair<SDValue, SDValue> SystemZSelectionDAGInfo::
EmitTargetCodeForStrcpy(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Dest, SDValue Src,
                        MachinePointerInfo DestPtrInfo,
                        MachinePointerInfo SrcPtrInfo, bool isStpcpy) const {
  SDVTList VTs = DAG.getVTList(Dest.getValueType(), MVT::Other);
  SDValue EndDest = DAG.getNode(SystemZISD::STPCPY, DL, VTs, Chain, Dest, Src,
                                DAG.getConstant(0, MVT::i32));
  return std::make_pair(isStpcpy ? EndDest : Dest, EndDest.getValue(1));
}

std::pair<SDValue, SDValue> SystemZSelectionDAGInfo::
EmitTargetCodeForStrcmp(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Src1, SDValue Src2,
                        MachinePointerInfo Op1PtrInfo,
                        MachinePointerInfo Op2PtrInfo) const {
  SDVTList VTs = DAG.getVTList(Src1.getValueType(), MVT::Other, MVT::Glue);
  SDValue Unused = DAG.getNode(SystemZISD::STRCMP, DL, VTs, Chain, Src1, Src2,
                               DAG.getConstant(0, MVT::i32));
  Chain = Unused.getValue(1);
  SDValue Glue = Chain.getValue(2);
  return std::make_pair(addIPMSequence(DL, Glue, DAG), Chain);
}

// Search from Src for a null character, stopping once Src reaches Limit.
// Return a pair of values, the first being the number of nonnull characters
// and the second being the out chain.
//
// This can be used for strlen by setting Limit to 0.
static std::pair<SDValue, SDValue> getBoundedStrlen(SelectionDAG &DAG, SDLoc DL,
                                                    SDValue Chain, SDValue Src,
                                                    SDValue Limit) {
  EVT PtrVT = Src.getValueType();
  SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other, MVT::Glue);
  SDValue End = DAG.getNode(SystemZISD::SEARCH_STRING, DL, VTs, Chain,
                            Limit, Src, DAG.getConstant(0, MVT::i32));
  Chain = End.getValue(1);
  SDValue Len = DAG.getNode(ISD::SUB, DL, PtrVT, End, Src);
  return std::make_pair(Len, Chain);
}    

std::pair<SDValue, SDValue> SystemZSelectionDAGInfo::
EmitTargetCodeForStrlen(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Src, MachinePointerInfo SrcPtrInfo) const {
  EVT PtrVT = Src.getValueType();
  return getBoundedStrlen(DAG, DL, Chain, Src, DAG.getConstant(0, PtrVT));
}

std::pair<SDValue, SDValue> SystemZSelectionDAGInfo::
EmitTargetCodeForStrnlen(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                         SDValue Src, SDValue MaxLength,
                         MachinePointerInfo SrcPtrInfo) const {
  EVT PtrVT = Src.getValueType();
  MaxLength = DAG.getZExtOrTrunc(MaxLength, DL, PtrVT);
  SDValue Limit = DAG.getNode(ISD::ADD, DL, PtrVT, Src, MaxLength);
  return getBoundedStrlen(DAG, DL, Chain, Src, Limit);
}
