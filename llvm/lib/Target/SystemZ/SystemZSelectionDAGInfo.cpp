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
