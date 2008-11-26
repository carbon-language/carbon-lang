//===-- PIC16ISelLowering.cpp - PIC16 DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PIC16 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-lower"

#include "PIC16ISelLowering.h"
#include "PIC16TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"


using namespace llvm;


// PIC16TargetLowering Constructor.
PIC16TargetLowering::PIC16TargetLowering(PIC16TargetMachine &TM)
  : TargetLowering(TM) {
  
  Subtarget = &TM.getSubtarget<PIC16Subtarget>();

  addRegisterClass(MVT::i8, PIC16::GPRRegisterClass);

  setShiftAmountType(MVT::i8);
  setShiftAmountFlavor(Extend);


  setOperationAction(ISD::GlobalAddress, MVT::i16, Custom);

  setOperationAction(ISD::LOAD,   MVT::i8,  Legal);
  setOperationAction(ISD::LOAD,   MVT::i16, Custom);
  setOperationAction(ISD::LOAD,   MVT::i32, Custom);

  setOperationAction(ISD::STORE,  MVT::i8,  Legal);
  setOperationAction(ISD::STORE,  MVT::i16, Custom);
  setOperationAction(ISD::STORE,  MVT::i32, Custom);

  setOperationAction(ISD::ADDE,    MVT::i8,  Custom);
  setOperationAction(ISD::ADDC,    MVT::i8,  Custom);
  setOperationAction(ISD::SUBE,    MVT::i8,  Custom);
  setOperationAction(ISD::SUBC,    MVT::i8,  Custom);
  setOperationAction(ISD::ADD,    MVT::i8,  Legal);
  setOperationAction(ISD::ADD,    MVT::i16, Custom);

  setOperationAction(ISD::OR,     MVT::i8,  Custom);
  setOperationAction(ISD::AND,    MVT::i8,  Custom);
  setOperationAction(ISD::XOR,    MVT::i8,  Custom);

  setOperationAction(ISD::SHL,    MVT::i16, Custom);
  setOperationAction(ISD::SHL,    MVT::i32, Custom);

  //setOperationAction(ISD::TRUNCATE, MVT::i16, Custom);
  setTruncStoreAction(MVT::i16,   MVT::i8,  Custom);

  // Now deduce the information based on the above mentioned 
  // actions
  computeRegisterProperties();
}

const char *PIC16TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default:                         return NULL;
  case PIC16ISD::Lo:               return "PIC16ISD::Lo";
  case PIC16ISD::Hi:               return "PIC16ISD::Hi";
  case PIC16ISD::MTLO:             return "PIC16ISD::MTLO";
  case PIC16ISD::MTHI:             return "PIC16ISD::MTHI";
  case PIC16ISD::Banksel:          return "PIC16ISD::Banksel";
  case PIC16ISD::PIC16Load:        return "PIC16ISD::PIC16Load";
  case PIC16ISD::PIC16Store:       return "PIC16ISD::PIC16Store";
  case PIC16ISD::BCF:              return "PIC16ISD::BCF";
  case PIC16ISD::LSLF:             return "PIC16ISD::LSLF";
  case PIC16ISD::LRLF:             return "PIC16ISD::LRLF";
  case PIC16ISD::RLF:              return "PIC16ISD::RLF";
  case PIC16ISD::RRF:              return "PIC16ISD::RRF";
  case PIC16ISD::Dummy:            return "PIC16ISD::Dummy";
  }
}

SDNode *PIC16TargetLowering::ReplaceNodeResults(SDNode *N, SelectionDAG &DAG) {
  switch (N->getOpcode()) {
    case ISD::GlobalAddress:
      return ExpandGlobalAddress(N, DAG);
    case ISD::STORE:
      return ExpandStore(N, DAG);
    case ISD::LOAD:
      return ExpandLoad(N, DAG);
    case ISD::ADD:
      return ExpandAdd(N, DAG);
    case ISD::SHL:
      return ExpandShift(N, DAG);
    default:
      assert (0 && "not implemented");
  }
}

SDNode *PIC16TargetLowering::ExpandStore(SDNode *N, SelectionDAG &DAG) { 
  StoreSDNode *St = cast<StoreSDNode>(N);
  SDValue Chain = St->getChain();
  SDValue Src = St->getValue();
  SDValue Ptr = St->getBasePtr();
  MVT ValueType = Src.getValueType();
  unsigned StoreOffset = 0;

  SDValue PtrLo, PtrHi;
  LegalizeAddress(Ptr, DAG, PtrLo, PtrHi, StoreOffset);
 
  if (ValueType == MVT::i8) {
    SDValue Store = DAG.getNode (PIC16ISD::PIC16Store, MVT::Other, Chain, Src,
                                 PtrLo, PtrHi, DAG.getConstant (0, MVT::i8));
    return Store.getNode();
  }
  else if (ValueType == MVT::i16) {
    // Get the Lo and Hi parts from MERGE_VALUE or BUILD_PAIR.
    SDValue SrcLo, SrcHi;
    GetExpandedParts(Src, DAG, SrcLo, SrcHi);
    SDValue ChainLo = Chain, ChainHi = Chain;
    if (Chain.getOpcode() == ISD::TokenFactor) {
      ChainLo = Chain.getOperand(0);
      ChainHi = Chain.getOperand(1);
    }
    SDValue Store1 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other,
                                 ChainLo,
                                 SrcLo, PtrLo, PtrHi,
                                 DAG.getConstant (0 + StoreOffset, MVT::i8));

    SDValue Store2 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainHi, 
                                 SrcHi, PtrLo, PtrHi,
                                 DAG.getConstant (1 + StoreOffset, MVT::i8));

    return DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(Store1),
                       getChain(Store2)).getNode();
  }
  else if (ValueType == MVT::i32) {
    // Get the Lo and Hi parts from MERGE_VALUE or BUILD_PAIR.
    SDValue SrcLo, SrcHi;
    GetExpandedParts(Src, DAG, SrcLo, SrcHi);

    // Get the expanded parts of each of SrcLo and SrcHi.
    SDValue SrcLo1, SrcLo2, SrcHi1, SrcHi2;
    GetExpandedParts(SrcLo, DAG, SrcLo1, SrcLo2);
    GetExpandedParts(SrcHi, DAG, SrcHi1, SrcHi2);

    SDValue ChainLo = Chain, ChainHi = Chain;
    if (Chain.getOpcode() == ISD::TokenFactor) {  
      ChainLo = Chain.getOperand(0);
      ChainHi = Chain.getOperand(1);
    }
    SDValue ChainLo1 = ChainLo, ChainLo2 = ChainLo, ChainHi1 = ChainHi,
            ChainHi2 = ChainHi;
    if (ChainLo.getOpcode() == ISD::TokenFactor) {
      ChainLo1 = ChainLo.getOperand(0);
      ChainLo2 = ChainLo.getOperand(1);
    }
    if (ChainHi.getOpcode() == ISD::TokenFactor) {
      ChainHi1 = ChainHi.getOperand(0);
      ChainHi2 = ChainHi.getOperand(1);
    }
    SDValue Store1 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other,
                                 ChainLo1,
                                 SrcLo1, PtrLo, PtrHi,
                                 DAG.getConstant (0 + StoreOffset, MVT::i8));

    SDValue Store2 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainLo2,
                                 SrcLo2, PtrLo, PtrHi,
                                 DAG.getConstant (1 + StoreOffset, MVT::i8));

    SDValue Store3 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainHi1,
                                 SrcHi1, PtrLo, PtrHi,
                                 DAG.getConstant (2 + StoreOffset, MVT::i8));

    SDValue Store4 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainHi2,
                                 SrcHi2, PtrLo, PtrHi,
                                 DAG.getConstant (3 + StoreOffset, MVT::i8));

    SDValue RetLo =  DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(Store1),
                                 getChain(Store2));
    SDValue RetHi =  DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(Store3),
                                getChain(Store4));
    return  DAG.getNode(ISD::TokenFactor, MVT::Other, RetLo, RetHi).getNode();

  }
  else {
    assert (0 && "value type not supported");
  }
}

// ExpandGlobalAddress - 
SDNode *PIC16TargetLowering::ExpandGlobalAddress(SDNode *N, SelectionDAG &DAG) {
  GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(SDValue(N, 0));
  
  SDValue TGA = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i8,
                                           G->getOffset());

  SDValue Lo = DAG.getNode(PIC16ISD::Lo, MVT::i8, TGA);
  SDValue Hi = DAG.getNode(PIC16ISD::Hi, MVT::i8, TGA);

  SDValue BP = DAG.getNode(ISD::BUILD_PAIR, MVT::i16, Lo, Hi);
  return BP.getNode();
}

bool PIC16TargetLowering::isDirectAddress(const SDValue &Op) {
  assert (Op.getNode() != NULL && "Can't operate on NULL SDNode!!");

  if (Op.getOpcode() == ISD::BUILD_PAIR) {
   if (Op.getOperand(0).getOpcode() == PIC16ISD::Lo) 
     return true;
  }
  return false;
}

// Return true if DirectAddress is in ROM_SPACE
bool PIC16TargetLowering::isRomAddress(const SDValue &Op) {

  // RomAddress is a GlobalAddress in ROM_SPACE_
  // If the Op is not a GlobalAddress return NULL without checking
  // anything further.
  if (!isDirectAddress(Op))
    return false; 

  // Its a GlobalAddress.
  // It is BUILD_PAIR((PIC16Lo TGA), (PIC16Hi TGA)) and Op is BUILD_PAIR
  SDValue TGA = Op.getOperand(0).getOperand(0);
  GlobalAddressSDNode *GSDN = dyn_cast<GlobalAddressSDNode>(TGA);
  const Type *ValueType = GSDN->getGlobal()->getType();

  if (!isa<PointerType>(ValueType)) {
    assert(0 && "TGA must be of a PointerType");
  }

  int AddrSpace = dyn_cast<PointerType>(ValueType)->getAddressSpace();
  if (AddrSpace == PIC16ISD::ROM_SPACE)
    return true;

  // Any other address space return it false
  return false;
}

// To extract chain value from the SDValue Nodes
// This function will help to maintain the chain extracting
// code at one place. In case of any change in future it will
// help maintain the code.
SDValue PIC16TargetLowering::getChain(SDValue &Op) { 
  SDValue Chain = Op.getValue(Op.getNode()->getNumValues() - 1);

  // All nodes may not produce a chain. Therefore following assert
  // verifies that the node is returning a chain only.
  assert (Chain.getValueType() == MVT::Other && "Node does not have a chain");

  return Chain;
}

void PIC16TargetLowering::GetExpandedParts(SDValue Op, SelectionDAG &DAG,
                                           SDValue &Lo, SDValue &Hi) {  
  SDNode *N = Op.getNode();
  unsigned NumValues = N->getNumValues();
  std::vector<MVT> VTs;
  MVT NewVT;
  std::vector<SDValue> Opers;

  // EXTRACT_ELEMENT should have same number and type of values that the 
  // node replacing the EXTRACT_ELEMENT should have. (i.e. extracted element)
  // Some nodes such as LOAD and PIC16Load have more than one values. In such 
  // cases EXTRACT_ELEMENT should have more than one values. Therefore creating
  // vector of Values for EXTRACT_ELEMENT. This list will have same number of 
  // values as the extracted element will have.

  for (unsigned i=0;i < NumValues; ++i) {
    NewVT = getTypeToTransformTo(N->getValueType(i));
    VTs.push_back(NewVT);
  }

  // extract the lo component
  Opers.push_back(Op);
  Opers.push_back(DAG.getConstant(0,MVT::i8));
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT,VTs,&Opers[0],Opers.size());

  // extract the hi component
  Opers.clear();
  Opers.push_back(Op);
  Opers.push_back(DAG.getConstant(1,MVT::i8));
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT,VTs,&Opers[0],Opers.size());
}

// This function legalizes the PIC16 Addresses. If the Pointer is  
//  -- Direct address variable residing 
//     --> then a Banksel for that variable will be created.
//  -- Rom variable            
//     --> then it will be treated as an indirect address.
//  -- Indirect address 
//     --> then the address will be loaded into FSR
//  -- ADD with constant operand
//     --> then constant operand of ADD will be returned as Offset
//         and non-constant operand of ADD will be treated as pointer.
// Returns the high and lo part of the address, and the offset(in case of ADD).

void PIC16TargetLowering:: LegalizeAddress(SDValue Ptr, SelectionDAG &DAG, 
                                           SDValue &Lo, SDValue &Hi,
                                           unsigned &Offset) {

  // Offset, by default, should be 0
  Offset = 0;

  // If the pointer is ADD with constant,
  // return the constant value as the offset  
  if (Ptr.getOpcode() == ISD::ADD) {
    SDValue OperLeft = Ptr.getOperand(0);
    SDValue OperRight = Ptr.getOperand(1);
    if (OperLeft.getOpcode() == ISD::Constant) {
      Offset = dyn_cast<ConstantSDNode>(OperLeft)->getZExtValue();
      Ptr = OperRight;
    } else {
      Ptr = OperLeft;
      Offset = dyn_cast<ConstantSDNode>(OperRight)->getZExtValue();
    }
  }

  if (isDirectAddress(Ptr) && !isRomAddress(Ptr)) {
    // Direct addressing case for RAM variables. The Hi part is constant
    // and the Lo part is the TGA itself.
    Lo = Ptr.getOperand(0).getOperand(0);

    // For direct addresses Hi is a constant. Value 1 for the constant
    // signifies that banksel needs to generated for it. Value 0 for
    // the constant signifies that banksel does not need to be generated 
    // for it. Mark it as 1 now and optimize later. 
    Hi = DAG.getConstant(1, MVT::i8);
    return; 
  }

  // Indirect addresses. Get the hi and lo parts of ptr. 
  GetExpandedParts(Ptr, DAG, Lo, Hi);

  // Put the hi and lo parts into FSR.
  Lo = DAG.getNode(PIC16ISD::MTLO, MVT::i8, Lo);
  Hi = DAG.getNode(PIC16ISD::MTHI, MVT::i8, Hi);

  return;
}

SDNode *PIC16TargetLowering::ExpandAdd(SDNode *N, SelectionDAG &DAG) {
  SDValue OperLeft = N->getOperand(0);
  SDValue OperRight = N->getOperand(1);

  if((OperLeft.getOpcode() == ISD::Constant) || 
     (OperRight.getOpcode() == ISD::Constant)) { 
    return NULL;
  }

  // These case are yet to be handled
  return NULL;
}

SDNode *PIC16TargetLowering::ExpandLoad(SDNode *N, SelectionDAG &DAG) {
  LoadSDNode *LD = dyn_cast<LoadSDNode>(SDValue(N, 0));
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();

  SDValue Load, Offset;
  SDVTList Tys; 
  MVT VT, NewVT;
  SDValue PtrLo, PtrHi;
  unsigned LoadOffset;

  // Legalize direct/indirect addresses. This will give the lo and hi parts
  // of the address and the offset.
  LegalizeAddress(Ptr, DAG, PtrLo, PtrHi, LoadOffset);

  // Load from the pointer (direct address or FSR) 
  VT = N->getValueType(0);
  unsigned NumLoads = VT.getSizeInBits() / 8; 
  std::vector<SDValue> PICLoads;
  unsigned iter;
  MVT MemVT = LD->getMemoryVT();
  if(ISD::isNON_EXTLoad(N)) {
    for (iter=0; iter<NumLoads ; ++iter) {
      // Add the pointer offset if any
      Offset = DAG.getConstant(iter + LoadOffset, MVT::i8);
      Tys = DAG.getVTList(MVT::i8, MVT::Other); 
      Load = DAG.getNode(PIC16ISD::PIC16Load, Tys, Chain, PtrLo, PtrHi,
                         Offset); 
      PICLoads.push_back(Load);
    }
  } else {
    // If it is extended load then use PIC16Load for Memory Bytes
    // and for all extended bytes perform action based on type of
    // extention - i.e. SignExtendedLoad or ZeroExtendedLoad

    
    // For extended loads this is the memory value type
    // i.e. without any extension
    MVT MemVT = LD->getMemoryVT();
    unsigned MemBytes = MemVT.getSizeInBits() / 8;
    unsigned ExtdBytes = VT.getSizeInBits() / 8;
    Offset = DAG.getConstant(LoadOffset, MVT::i8);

    Tys = DAG.getVTList(MVT::i8, MVT::Other); 
    // For MemBytes generate PIC16Load with proper offset
    for (iter=0; iter<MemBytes; ++iter) {
      // Add the pointer offset if any
      Offset = DAG.getConstant(iter + LoadOffset, MVT::i8);
      Load = DAG.getNode(PIC16ISD::PIC16Load, Tys, Chain, PtrLo, PtrHi,
                         Offset); 
      PICLoads.push_back(Load);
    }

    // For SignExtendedLoad
    if (ISD::isSEXTLoad(N)) {
      // For all ExtdBytes use the Right Shifted(Arithmetic) Value of the 
      // highest MemByte
      SDValue SRA = DAG.getNode(ISD::SRA, MVT::i8, Load, 
                                DAG.getConstant(7, MVT::i8));
      for (iter=MemBytes; iter<ExtdBytes; ++iter) { 
        PICLoads.push_back(SRA);
      }
    } else if (ISD::isZEXTLoad(N)) {
      // ZeroExtendedLoad -- For all ExtdBytes use constant 0
      SDValue ConstZero = DAG.getConstant(0, MVT::i8);
      for (iter=MemBytes; iter<ExtdBytes; ++iter) { 
        PICLoads.push_back(ConstZero);
      }
    }
  }
  SDValue BP;

  if (VT == MVT::i8) {
    // Operand of Load is illegal -- Load itself is legal
    return PICLoads[0].getNode();
  }
  else if (VT == MVT::i16) {
    BP = DAG.getNode(ISD::BUILD_PAIR, VT, PICLoads[0], PICLoads[1]);
    if (MemVT == MVT::i8)
      Chain = getChain(PICLoads[0]);
    else
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(PICLoads[0]),
                          getChain(PICLoads[1]));
  } else if (VT == MVT::i32) {
    SDValue BPs[2];
    BPs[0] = DAG.getNode(ISD::BUILD_PAIR, MVT::i16, PICLoads[0], PICLoads[1]);
    BPs[1] = DAG.getNode(ISD::BUILD_PAIR, MVT::i16, PICLoads[2], PICLoads[3]);
    BP = DAG.getNode(ISD::BUILD_PAIR, VT, BPs[0], BPs[1]);
    if (MemVT == MVT::i8)
      Chain = getChain(PICLoads[0]);
    else if (MemVT == MVT::i16)
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(PICLoads[0]),
                          getChain(PICLoads[1]));
    else {
      SDValue Chains[2];
      Chains[0] = DAG.getNode(ISD::TokenFactor, MVT::Other,
                              getChain(PICLoads[0]), getChain(PICLoads[1]));
      Chains[1] = DAG.getNode(ISD::TokenFactor, MVT::Other,
                              getChain(PICLoads[2]), getChain(PICLoads[3]));
      Chain =  DAG.getNode(ISD::TokenFactor, MVT::Other, Chains[0], Chains[1]);
    }
  }
  Tys = DAG.getVTList(VT, MVT::Other); 
  SDValue MergeV = DAG.getNode(ISD::MERGE_VALUES, Tys, BP, Chain);
  return MergeV.getNode();

}

SDNode *PIC16TargetLowering::ExpandShift(SDNode *N, SelectionDAG &DAG) {
  SDValue Value = N->getOperand(0);
  SDValue Amt = N->getOperand(1);
  SDValue BCF, BCFInput;
  SDVTList Tys; 
  SDValue ShfCom;   // Shift Component - Lo component should be shifted
  SDValue RotCom;   // Rotate Component- Hi component should be rotated
  PIC16ISD::NodeType ShfNode, RotNode; 
  
  // Currently handling Constant shift only
  if (Amt.getOpcode() != ISD::Constant)
    return NULL;

  // Following code considers 16 bit left-shift only
  if (N->getValueType(0) != MVT::i16)
    return NULL;

  if (N->getOpcode() == ISD::SHL) {
    ShfNode = PIC16ISD::LSLF;
    RotNode = PIC16ISD::RLF;
  } else if (N->getOpcode() == ISD::SRL) {
    ShfNode = PIC16ISD::LRLF;
    RotNode = PIC16ISD::RRF;
  }
  unsigned ShiftAmt = dyn_cast<ConstantSDNode>(Amt)->getZExtValue();
  SDValue StatusReg = DAG.getRegister(PIC16::STATUS, MVT::i8);
  // 0th Bit in StatusReg is CarryBit 
  SDValue CarryBit= DAG.getConstant(0, MVT::i8);

  GetExpandedParts(Value, DAG, ShfCom, RotCom);
  BCFInput = DAG.getNode(PIC16ISD::Dummy, MVT::Flag); 
  Tys = DAG.getVTList(MVT::i8, MVT::Flag);

  for (unsigned i=0;i<ShiftAmt;i++) {
    BCF = DAG.getNode(PIC16ISD::BCF, MVT::Flag, StatusReg, CarryBit, BCFInput);

    // Following are Two-Address Instructions
    ShfCom = DAG.getNode(ShfNode, Tys, ShfCom, BCF);
    RotCom = DAG.getNode(RotNode, Tys, RotCom, ShfCom.getValue(1));

    BCFInput = RotCom.getValue(1); 
  }

  SDValue BP = DAG.getNode(ISD::BUILD_PAIR, N->getValueType(0), ShfCom, RotCom);
  return BP.getNode();
}

SDValue PIC16TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
    case ISD::FORMAL_ARGUMENTS:
      return LowerFORMAL_ARGUMENTS(Op, DAG);
    case ISD::ADDC:
      return LowerADDC(Op, DAG);
    case ISD::ADDE:
      return LowerADDE(Op, DAG);
    case ISD::SUBE:
      return LowerSUBE(Op, DAG);
    case ISD::SUBC:
      return LowerSUBC(Op, DAG);
    case ISD::LOAD:
      return SDValue(ExpandLoad(Op.getNode(), DAG), Op.getResNo());
    case ISD::STORE:
      return SDValue(ExpandStore(Op.getNode(), DAG), Op.getResNo());
    case ISD::SHL:
      return SDValue(ExpandShift(Op.getNode(), DAG), Op.getResNo());
    case ISD::OR:
    case ISD::AND:
    case ISD::XOR:
      return LowerBinOp(Op, DAG);
  }
  return SDValue();
}

SDValue PIC16TargetLowering::ConvertToMemOperand(SDValue Op,
                                                 SelectionDAG &DAG) {

  assert (Op.getValueType() == MVT::i8 
          && "illegal value type to store on stack.");

  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string FuncName = Func->getName();

  char *tmpName = new char [strlen(FuncName.c_str()) +  6];

  // Put the value on stack.
  // Get a stack slot index and convert to es.
  int FI = MF.getFrameInfo()->CreateStackObject(1, 1);
  sprintf(tmpName, "%s.tmp", FuncName.c_str());
  SDValue ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);

  // Store the value to ES.
  SDValue Store = DAG.getNode (PIC16ISD::PIC16Store, MVT::Other,
                               DAG.getEntryNode(),
                               Op, ES, 
                               DAG.getConstant (1, MVT::i8), // Banksel.
                               DAG.getConstant (FI, MVT::i8));

  // Load the value from ES.
  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other);
  SDValue Load = DAG.getNode(PIC16ISD::PIC16Load, Tys, Store,
                             ES, DAG.getConstant (1, MVT::i8),
                             DAG.getConstant (FI, MVT::i8));
    
  return Load.getValue(0);
}

SDValue PIC16TargetLowering:: LowerBinOp(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal Op to lower");

  // Return the original Op if the one of the operands is already a load.
  if (Op.getOperand(0).getOpcode() == PIC16ISD::PIC16Load
      || Op.getOperand(1).getOpcode() == PIC16ISD::PIC16Load)
    return Op;

  // Put one value on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(1), DAG);

  return DAG.getNode(Op.getOpcode(), MVT::i8, Op.getOperand(0), NewVal);
}

SDValue PIC16TargetLowering:: LowerADDC(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal addc to lower");

  // Nothing to do if the one of the operands is already a load.
  if (Op.getOperand(0).getOpcode() == PIC16ISD::PIC16Load 
      || Op.getOperand(1).getOpcode() == PIC16ISD::PIC16Load)
    return SDValue();

  // Put one value on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(1), DAG);
    
  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);
  return DAG.getNode(ISD::ADDC, Tys, Op.getOperand(0), NewVal);
}

SDValue PIC16TargetLowering:: LowerADDE(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal adde to lower");

  // Nothing to do if the one of the operands is already a load.
  if (Op.getOperand(0).getOpcode() == PIC16ISD::PIC16Load 
      || Op.getOperand(1).getOpcode() == PIC16ISD::PIC16Load)
    return SDValue();

  // Put one value on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(1), DAG);
    
  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);
  return DAG.getNode(ISD::ADDE, Tys, Op.getOperand(0), NewVal, 
                     Op.getOperand(2));
}

SDValue PIC16TargetLowering:: LowerSUBC(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal subc to lower");

  // Nothing to do if the first operand is already a load.
  if (Op.getOperand(0).getOpcode() == PIC16ISD::PIC16Load)
    return SDValue();

  // Put first operand on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(0), DAG);

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);
  return DAG.getNode(ISD::SUBC, Tys, NewVal, Op.getOperand(1));
}

SDValue PIC16TargetLowering:: LowerSUBE(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal sube to lower");

  // Nothing to do if the first operand is already a load.
  if (Op.getOperand(0).getOpcode() == PIC16ISD::PIC16Load)
    return SDValue();

  // Put first operand on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(0), DAG);

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);
  return DAG.getNode(ISD::SUBE, Tys, NewVal, Op.getOperand(1),
                     Op.getOperand(2));
}

// LowerFORMAL_ARGUMENTS - In Lowering FORMAL ARGUMENTS - MERGE_VALUES nodes
// is returned. MERGE_VALUES nodes number of operands and number of values are
// equal. Therefore to construct MERGE_VALUE node, UNDEF nodes equal to the
// number of arguments of function have been created.

SDValue PIC16TargetLowering:: LowerFORMAL_ARGUMENTS(SDValue Op, 
                                                    SelectionDAG &DAG) {
  SmallVector<SDValue, 8> ArgValues;
  unsigned NumArgs = Op.getNumOperands() - 3;

  // Creating UNDEF nodes to meet the requirement of MERGE_VALUES node.
  for(unsigned i = 0 ; i<NumArgs ; i++) {
    SDValue TempNode = DAG.getNode(ISD::UNDEF, Op.getNode()->getValueType(i));
    ArgValues.push_back(TempNode);
  }

  ArgValues.push_back(Op.getOperand(0));
  return DAG.getNode(ISD::MERGE_VALUES, Op.getNode()->getVTList(), 
                     &ArgValues[0],
                     ArgValues.size()).getValue(Op.getResNo());
}

// Perform DAGCombine of PIC16Load 
SDValue PIC16TargetLowering::
PerformPIC16LoadCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Chain = N->getOperand(0); 
  if (N->hasNUsesOfValue(0, 0)) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N,1), Chain);
  }
  return SDValue();
}


SDValue PIC16TargetLowering::PerformDAGCombine(SDNode *N, 
                                               DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case PIC16ISD::PIC16Load:
    return PerformPIC16LoadCombine(N, DCI);
  }
  return SDValue();
}
