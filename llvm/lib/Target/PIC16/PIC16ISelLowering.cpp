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
#include "PIC16TargetObjectFile.h"
#include "PIC16TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Function.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"


using namespace llvm;

static const char *getIntrinsicName(unsigned opcode) {
  std::string Basename;
  switch(opcode) {
  default: llvm_unreachable("do not know intrinsic name");
  // Arithmetic Right shift for integer types.
  case PIC16ISD::SRA_I8: Basename = "sra.i8"; break;
  case RTLIB::SRA_I16: Basename = "sra.i16"; break;
  case RTLIB::SRA_I32: Basename = "sra.i32"; break;

  // Left shift for integer types.
  case PIC16ISD::SLL_I8: Basename = "sll.i8"; break;
  case RTLIB::SHL_I16: Basename = "sll.i16"; break;
  case RTLIB::SHL_I32: Basename = "sll.i32"; break;

  // Logical Right Shift for integer types.
  case PIC16ISD::SRL_I8: Basename = "srl.i8"; break;
  case RTLIB::SRL_I16: Basename = "srl.i16"; break;
  case RTLIB::SRL_I32: Basename = "srl.i32"; break;

  // Multiply for integer types.
  case PIC16ISD::MUL_I8: Basename = "mul.i8"; break;
  case RTLIB::MUL_I16: Basename = "mul.i16"; break;
  case RTLIB::MUL_I32: Basename = "mul.i32"; break;

  // Signed division for integers.
  case RTLIB::SDIV_I16: Basename = "sdiv.i16"; break;
  case RTLIB::SDIV_I32: Basename = "sdiv.i32"; break;

  // Unsigned division for integers.
  case RTLIB::UDIV_I16: Basename = "udiv.i16"; break;
  case RTLIB::UDIV_I32: Basename = "udiv.i32"; break;

  // Signed Modulas for integers.
  case RTLIB::SREM_I16: Basename = "srem.i16"; break;
  case RTLIB::SREM_I32: Basename = "srem.i32"; break;

  // Unsigned Modulas for integers.
  case RTLIB::UREM_I16: Basename = "urem.i16"; break;
  case RTLIB::UREM_I32: Basename = "urem.i32"; break;

  //////////////////////
  // LIBCALLS FOR FLOATS
  //////////////////////

  // Float to signed integrals
  case RTLIB::FPTOSINT_F32_I8: Basename = "f32_to_si32"; break;
  case RTLIB::FPTOSINT_F32_I16: Basename = "f32_to_si32"; break;
  case RTLIB::FPTOSINT_F32_I32: Basename = "f32_to_si32"; break;

  // Signed integrals to float. char and int are first sign extended to i32 
  // before being converted to float, so an I8_F32 or I16_F32 isn't required.
  case RTLIB::SINTTOFP_I32_F32: Basename = "si32_to_f32"; break;

  // Float to Unsigned conversions.
  // Signed conversion can be used for unsigned conversion as well.
  // In signed and unsigned versions only the interpretation of the 
  // MSB is different. Bit representation remains the same. 
  case RTLIB::FPTOUINT_F32_I8: Basename = "f32_to_si32"; break;
  case RTLIB::FPTOUINT_F32_I16: Basename = "f32_to_si32"; break;
  case RTLIB::FPTOUINT_F32_I32: Basename = "f32_to_si32"; break;

  // Unsigned to Float conversions. char and int are first zero extended 
  // before being converted to float.
  case RTLIB::UINTTOFP_I32_F32: Basename = "ui32_to_f32"; break;
               
  // Floating point add, sub, mul, div.
  case RTLIB::ADD_F32: Basename = "add.f32"; break;
  case RTLIB::SUB_F32: Basename = "sub.f32"; break;
  case RTLIB::MUL_F32: Basename = "mul.f32"; break;
  case RTLIB::DIV_F32: Basename = "div.f32"; break;

  // Floating point comparison
  case RTLIB::O_F32: Basename = "unordered.f32"; break;
  case RTLIB::UO_F32: Basename = "unordered.f32"; break;
  case RTLIB::OLE_F32: Basename = "le.f32"; break;
  case RTLIB::OGE_F32: Basename = "ge.f32"; break;
  case RTLIB::OLT_F32: Basename = "lt.f32"; break;
  case RTLIB::OGT_F32: Basename = "gt.f32"; break;
  case RTLIB::OEQ_F32: Basename = "eq.f32"; break;
  case RTLIB::UNE_F32: Basename = "neq.f32"; break;
  }
  
  std::string prefix = PAN::getTagName(PAN::PREFIX_SYMBOL);
  std::string tagname = PAN::getTagName(PAN::LIBCALL);
  std::string Fullname = prefix + tagname + Basename; 

  // The name has to live through program life.
  return createESName(Fullname);
}

// getStdLibCallName - Get the name for the standard library function.
static const char *getStdLibCallName(unsigned opcode) {
  std::string BaseName;
  switch(opcode) {
    case RTLIB::COS_F32: BaseName = "cos";
      break;
    case RTLIB::SIN_F32: BaseName = "sin";
      break;
    case RTLIB::MEMCPY: BaseName = "memcpy";
      break;
    case RTLIB::MEMSET: BaseName = "memset";
      break;
    case RTLIB::MEMMOVE: BaseName = "memmove";
      break;
    default: llvm_unreachable("do not know std lib call name");
  }
  std::string prefix = PAN::getTagName(PAN::PREFIX_SYMBOL);
  std::string LibCallName = prefix + BaseName;

  // The name has to live through program life.
  return createESName(LibCallName);
}

// PIC16TargetLowering Constructor.
PIC16TargetLowering::PIC16TargetLowering(PIC16TargetMachine &TM)
  : TargetLowering(TM, new PIC16TargetObjectFile()), TmpSize(0) {
 
  Subtarget = &TM.getSubtarget<PIC16Subtarget>();

  addRegisterClass(MVT::i8, PIC16::GPRRegisterClass);

  setShiftAmountType(MVT::i8);
  
  // Std lib call names
  setLibcallName(RTLIB::COS_F32, getStdLibCallName(RTLIB::COS_F32));
  setLibcallName(RTLIB::SIN_F32, getStdLibCallName(RTLIB::SIN_F32));
  setLibcallName(RTLIB::MEMCPY, getStdLibCallName(RTLIB::MEMCPY));
  setLibcallName(RTLIB::MEMSET, getStdLibCallName(RTLIB::MEMSET));
  setLibcallName(RTLIB::MEMMOVE, getStdLibCallName(RTLIB::MEMMOVE));

  // SRA library call names
  setPIC16LibcallName(PIC16ISD::SRA_I8, getIntrinsicName(PIC16ISD::SRA_I8));
  setLibcallName(RTLIB::SRA_I16, getIntrinsicName(RTLIB::SRA_I16));
  setLibcallName(RTLIB::SRA_I32, getIntrinsicName(RTLIB::SRA_I32));

  // SHL library call names
  setPIC16LibcallName(PIC16ISD::SLL_I8, getIntrinsicName(PIC16ISD::SLL_I8));
  setLibcallName(RTLIB::SHL_I16, getIntrinsicName(RTLIB::SHL_I16));
  setLibcallName(RTLIB::SHL_I32, getIntrinsicName(RTLIB::SHL_I32));

  // SRL library call names
  setPIC16LibcallName(PIC16ISD::SRL_I8, getIntrinsicName(PIC16ISD::SRL_I8));
  setLibcallName(RTLIB::SRL_I16, getIntrinsicName(RTLIB::SRL_I16));
  setLibcallName(RTLIB::SRL_I32, getIntrinsicName(RTLIB::SRL_I32));

  // MUL Library call names
  setPIC16LibcallName(PIC16ISD::MUL_I8, getIntrinsicName(PIC16ISD::MUL_I8));
  setLibcallName(RTLIB::MUL_I16, getIntrinsicName(RTLIB::MUL_I16));
  setLibcallName(RTLIB::MUL_I32, getIntrinsicName(RTLIB::MUL_I32));

  // Signed division lib call names
  setLibcallName(RTLIB::SDIV_I16, getIntrinsicName(RTLIB::SDIV_I16));
  setLibcallName(RTLIB::SDIV_I32, getIntrinsicName(RTLIB::SDIV_I32));

  // Unsigned division lib call names
  setLibcallName(RTLIB::UDIV_I16, getIntrinsicName(RTLIB::UDIV_I16));
  setLibcallName(RTLIB::UDIV_I32, getIntrinsicName(RTLIB::UDIV_I32));

  // Signed remainder lib call names
  setLibcallName(RTLIB::SREM_I16, getIntrinsicName(RTLIB::SREM_I16));
  setLibcallName(RTLIB::SREM_I32, getIntrinsicName(RTLIB::SREM_I32));

  // Unsigned remainder lib call names
  setLibcallName(RTLIB::UREM_I16, getIntrinsicName(RTLIB::UREM_I16));
  setLibcallName(RTLIB::UREM_I32, getIntrinsicName(RTLIB::UREM_I32));
 
  // Floating point to signed int conversions.
  setLibcallName(RTLIB::FPTOSINT_F32_I8, 
                 getIntrinsicName(RTLIB::FPTOSINT_F32_I8));
  setLibcallName(RTLIB::FPTOSINT_F32_I16, 
                 getIntrinsicName(RTLIB::FPTOSINT_F32_I16));
  setLibcallName(RTLIB::FPTOSINT_F32_I32, 
                 getIntrinsicName(RTLIB::FPTOSINT_F32_I32));

  // Signed int to floats.
  setLibcallName(RTLIB::SINTTOFP_I32_F32, 
                 getIntrinsicName(RTLIB::SINTTOFP_I32_F32));

  // Floating points to unsigned ints.
  setLibcallName(RTLIB::FPTOUINT_F32_I8, 
                 getIntrinsicName(RTLIB::FPTOUINT_F32_I8));
  setLibcallName(RTLIB::FPTOUINT_F32_I16, 
                 getIntrinsicName(RTLIB::FPTOUINT_F32_I16));
  setLibcallName(RTLIB::FPTOUINT_F32_I32, 
                 getIntrinsicName(RTLIB::FPTOUINT_F32_I32));

  // Unsigned int to floats.
  setLibcallName(RTLIB::UINTTOFP_I32_F32, 
                 getIntrinsicName(RTLIB::UINTTOFP_I32_F32));

  // Floating point add, sub, mul ,div.
  setLibcallName(RTLIB::ADD_F32, getIntrinsicName(RTLIB::ADD_F32));
  setLibcallName(RTLIB::SUB_F32, getIntrinsicName(RTLIB::SUB_F32));
  setLibcallName(RTLIB::MUL_F32, getIntrinsicName(RTLIB::MUL_F32));
  setLibcallName(RTLIB::DIV_F32, getIntrinsicName(RTLIB::DIV_F32));

  // Floationg point comparison
  setLibcallName(RTLIB::UO_F32, getIntrinsicName(RTLIB::UO_F32));
  setLibcallName(RTLIB::OLE_F32, getIntrinsicName(RTLIB::OLE_F32));
  setLibcallName(RTLIB::OGE_F32, getIntrinsicName(RTLIB::OGE_F32));
  setLibcallName(RTLIB::OLT_F32, getIntrinsicName(RTLIB::OLT_F32));
  setLibcallName(RTLIB::OGT_F32, getIntrinsicName(RTLIB::OGT_F32));
  setLibcallName(RTLIB::OEQ_F32, getIntrinsicName(RTLIB::OEQ_F32));
  setLibcallName(RTLIB::UNE_F32, getIntrinsicName(RTLIB::UNE_F32));

  // Return value comparisons of floating point calls. 
  setCmpLibcallCC(RTLIB::OEQ_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::UNE_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::OLT_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::OLE_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::OGE_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::OGT_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::UO_F32, ISD::SETNE);
  setCmpLibcallCC(RTLIB::O_F32, ISD::SETEQ);

  setOperationAction(ISD::GlobalAddress, MVT::i16, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i16, Custom);

  setOperationAction(ISD::LOAD,   MVT::i8,  Legal);
  setOperationAction(ISD::LOAD,   MVT::i16, Custom);
  setOperationAction(ISD::LOAD,   MVT::i32, Custom);

  setOperationAction(ISD::STORE,  MVT::i8,  Legal);
  setOperationAction(ISD::STORE,  MVT::i16, Custom);
  setOperationAction(ISD::STORE,  MVT::i32, Custom);
  setOperationAction(ISD::STORE,  MVT::i64, Custom);

  setOperationAction(ISD::ADDE,    MVT::i8,  Custom);
  setOperationAction(ISD::ADDC,    MVT::i8,  Custom);
  setOperationAction(ISD::SUBE,    MVT::i8,  Custom);
  setOperationAction(ISD::SUBC,    MVT::i8,  Custom);
  setOperationAction(ISD::SUB,    MVT::i8,  Custom);
  setOperationAction(ISD::ADD,    MVT::i8,  Custom);
  setOperationAction(ISD::ADD,    MVT::i16, Custom);

  setOperationAction(ISD::OR,     MVT::i8,  Custom);
  setOperationAction(ISD::AND,    MVT::i8,  Custom);
  setOperationAction(ISD::XOR,    MVT::i8,  Custom);

  setOperationAction(ISD::FrameIndex, MVT::i16, Custom);

  setOperationAction(ISD::MUL,    MVT::i8,  Custom);

  setOperationAction(ISD::SMUL_LOHI,    MVT::i8,  Expand);
  setOperationAction(ISD::UMUL_LOHI,    MVT::i8,  Expand);
  setOperationAction(ISD::MULHU,        MVT::i8, Expand);
  setOperationAction(ISD::MULHS,        MVT::i8, Expand);

  setOperationAction(ISD::SRA,    MVT::i8,  Custom);
  setOperationAction(ISD::SHL,    MVT::i8,  Custom);
  setOperationAction(ISD::SRL,    MVT::i8,  Custom);

  setOperationAction(ISD::ROTL,    MVT::i8,  Expand);
  setOperationAction(ISD::ROTR,    MVT::i8,  Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // PIC16 does not support shift parts
  setOperationAction(ISD::SRA_PARTS,    MVT::i8, Expand);
  setOperationAction(ISD::SHL_PARTS,    MVT::i8, Expand);
  setOperationAction(ISD::SRL_PARTS,    MVT::i8, Expand);


  // PIC16 does not have a SETCC, expand it to SELECT_CC.
  setOperationAction(ISD::SETCC,  MVT::i8, Expand);
  setOperationAction(ISD::SELECT,  MVT::i8, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  setOperationAction(ISD::SELECT_CC,  MVT::i8, Custom);
  setOperationAction(ISD::BR_CC,  MVT::i8, Custom);

  //setOperationAction(ISD::TRUNCATE, MVT::i16, Custom);
  setTruncStoreAction(MVT::i16,   MVT::i8,  Custom);

  // Now deduce the information based on the above mentioned 
  // actions
  computeRegisterProperties();
}

// getOutFlag - Extract the flag result if the Op has it.
static SDValue getOutFlag(SDValue &Op) {
  // Flag is the last value of the node.
  SDValue Flag = Op.getValue(Op.getNode()->getNumValues() - 1);

  assert (Flag.getValueType() == MVT::Flag 
          && "Node does not have an out Flag");

  return Flag;
}
// Get the TmpOffset for FrameIndex
unsigned PIC16TargetLowering::GetTmpOffsetForFI(unsigned FI, unsigned size) {
  std::map<unsigned, unsigned>::iterator 
            MapIt = FiTmpOffsetMap.find(FI);
  if (MapIt != FiTmpOffsetMap.end())
      return MapIt->second;

  // This FI (FrameIndex) is not yet mapped, so map it
  FiTmpOffsetMap[FI] = TmpSize; 
  TmpSize += size;
  return FiTmpOffsetMap[FI];
}

// To extract chain value from the SDValue Nodes
// This function will help to maintain the chain extracting
// code at one place. In case of any change in future it will
// help maintain the code.
static SDValue getChain(SDValue &Op) { 
  SDValue Chain = Op.getValue(Op.getNode()->getNumValues() - 1);

  // If the last value returned in Flag then the chain is
  // second last value returned.
  if (Chain.getValueType() == MVT::Flag)
    Chain = Op.getValue(Op.getNode()->getNumValues() - 2);
  
  // All nodes may not produce a chain. Therefore following assert
  // verifies that the node is returning a chain only.
  assert (Chain.getValueType() == MVT::Other 
          && "Node does not have a chain");

  return Chain;
}

/// PopulateResults - Helper function to LowerOperation.
/// If a node wants to return multiple results after lowering,
/// it stuffs them into an array of SDValue called Results.

static void PopulateResults(SDValue N, SmallVectorImpl<SDValue>&Results) {
  if (N.getOpcode() == ISD::MERGE_VALUES) {
    int NumResults = N.getNumOperands();
    for( int i = 0; i < NumResults; i++)
      Results.push_back(N.getOperand(i));
  }
  else
    Results.push_back(N);
}

MVT::SimpleValueType
PIC16TargetLowering::getSetCCResultType(MVT ValType) const {
  return MVT::i8;
}

/// The type legalizer framework of generating legalizer can generate libcalls
/// only when the operand/result types are illegal.
/// PIC16 needs to generate libcalls even for the legal types (i8) for some ops.
/// For example an arithmetic right shift. These functions are used to lower
/// such operations that generate libcall for legal types.

void 
PIC16TargetLowering::setPIC16LibcallName(PIC16ISD::PIC16Libcall Call,
                                         const char *Name) {
  PIC16LibcallNames[Call] = Name; 
}

const char *
PIC16TargetLowering::getPIC16LibcallName(PIC16ISD::PIC16Libcall Call) {
  return PIC16LibcallNames[Call];
}

SDValue
PIC16TargetLowering::MakePIC16Libcall(PIC16ISD::PIC16Libcall Call,
                                      MVT RetVT, const SDValue *Ops,
                                      unsigned NumOps, bool isSigned,
                                      SelectionDAG &DAG, DebugLoc dl) {

  TargetLowering::ArgListTy Args;
  Args.reserve(NumOps);

  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 0; i != NumOps; ++i) {
    Entry.Node = Ops[i];
    Entry.Ty = Entry.Node.getValueType().getTypeForMVT();
    Entry.isSExt = isSigned;
    Entry.isZExt = !isSigned;
    Args.push_back(Entry);
  }
  SDValue Callee = DAG.getExternalSymbol(getPIC16LibcallName(Call), MVT::i16);

   const Type *RetTy = RetVT.getTypeForMVT();
   std::pair<SDValue,SDValue> CallInfo = 
     LowerCallTo(DAG.getEntryNode(), RetTy, isSigned, !isSigned, false,
                 false, 0, CallingConv::C, false,
                 /*isReturnValueUsed=*/true,
                 Callee, Args, DAG, dl);

  return CallInfo.first;
}

const char *PIC16TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default:                         return NULL;
  case PIC16ISD::Lo:               return "PIC16ISD::Lo";
  case PIC16ISD::Hi:               return "PIC16ISD::Hi";
  case PIC16ISD::MTLO:             return "PIC16ISD::MTLO";
  case PIC16ISD::MTHI:             return "PIC16ISD::MTHI";
  case PIC16ISD::MTPCLATH:         return "PIC16ISD::MTPCLATH";
  case PIC16ISD::PIC16Connect:     return "PIC16ISD::PIC16Connect";
  case PIC16ISD::Banksel:          return "PIC16ISD::Banksel";
  case PIC16ISD::PIC16Load:        return "PIC16ISD::PIC16Load";
  case PIC16ISD::PIC16LdArg:       return "PIC16ISD::PIC16LdArg";
  case PIC16ISD::PIC16LdWF:        return "PIC16ISD::PIC16LdWF";
  case PIC16ISD::PIC16Store:       return "PIC16ISD::PIC16Store";
  case PIC16ISD::PIC16StWF:        return "PIC16ISD::PIC16StWF";
  case PIC16ISD::BCF:              return "PIC16ISD::BCF";
  case PIC16ISD::LSLF:             return "PIC16ISD::LSLF";
  case PIC16ISD::LRLF:             return "PIC16ISD::LRLF";
  case PIC16ISD::RLF:              return "PIC16ISD::RLF";
  case PIC16ISD::RRF:              return "PIC16ISD::RRF";
  case PIC16ISD::CALL:             return "PIC16ISD::CALL";
  case PIC16ISD::CALLW:            return "PIC16ISD::CALLW";
  case PIC16ISD::SUBCC:            return "PIC16ISD::SUBCC";
  case PIC16ISD::SELECT_ICC:       return "PIC16ISD::SELECT_ICC";
  case PIC16ISD::BRCOND:           return "PIC16ISD::BRCOND";
  case PIC16ISD::RET:              return "PIC16ISD::RET";
  case PIC16ISD::Dummy:            return "PIC16ISD::Dummy";
  }
}

void PIC16TargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue>&Results,
                                             SelectionDAG &DAG) {

  switch (N->getOpcode()) {
    case ISD::GlobalAddress:
      Results.push_back(ExpandGlobalAddress(N, DAG));
      return;
    case ISD::ExternalSymbol:
      Results.push_back(ExpandExternalSymbol(N, DAG));
      return;
    case ISD::STORE:
      Results.push_back(ExpandStore(N, DAG));
      return;
    case ISD::LOAD:
      PopulateResults(ExpandLoad(N, DAG), Results);
      return;
    case ISD::ADD:
      // Results.push_back(ExpandAdd(N, DAG));
      return;
    case ISD::FrameIndex:
      Results.push_back(ExpandFrameIndex(N, DAG));
      return;
    default:
      assert (0 && "not implemented");
      return;
  }
}

SDValue PIC16TargetLowering::ExpandFrameIndex(SDNode *N, SelectionDAG &DAG) {

  // Currently handling FrameIndex of size MVT::i16 only
  // One example of this scenario is when return value is written on
  // FrameIndex#0

  if (N->getValueType(0) != MVT::i16)
    return SDValue();

  // Expand the FrameIndex into ExternalSymbol and a Constant node
  // The constant will represent the frame index number
  // Get the current function frame
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string Name = Func->getName();
  
  FrameIndexSDNode *FR = dyn_cast<FrameIndexSDNode>(SDValue(N,0));
  // FIXME there isn't really debug info here
  DebugLoc dl = FR->getDebugLoc();

  // Expand FrameIndex like GlobalAddress and ExternalSymbol
  // Also use Offset field for lo and hi parts. The default 
  // offset is zero.

  SDValue ES;
  int FrameOffset;
  SDValue FI = SDValue(N,0);
  LegalizeFrameIndex(FI, DAG, ES, FrameOffset);
  SDValue Offset = DAG.getConstant(FrameOffset, MVT::i8);
  SDValue Lo = DAG.getNode(PIC16ISD::Lo, dl, MVT::i8, ES, Offset);
  SDValue Hi = DAG.getNode(PIC16ISD::Hi, dl, MVT::i8, ES, Offset);
  return DAG.getNode(ISD::BUILD_PAIR, dl, N->getValueType(0), Lo, Hi);
}


SDValue PIC16TargetLowering::ExpandStore(SDNode *N, SelectionDAG &DAG) { 
  StoreSDNode *St = cast<StoreSDNode>(N);
  SDValue Chain = St->getChain();
  SDValue Src = St->getValue();
  SDValue Ptr = St->getBasePtr();
  MVT ValueType = Src.getValueType();
  unsigned StoreOffset = 0;
  DebugLoc dl = N->getDebugLoc();

  SDValue PtrLo, PtrHi;
  LegalizeAddress(Ptr, DAG, PtrLo, PtrHi, StoreOffset, dl);
 
  if (ValueType == MVT::i8) {
    return DAG.getNode (PIC16ISD::PIC16Store, dl, MVT::Other, Chain, Src,
                        PtrLo, PtrHi, 
                        DAG.getConstant (0 + StoreOffset, MVT::i8));
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
    SDValue Store1 = DAG.getNode(PIC16ISD::PIC16Store, dl, MVT::Other,
                                 ChainLo,
                                 SrcLo, PtrLo, PtrHi,
                                 DAG.getConstant (0 + StoreOffset, MVT::i8));

    SDValue Store2 = DAG.getNode(PIC16ISD::PIC16Store, dl, MVT::Other, ChainHi, 
                                 SrcHi, PtrLo, PtrHi,
                                 DAG.getConstant (1 + StoreOffset, MVT::i8));

    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, getChain(Store1),
                       getChain(Store2));
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
    SDValue Store1 = DAG.getNode(PIC16ISD::PIC16Store, dl, MVT::Other,
                                 ChainLo1,
                                 SrcLo1, PtrLo, PtrHi,
                                 DAG.getConstant (0 + StoreOffset, MVT::i8));

    SDValue Store2 = DAG.getNode(PIC16ISD::PIC16Store, dl, MVT::Other, ChainLo2,
                                 SrcLo2, PtrLo, PtrHi,
                                 DAG.getConstant (1 + StoreOffset, MVT::i8));

    SDValue Store3 = DAG.getNode(PIC16ISD::PIC16Store, dl, MVT::Other, ChainHi1,
                                 SrcHi1, PtrLo, PtrHi,
                                 DAG.getConstant (2 + StoreOffset, MVT::i8));

    SDValue Store4 = DAG.getNode(PIC16ISD::PIC16Store, dl, MVT::Other, ChainHi2,
                                 SrcHi2, PtrLo, PtrHi,
                                 DAG.getConstant (3 + StoreOffset, MVT::i8));

    SDValue RetLo =  DAG.getNode(ISD::TokenFactor, dl, MVT::Other, 
                                 getChain(Store1), getChain(Store2));
    SDValue RetHi =  DAG.getNode(ISD::TokenFactor, dl, MVT::Other, 
                                 getChain(Store3), getChain(Store4));
    return  DAG.getNode(ISD::TokenFactor, dl, MVT::Other, RetLo, RetHi);

  } else if (ValueType == MVT::i64) {
    SDValue SrcLo, SrcHi;
    GetExpandedParts(Src, DAG, SrcLo, SrcHi);
    SDValue ChainLo = Chain, ChainHi = Chain;
    if (Chain.getOpcode() == ISD::TokenFactor) {
      ChainLo = Chain.getOperand(0);
      ChainHi = Chain.getOperand(1);
    }
    SDValue Store1 = DAG.getStore(ChainLo, dl, SrcLo, Ptr, NULL,
                                  0 + StoreOffset);

    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getConstant(4, Ptr.getValueType()));
    SDValue Store2 = DAG.getStore(ChainHi, dl, SrcHi, Ptr, NULL,
                                  1 + StoreOffset);

    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1,
                       Store2);
  } else {
    assert (0 && "value type not supported");
    return SDValue();
  }
}

SDValue PIC16TargetLowering::ExpandExternalSymbol(SDNode *N, SelectionDAG &DAG)
{
  ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(SDValue(N, 0));
  // FIXME there isn't really debug info here
  DebugLoc dl = ES->getDebugLoc();

  SDValue TES = DAG.getTargetExternalSymbol(ES->getSymbol(), MVT::i8);
  SDValue Offset = DAG.getConstant(0, MVT::i8);
  SDValue Lo = DAG.getNode(PIC16ISD::Lo, dl, MVT::i8, TES, Offset);
  SDValue Hi = DAG.getNode(PIC16ISD::Hi, dl, MVT::i8, TES, Offset);

  return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i16, Lo, Hi);
}

// ExpandGlobalAddress - 
SDValue PIC16TargetLowering::ExpandGlobalAddress(SDNode *N, SelectionDAG &DAG) {
  GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(SDValue(N, 0));
  // FIXME there isn't really debug info here
  DebugLoc dl = G->getDebugLoc();
  
  SDValue TGA = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i8,
                                           G->getOffset());

  SDValue Offset = DAG.getConstant(0, MVT::i8);
  SDValue Lo = DAG.getNode(PIC16ISD::Lo, dl, MVT::i8, TGA, Offset);
  SDValue Hi = DAG.getNode(PIC16ISD::Hi, dl, MVT::i8, TGA, Offset);

  return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i16, Lo, Hi);
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

  if (GSDN->getAddressSpace() == PIC16ISD::ROM_SPACE)
    return true;

  // Any other address space return it false
  return false;
}


// GetExpandedParts - This function is on the similiar lines as
// the GetExpandedInteger in type legalizer is. This returns expanded
// parts of Op in Lo and Hi. 

void PIC16TargetLowering::GetExpandedParts(SDValue Op, SelectionDAG &DAG,
                                           SDValue &Lo, SDValue &Hi) {  
  SDNode *N = Op.getNode();
  DebugLoc dl = N->getDebugLoc();
  MVT NewVT = getTypeToTransformTo(N->getValueType(0));

  // Extract the lo component.
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, NewVT, Op,
                   DAG.getConstant(0, MVT::i8));

  // extract the hi component
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, NewVT, Op,
                   DAG.getConstant(1, MVT::i8));
}

// Legalize FrameIndex into ExternalSymbol and offset.
void 
PIC16TargetLowering::LegalizeFrameIndex(SDValue Op, SelectionDAG &DAG,
                                        SDValue &ES, int &Offset) {

  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const std::string Name = Func->getName();

  FrameIndexSDNode *FR = dyn_cast<FrameIndexSDNode>(Op);

  // FrameIndices are not stack offsets. But they represent the request
  // for space on stack. That space requested may be more than one byte. 
  // Therefore, to calculate the stack offset that a FrameIndex aligns
  // with, we need to traverse all the FrameIndices available earlier in 
  // the list and add their requested size.
  unsigned FIndex = FR->getIndex();
  const char *tmpName;
  if (FIndex < ReservedFrameCount) {
    tmpName = createESName(PAN::getFrameLabel(Name));
    ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);
    Offset = 0;
    for (unsigned i=0; i<FIndex ; ++i) {
      Offset += MFI->getObjectSize(i);
    }
  } else {
   // FrameIndex has been made for some temporary storage 
    tmpName = createESName(PAN::getTempdataLabel(Name));
    ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);
    Offset = GetTmpOffsetForFI(FIndex, MFI->getObjectSize(FIndex));
  }

  return;
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

void PIC16TargetLowering::LegalizeAddress(SDValue Ptr, SelectionDAG &DAG, 
                                          SDValue &Lo, SDValue &Hi,
                                          unsigned &Offset, DebugLoc dl) {

  // Offset, by default, should be 0
  Offset = 0;

  // If the pointer is ADD with constant,
  // return the constant value as the offset  
  if (Ptr.getOpcode() == ISD::ADD) {
    SDValue OperLeft = Ptr.getOperand(0);
    SDValue OperRight = Ptr.getOperand(1);
    if ((OperLeft.getOpcode() == ISD::Constant) &&
        (dyn_cast<ConstantSDNode>(OperLeft)->getZExtValue() < 32 )) {
      Offset = dyn_cast<ConstantSDNode>(OperLeft)->getZExtValue();
      Ptr = OperRight;
    } else if ((OperRight.getOpcode() == ISD::Constant)  &&
               (dyn_cast<ConstantSDNode>(OperRight)->getZExtValue() < 32 )){
      Offset = dyn_cast<ConstantSDNode>(OperRight)->getZExtValue();
      Ptr = OperLeft;
    }
  }

  // If the pointer is Type i8 and an external symbol
  // then treat it as direct address.
  // One example for such case is storing and loading
  // from function frame during a call
  if (Ptr.getValueType() == MVT::i8) {
    switch (Ptr.getOpcode()) {
    case ISD::TargetExternalSymbol:
      Lo = Ptr;
      Hi = DAG.getConstant(1, MVT::i8);
      return;
    }
  }

  // Expansion of FrameIndex has Lo/Hi parts
  if (isDirectAddress(Ptr)) { 
      SDValue TFI = Ptr.getOperand(0).getOperand(0); 
      int FrameOffset;
      if (TFI.getOpcode() == ISD::TargetFrameIndex) {
        LegalizeFrameIndex(TFI, DAG, Lo, FrameOffset);
        Hi = DAG.getConstant(1, MVT::i8);
        Offset += FrameOffset; 
        return;
      } else if (TFI.getOpcode() == ISD::TargetExternalSymbol) {
        // FrameIndex has already been expanded.
        // Now just make use of its expansion
        Lo = TFI;
        Hi = DAG.getConstant(1, MVT::i8);
        SDValue FOffset = Ptr.getOperand(0).getOperand(1);
        assert (FOffset.getOpcode() == ISD::Constant && 
                          "Invalid operand of PIC16ISD::Lo");
        Offset += dyn_cast<ConstantSDNode>(FOffset)->getZExtValue();
        return;
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
  Lo = DAG.getNode(PIC16ISD::MTLO, dl, MVT::i8, Lo);
  Hi = DAG.getNode(PIC16ISD::MTHI, dl, MVT::i8, Hi);

  return;
}

SDValue PIC16TargetLowering::ExpandLoad(SDNode *N, SelectionDAG &DAG) {
  LoadSDNode *LD = dyn_cast<LoadSDNode>(SDValue(N, 0));
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();
  DebugLoc dl = LD->getDebugLoc();

  SDValue Load, Offset;
  SDVTList Tys; 
  MVT VT, NewVT;
  SDValue PtrLo, PtrHi;
  unsigned LoadOffset;

  // Legalize direct/indirect addresses. This will give the lo and hi parts
  // of the address and the offset.
  LegalizeAddress(Ptr, DAG, PtrLo, PtrHi, LoadOffset, dl);

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
      Load = DAG.getNode(PIC16ISD::PIC16Load, dl, Tys, Chain, PtrLo, PtrHi,
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
    // if MVT::i1 is extended to MVT::i8 then MemBytes will be zero
    // So set it to one
    if (MemBytes == 0) MemBytes = 1;
    
    unsigned ExtdBytes = VT.getSizeInBits() / 8;
    Offset = DAG.getConstant(LoadOffset, MVT::i8);

    Tys = DAG.getVTList(MVT::i8, MVT::Other); 
    // For MemBytes generate PIC16Load with proper offset
    for (iter=0; iter < MemBytes; ++iter) {
      // Add the pointer offset if any
      Offset = DAG.getConstant(iter + LoadOffset, MVT::i8);
      Load = DAG.getNode(PIC16ISD::PIC16Load, dl, Tys, Chain, PtrLo, PtrHi,
                         Offset); 
      PICLoads.push_back(Load);
    }

    // For SignExtendedLoad
    if (ISD::isSEXTLoad(N)) {
      // For all ExtdBytes use the Right Shifted(Arithmetic) Value of the 
      // highest MemByte
      SDValue SRA = DAG.getNode(ISD::SRA, dl, MVT::i8, Load, 
                                DAG.getConstant(7, MVT::i8));
      for (iter=MemBytes; iter<ExtdBytes; ++iter) { 
        PICLoads.push_back(SRA);
      }
    } else if (ISD::isZEXTLoad(N) || ISD::isEXTLoad(N)) {
    //} else if (ISD::isZEXTLoad(N)) {
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
    return PICLoads[0];
  }
  else if (VT == MVT::i16) {
    BP = DAG.getNode(ISD::BUILD_PAIR, dl, VT, PICLoads[0], PICLoads[1]);
    if (MemVT == MVT::i8)
      Chain = getChain(PICLoads[0]);
    else
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, 
                          getChain(PICLoads[0]), getChain(PICLoads[1]));
  } else if (VT == MVT::i32) {
    SDValue BPs[2];
    BPs[0] = DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i16, 
                         PICLoads[0], PICLoads[1]);
    BPs[1] = DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i16,
                         PICLoads[2], PICLoads[3]);
    BP = DAG.getNode(ISD::BUILD_PAIR, dl, VT, BPs[0], BPs[1]);
    if (MemVT == MVT::i8)
      Chain = getChain(PICLoads[0]);
    else if (MemVT == MVT::i16)
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, 
                          getChain(PICLoads[0]), getChain(PICLoads[1]));
    else {
      SDValue Chains[2];
      Chains[0] = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                              getChain(PICLoads[0]), getChain(PICLoads[1]));
      Chains[1] = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                              getChain(PICLoads[2]), getChain(PICLoads[3]));
      Chain =  DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                           Chains[0], Chains[1]);
    }
  }
  Tys = DAG.getVTList(VT, MVT::Other); 
  return DAG.getNode(ISD::MERGE_VALUES, dl, Tys, BP, Chain);
}

SDValue PIC16TargetLowering::LowerShift(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal shift to lower");
 
  SDNode *N = Op.getNode();
  SDValue Value = N->getOperand(0);
  SDValue Amt = N->getOperand(1);
  PIC16ISD::PIC16Libcall CallCode;
  switch (N->getOpcode()) {
  case ISD::SRA:
    CallCode = PIC16ISD::SRA_I8;
    break;
  case ISD::SHL:
    CallCode = PIC16ISD::SLL_I8;
    break;
  case ISD::SRL:
    CallCode = PIC16ISD::SRL_I8;
    break;
  default:
    assert ( 0 && "This shift is not implemented yet.");
    return SDValue();
  }
  SmallVector<SDValue, 2> Ops(2);
  Ops[0] = Value;
  Ops[1] = Amt;
  SDValue Call = MakePIC16Libcall(CallCode, N->getValueType(0), &Ops[0], 2, 
                                  true, DAG, N->getDebugLoc());
  return Call;
}

SDValue PIC16TargetLowering::LowerMUL(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal multiply to lower");

  SDNode *N = Op.getNode();
  SmallVector<SDValue, 2> Ops(2);
  Ops[0] = N->getOperand(0);
  Ops[1] = N->getOperand(1);
  SDValue Call = MakePIC16Libcall(PIC16ISD::MUL_I8, N->getValueType(0), 
                                  &Ops[0], 2, true, DAG, N->getDebugLoc());
  return Call;
}

void
PIC16TargetLowering::LowerOperationWrapper(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) {
  SDValue Op = SDValue(N, 0);
  SDValue Res;
  unsigned i;
  switch (Op.getOpcode()) {
    case ISD::LOAD:
      Res = ExpandLoad(Op.getNode(), DAG); break;
    default: {
      // All other operations are handled in LowerOperation.
      Res = LowerOperation(Op, DAG);
      if (Res.getNode())
        Results.push_back(Res);
        
      return; 
    }
  }

  N = Res.getNode();
  unsigned NumValues = N->getNumValues(); 
  for (i = 0; i < NumValues ; i++) {
    Results.push_back(SDValue(N, i)); 
  }
}

SDValue PIC16TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
      return LowerADD(Op, DAG);
    case ISD::SUB:
    case ISD::SUBC:
    case ISD::SUBE:
      return LowerSUB(Op, DAG);
    case ISD::LOAD:
      return ExpandLoad(Op.getNode(), DAG);
    case ISD::STORE:
      return ExpandStore(Op.getNode(), DAG);
    case ISD::MUL:
      return LowerMUL(Op, DAG);
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
      return LowerShift(Op, DAG);
    case ISD::OR:
    case ISD::AND:
    case ISD::XOR:
      return LowerBinOp(Op, DAG);
    case ISD::BR_CC:
      return LowerBR_CC(Op, DAG);
    case ISD::SELECT_CC:
      return LowerSELECT_CC(Op, DAG);
  }
  return SDValue();
}

SDValue PIC16TargetLowering::ConvertToMemOperand(SDValue Op,
                                                 SelectionDAG &DAG,
                                                 DebugLoc dl) {
  assert (Op.getValueType() == MVT::i8 
          && "illegal value type to store on stack.");

  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string FuncName = Func->getName();


  // Put the value on stack.
  // Get a stack slot index and convert to es.
  int FI = MF.getFrameInfo()->CreateStackObject(1, 1);
  const char *tmpName = createESName(PAN::getTempdataLabel(FuncName));
  SDValue ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);

  // Store the value to ES.
  SDValue Store = DAG.getNode (PIC16ISD::PIC16Store, dl, MVT::Other,
                               DAG.getEntryNode(),
                               Op, ES, 
                               DAG.getConstant (1, MVT::i8), // Banksel.
                               DAG.getConstant (GetTmpOffsetForFI(FI, 1), 
                                                MVT::i8));

  // Load the value from ES.
  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other);
  SDValue Load = DAG.getNode(PIC16ISD::PIC16Load, dl, Tys, Store,
                             ES, DAG.getConstant (1, MVT::i8),
                             DAG.getConstant (GetTmpOffsetForFI(FI, 1), 
                             MVT::i8));
    
  return Load.getValue(0);
}

SDValue PIC16TargetLowering::
LowerIndirectCallArguments(SDValue Chain, SDValue InFlag,
                           SDValue DataAddr_Lo, SDValue DataAddr_Hi,
                           const SmallVectorImpl<ISD::OutputArg> &Outs,
                           DebugLoc dl, SelectionDAG &DAG) {
  unsigned NumOps = Outs.size();

  // If call has no arguments then do nothing and return.
  if (NumOps == 0)
    return Chain;

  std::vector<SDValue> Ops;
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDValue Arg, StoreRet;

  // For PIC16 ABI the arguments come after the return value. 
  unsigned RetVals = Outs.size();
  for (unsigned i = 0, ArgOffset = RetVals; i < NumOps; i++) {
    // Get the arguments
    Arg = Outs[i].Val;
    
    Ops.clear();
    Ops.push_back(Chain);
    Ops.push_back(Arg);
    Ops.push_back(DataAddr_Lo);
    Ops.push_back(DataAddr_Hi);
    Ops.push_back(DAG.getConstant(ArgOffset, MVT::i8));
    Ops.push_back(InFlag);

    StoreRet = DAG.getNode (PIC16ISD::PIC16StWF, dl, Tys, &Ops[0], Ops.size());

    Chain = getChain(StoreRet);
    InFlag = getOutFlag(StoreRet);
    ArgOffset++;
  }
  return Chain;
}

SDValue PIC16TargetLowering::
LowerDirectCallArguments(SDValue ArgLabel, SDValue Chain, SDValue InFlag,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         DebugLoc dl, SelectionDAG &DAG) {
  unsigned NumOps = Outs.size();
  std::string Name;
  SDValue Arg, StoreAt;
  MVT ArgVT;
  unsigned Size=0;

  // If call has no arguments then do nothing and return.
  if (NumOps == 0)
    return Chain; 

  // FIXME: This portion of code currently assumes only
  // primitive types being passed as arguments.

  // Legalize the address before use
  SDValue PtrLo, PtrHi;
  unsigned AddressOffset;
  int StoreOffset = 0;
  LegalizeAddress(ArgLabel, DAG, PtrLo, PtrHi, AddressOffset, dl);
  SDValue StoreRet;

  std::vector<SDValue> Ops;
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  for (unsigned i=0, Offset = 0; i<NumOps; i++) {
    // Get the argument
    Arg = Outs[i].Val;
    StoreOffset = (Offset + AddressOffset);
   
    // Store the argument on frame

    Ops.clear();
    Ops.push_back(Chain);
    Ops.push_back(Arg);
    Ops.push_back(PtrLo);
    Ops.push_back(PtrHi);
    Ops.push_back(DAG.getConstant(StoreOffset, MVT::i8));
    Ops.push_back(InFlag);

    StoreRet = DAG.getNode (PIC16ISD::PIC16StWF, dl, Tys, &Ops[0], Ops.size());

    Chain = getChain(StoreRet);
    InFlag = getOutFlag(StoreRet);

    // Update the frame offset to be used for next argument
    ArgVT = Arg.getValueType();
    Size = ArgVT.getSizeInBits();
    Size = Size/8;    // Calculate size in bytes
    Offset += Size;   // Increase the frame offset
  }
  return Chain;
}

SDValue PIC16TargetLowering::
LowerIndirectCallReturn(SDValue Chain, SDValue InFlag,
                        SDValue DataAddr_Lo, SDValue DataAddr_Hi,
                        const SmallVectorImpl<ISD::InputArg> &Ins,
                        DebugLoc dl, SelectionDAG &DAG,
                        SmallVectorImpl<SDValue> &InVals) {
  unsigned RetVals = Ins.size();

  // If call does not have anything to return
  // then do nothing and go back.
  if (RetVals == 0)
    return Chain;

  // Call has something to return
  SDValue LoadRet;

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other, MVT::Flag);
  for(unsigned i=0;i<RetVals;i++) {
    LoadRet = DAG.getNode(PIC16ISD::PIC16LdWF, dl, Tys, Chain, DataAddr_Lo,
                          DataAddr_Hi, DAG.getConstant(i, MVT::i8),
                          InFlag);
    InFlag = getOutFlag(LoadRet);
    Chain = getChain(LoadRet);
    InVals.push_back(LoadRet);
  }
  return Chain;
}

SDValue PIC16TargetLowering::
LowerDirectCallReturn(SDValue RetLabel, SDValue Chain, SDValue InFlag,
                      const SmallVectorImpl<ISD::InputArg> &Ins,
                      DebugLoc dl, SelectionDAG &DAG,
                      SmallVectorImpl<SDValue> &InVals) {

  // Currently handling primitive types only. They will come in
  // i8 parts
  unsigned RetVals = Ins.size();

  // Return immediately if the return type is void
  if (RetVals == 0)
    return Chain;

  // Call has something to return
  
  // Legalize the address before use
  SDValue LdLo, LdHi;
  unsigned LdOffset;
  LegalizeAddress(RetLabel, DAG, LdLo, LdHi, LdOffset, dl);

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other, MVT::Flag);
  SDValue LoadRet;
 
  for(unsigned i=0, Offset=0;i<RetVals;i++) {

    LoadRet = DAG.getNode(PIC16ISD::PIC16LdWF, dl, Tys, Chain, LdLo, LdHi,
                          DAG.getConstant(LdOffset + Offset, MVT::i8),
                          InFlag);

    InFlag = getOutFlag(LoadRet);

    Chain = getChain(LoadRet);
    Offset++;
    InVals.push_back(LoadRet);
  }

  return Chain;
}

SDValue
PIC16TargetLowering::LowerReturn(SDValue Chain,
                                 unsigned CallConv, bool isVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 DebugLoc dl, SelectionDAG &DAG) {

  // Number of values to return 
  unsigned NumRet = Outs.size();

  // Function returns value always on stack with the offset starting
  // from 0 
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *F = MF.getFunction();
  std::string FuncName = F->getName();

  const char *tmpName = createESName(PAN::getFrameLabel(FuncName));
  SDVTList VTs  = DAG.getVTList (MVT::i8, MVT::Other);
  SDValue ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);
  SDValue BS = DAG.getConstant(1, MVT::i8);
  SDValue RetVal;
  for(unsigned i=0;i<NumRet; ++i) {
    RetVal = Outs[i].Val;
    Chain =  DAG.getNode (PIC16ISD::PIC16Store, dl, MVT::Other, Chain, RetVal,
                        ES, BS,
                        DAG.getConstant (i, MVT::i8));
      
  }
  return DAG.getNode(PIC16ISD::RET, dl, MVT::Other, Chain);
}

void PIC16TargetLowering::
GetDataAddress(DebugLoc dl, SDValue Callee, SDValue &Chain, 
               SDValue &DataAddr_Lo, SDValue &DataAddr_Hi,
               SelectionDAG &DAG) {
   assert (Callee.getOpcode() == PIC16ISD::PIC16Connect
           && "Don't know what to do of such callee!!");
   SDValue ZeroOperand = DAG.getConstant(0, MVT::i8);
   SDValue SeqStart  = DAG.getCALLSEQ_START(Chain, ZeroOperand);
   Chain = getChain(SeqStart);
   SDValue OperFlag = getOutFlag(SeqStart); // To manage the data dependency

   // Get the Lo and Hi part of code address
   SDValue Lo = Callee.getOperand(0);
   SDValue Hi = Callee.getOperand(1);

   SDValue Data_Lo, Data_Hi;
   SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other, MVT::Flag);
   // Subtract 2 from Address to get the Lower part of DataAddress.
   SDVTList VTList = DAG.getVTList(MVT::i8, MVT::Flag);
   Data_Lo = DAG.getNode(ISD::SUBC, dl, VTList, Lo, 
                         DAG.getConstant(2, MVT::i8));
   SDValue Ops[3] = { Hi, DAG.getConstant(0, MVT::i8), Data_Lo.getValue(1)};
   Data_Hi = DAG.getNode(ISD::SUBE, dl, VTList, Ops, 3);
   SDValue PCLATH = DAG.getNode(PIC16ISD::MTPCLATH, dl, MVT::i8, Data_Hi);
   Callee = DAG.getNode(PIC16ISD::PIC16Connect, dl, MVT::i8, Data_Lo, PCLATH);
   SDValue Call = DAG.getNode(PIC16ISD::CALLW, dl, Tys, Chain, Callee,
                              OperFlag);
   Chain = getChain(Call);
   OperFlag = getOutFlag(Call);
   SDValue SeqEnd = DAG.getCALLSEQ_END(Chain, ZeroOperand, ZeroOperand,
                                       OperFlag);
   Chain = getChain(SeqEnd);
   OperFlag = getOutFlag(SeqEnd);

   // Low part of Data Address 
   DataAddr_Lo = DAG.getNode(PIC16ISD::MTLO, dl, MVT::i8, Call, OperFlag);

   // Make the second call.
   SeqStart  = DAG.getCALLSEQ_START(Chain, ZeroOperand);
   Chain = getChain(SeqStart);
   OperFlag = getOutFlag(SeqStart); // To manage the data dependency

   // Subtract 1 from Address to get high part of data address.
   Data_Lo = DAG.getNode(ISD::SUBC, dl, VTList, Lo, 
                         DAG.getConstant(1, MVT::i8));
   SDValue HiOps[3] = { Hi, DAG.getConstant(0, MVT::i8), Data_Lo.getValue(1)};
   Data_Hi = DAG.getNode(ISD::SUBE, dl, VTList, HiOps, 3);
   PCLATH = DAG.getNode(PIC16ISD::MTPCLATH, dl, MVT::i8, Data_Hi);

   // Use new Lo to make another CALLW
   Callee = DAG.getNode(PIC16ISD::PIC16Connect, dl, MVT::i8, Data_Lo, PCLATH);
   Call = DAG.getNode(PIC16ISD::CALLW, dl, Tys, Chain, Callee, OperFlag);
   Chain = getChain(Call);
   OperFlag = getOutFlag(Call);
   SeqEnd = DAG.getCALLSEQ_END(Chain, ZeroOperand, ZeroOperand,
                                        OperFlag);
   Chain = getChain(SeqEnd);
   OperFlag = getOutFlag(SeqEnd);
   // Hi part of Data Address
   DataAddr_Hi = DAG.getNode(PIC16ISD::MTHI, dl, MVT::i8, Call, OperFlag);
}

SDValue
PIC16TargetLowering::LowerCall(SDValue Chain, SDValue Callee,
                               unsigned CallConv, bool isVarArg,
                               bool isTailCall,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               DebugLoc dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) {

    assert(Callee.getValueType() == MVT::i16 &&
           "Don't know how to legalize this call node!!!");

    // The flag to track if this is a direct or indirect call.
    bool IsDirectCall = true;    
    unsigned RetVals = Ins.size();
    unsigned NumArgs = Outs.size();

    SDValue DataAddr_Lo, DataAddr_Hi; 
    if (!isa<GlobalAddressSDNode>(Callee) &&
        !isa<ExternalSymbolSDNode>(Callee)) {
       IsDirectCall = false;    // This is indirect call

       // If this is an indirect call then to pass the arguments
       // and read the return value back, we need the data address
       // of the function being called.
       // To get the data address two more calls need to be made.

       // Come here for indirect calls
       SDValue Lo, Hi;
       // Indirect addresses. Get the hi and lo parts of ptr.
       GetExpandedParts(Callee, DAG, Lo, Hi);
       // Connect Lo and Hi parts of the callee with the PIC16Connect
       Callee = DAG.getNode(PIC16ISD::PIC16Connect, dl, MVT::i8, Lo, Hi);

       // Read DataAddress only if we have to pass arguments or 
       // read return value. 
       if ((RetVals > 0) || (NumArgs > 0)) 
         GetDataAddress(dl, Callee, Chain, DataAddr_Lo, DataAddr_Hi, DAG);
    }

    SDValue ZeroOperand = DAG.getConstant(0, MVT::i8);

    // Start the call sequence.
    // Carring the Constant 0 along the CALLSEQSTART
    // because there is nothing else to carry.
    SDValue SeqStart  = DAG.getCALLSEQ_START(Chain, ZeroOperand);
    Chain = getChain(SeqStart);
    SDValue OperFlag = getOutFlag(SeqStart); // To manage the data dependency
    std::string Name;

    // For any direct call - callee will be GlobalAddressNode or
    // ExternalSymbol
    SDValue ArgLabel, RetLabel;
    if (IsDirectCall) { 
       // Considering the GlobalAddressNode case here.
       if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
          GlobalValue *GV = G->getGlobal();
          Callee = DAG.getTargetGlobalAddress(GV, MVT::i8);
          Name = G->getGlobal()->getName();
       } else {// Considering the ExternalSymbol case here
          ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Callee);
          Callee = DAG.getTargetExternalSymbol(ES->getSymbol(), MVT::i8); 
          Name = ES->getSymbol();
       }

       // Label for argument passing
       const char *argFrame = createESName(PAN::getArgsLabel(Name));
       ArgLabel = DAG.getTargetExternalSymbol(argFrame, MVT::i8);

       // Label for reading return value
       const char *retName = createESName(PAN::getRetvalLabel(Name));
       RetLabel = DAG.getTargetExternalSymbol(retName, MVT::i8);
    } else {
       // if indirect call
       SDValue CodeAddr_Lo = Callee.getOperand(0);
       SDValue CodeAddr_Hi = Callee.getOperand(1);

       /*CodeAddr_Lo = DAG.getNode(ISD::ADD, dl, MVT::i8, CodeAddr_Lo,
                                 DAG.getConstant(2, MVT::i8));*/

       // move Hi part in PCLATH
       CodeAddr_Hi = DAG.getNode(PIC16ISD::MTPCLATH, dl, MVT::i8, CodeAddr_Hi);
       Callee = DAG.getNode(PIC16ISD::PIC16Connect, dl, MVT::i8, CodeAddr_Lo,
                            CodeAddr_Hi);
    } 

    // Pass the argument to function before making the call.
    SDValue CallArgs;
    if (IsDirectCall) {
      CallArgs = LowerDirectCallArguments(ArgLabel, Chain, OperFlag,
                                          Outs, dl, DAG);
      Chain = getChain(CallArgs);
      OperFlag = getOutFlag(CallArgs);
    } else {
      CallArgs = LowerIndirectCallArguments(Chain, OperFlag, DataAddr_Lo,
                                            DataAddr_Hi, Outs, dl, DAG);
      Chain = getChain(CallArgs);
      OperFlag = getOutFlag(CallArgs);
    }

    SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
    SDValue PICCall = DAG.getNode(PIC16ISD::CALL, dl, Tys, Chain, Callee,
                                  OperFlag);
    Chain = getChain(PICCall);
    OperFlag = getOutFlag(PICCall);


    // Carrying the Constant 0 along the CALLSEQSTART
    // because there is nothing else to carry.
    SDValue SeqEnd = DAG.getCALLSEQ_END(Chain, ZeroOperand, ZeroOperand,
                                        OperFlag);
    Chain = getChain(SeqEnd);
    OperFlag = getOutFlag(SeqEnd);

    // Lower the return value reading after the call.
    if (IsDirectCall)
      return LowerDirectCallReturn(RetLabel, Chain, OperFlag,
                                   Ins, dl, DAG, InVals);
    else
      return LowerIndirectCallReturn(Chain, OperFlag, DataAddr_Lo,
                                     DataAddr_Hi, Ins, dl, DAG, InVals);
}

bool PIC16TargetLowering::isDirectLoad(const SDValue Op) {
  if (Op.getOpcode() == PIC16ISD::PIC16Load)
    if (Op.getOperand(1).getOpcode() == ISD::TargetGlobalAddress
     || Op.getOperand(1).getOpcode() == ISD::TargetExternalSymbol)
      return true;
  return false;
}

// NeedToConvertToMemOp - Returns true if one of the operands of the
// operation 'Op' needs to be put into memory. Also returns the
// operand no. of the operand to be converted in 'MemOp'. Remember, PIC16 has 
// no instruction that can operation on two registers. Most insns take
// one register and one memory operand (addwf) / Constant (addlw).
bool PIC16TargetLowering::NeedToConvertToMemOp(SDValue Op, unsigned &MemOp) {
  // If one of the operand is a constant, return false.
  if (Op.getOperand(0).getOpcode() == ISD::Constant ||
      Op.getOperand(1).getOpcode() == ISD::Constant)
    return false;    

  // Return false if one of the operands is already a direct
  // load and that operand has only one use.
  if (isDirectLoad(Op.getOperand(0))) {
    if (Op.getOperand(0).hasOneUse())
      return false;
    else 
      MemOp = 0;
  }
  if (isDirectLoad(Op.getOperand(1))) {
    if (Op.getOperand(1).hasOneUse())
      return false;
    else 
      MemOp = 1; 
  }
  return true;
}  

// LowerBinOp - Lower a commutative binary operation that does not
// affect status flag carry.
SDValue PIC16TargetLowering::LowerBinOp(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();

  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal Op to lower");

  unsigned MemOp = 1;
  if (NeedToConvertToMemOp(Op, MemOp)) {
    // Put one value on stack.
    SDValue NewVal = ConvertToMemOperand (Op.getOperand(MemOp), DAG, dl);

    return DAG.getNode(Op.getOpcode(), dl, MVT::i8, Op.getOperand(MemOp ^ 1),
    NewVal);
  }
  else {
    return Op;
  }
}

// LowerADD - Lower all types of ADD operations including the ones
// that affects carry.
SDValue PIC16TargetLowering::LowerADD(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal add to lower");
  DebugLoc dl = Op.getDebugLoc();
  unsigned MemOp = 1;
  if (NeedToConvertToMemOp(Op, MemOp)) {
    // Put one value on stack.
    SDValue NewVal = ConvertToMemOperand (Op.getOperand(MemOp), DAG, dl);
    
    // ADDC and ADDE produce two results.
    SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);

    // ADDE has three operands, the last one is the carry bit.
    if (Op.getOpcode() == ISD::ADDE)
      return DAG.getNode(Op.getOpcode(), dl, Tys, Op.getOperand(MemOp ^ 1),
                         NewVal, Op.getOperand(2));
    // ADDC has two operands.
    else if (Op.getOpcode() == ISD::ADDC)
      return DAG.getNode(Op.getOpcode(), dl, Tys, Op.getOperand(MemOp ^ 1),
                         NewVal);
    // ADD it is. It produces only one result.
    else
      return DAG.getNode(Op.getOpcode(), dl, MVT::i8, Op.getOperand(MemOp ^ 1),
                         NewVal);
  }
  else
    return Op;
}

SDValue PIC16TargetLowering::LowerSUB(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal sub to lower");

  // Nothing to do if the first operand is already a direct load and it has
  // only one use.
  if (isDirectLoad(Op.getOperand(0)) && Op.getOperand(0).hasOneUse())
    return Op;

  // Put first operand on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(0), DAG, dl);

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);
  switch (Op.getOpcode()) {
    default:
      assert (0 && "Opcode unknown."); 
    case ISD::SUBE:
      return DAG.getNode(Op.getOpcode(), dl, Tys, NewVal, Op.getOperand(1),
                         Op.getOperand(2));
      break;
    case ISD::SUBC:
      return DAG.getNode(Op.getOpcode(), dl, Tys, NewVal, Op.getOperand(1));
      break;
    case ISD::SUB:
      return DAG.getNode(Op.getOpcode(), dl, MVT::i8, NewVal, Op.getOperand(1));
      break;
  }
}

void PIC16TargetLowering::InitReservedFrameCount(const Function *F) {
  unsigned NumArgs = F->arg_size();

  bool isVoidFunc = (F->getReturnType()->getTypeID() == Type::VoidTyID);

  if (isVoidFunc)
    ReservedFrameCount = NumArgs;
  else
    ReservedFrameCount = NumArgs + 1;
}

// LowerFormalArguments - Argument values are loaded from the
// <fname>.args + offset. All arguments are already broken to leaglized
// types, so the offset just runs from 0 to NumArgVals - 1.

SDValue
PIC16TargetLowering::LowerFormalArguments(SDValue Chain,
                                          unsigned CallConv,
                                          bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                          DebugLoc dl,
                                          SelectionDAG &DAG,
                                          SmallVectorImpl<SDValue> &InVals) {
  unsigned NumArgVals = Ins.size();

  // Get the callee's name to create the <fname>.args label to pass args.
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *F = MF.getFunction();
  std::string FuncName = F->getName();

  // Reset the map of FI and TmpOffset 
  ResetTmpOffsetMap();
  // Initialize the ReserveFrameCount
  InitReservedFrameCount(F);

  // Create the <fname>.args external symbol.
  const char *tmpName = createESName(PAN::getArgsLabel(FuncName));
  SDValue ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);

  // Load arg values from the label + offset.
  SDVTList VTs  = DAG.getVTList (MVT::i8, MVT::Other);
  SDValue BS = DAG.getConstant(1, MVT::i8);
  for (unsigned i = 0; i < NumArgVals ; ++i) {
    SDValue Offset = DAG.getConstant(i, MVT::i8);
    SDValue PICLoad = DAG.getNode(PIC16ISD::PIC16LdArg, dl, VTs, Chain, ES, BS,
                                  Offset);
    Chain = getChain(PICLoad);
    InVals.push_back(PICLoad);
  }

  return Chain;
}

// Perform DAGCombine of PIC16Load.
// FIXME - Need a more elaborate comment here.
SDValue PIC16TargetLowering::
PerformPIC16LoadCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Chain = N->getOperand(0); 
  if (N->hasNUsesOfValue(0, 0)) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N,1), Chain);
  }
  return SDValue();
}

// For all the functions with arguments some STORE nodes are generated 
// that store the argument on the frameindex. However in PIC16 the arguments
// are passed on stack only. Therefore these STORE nodes are redundant. 
// To remove these STORE nodes will be removed in PerformStoreCombine 
//
// Currently this function is doint nothing and will be updated for removing
// unwanted store operations
SDValue PIC16TargetLowering::
PerformStoreCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  return SDValue(N, 0);
  /*
  // Storing an undef value is of no use, so remove it
  if (isStoringUndef(N, Chain, DAG)) {
    return Chain; // remove the store and return the chain
  }
  //else everything is ok.
  return SDValue(N, 0);
  */
}

SDValue PIC16TargetLowering::PerformDAGCombine(SDNode *N, 
                                               DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case ISD::STORE:   
   return PerformStoreCombine(N, DCI); 
  case PIC16ISD::PIC16Load:   
    return PerformPIC16LoadCombine(N, DCI);
  }
  return SDValue();
}

static PIC16CC::CondCodes IntCCToPIC16CC(ISD::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown condition code!");
  case ISD::SETNE:  return PIC16CC::NE;
  case ISD::SETEQ:  return PIC16CC::EQ;
  case ISD::SETGT:  return PIC16CC::GT;
  case ISD::SETGE:  return PIC16CC::GE;
  case ISD::SETLT:  return PIC16CC::LT;
  case ISD::SETLE:  return PIC16CC::LE;
  case ISD::SETULT: return PIC16CC::ULT;
  case ISD::SETULE: return PIC16CC::ULE;
  case ISD::SETUGE: return PIC16CC::UGE;
  case ISD::SETUGT: return PIC16CC::UGT;
  }
}

// Look at LHS/RHS/CC and see if they are a lowered setcc instruction.  If so
// set LHS/RHS and SPCC to the LHS/RHS of the setcc and SPCC to the condition.
static void LookThroughSetCC(SDValue &LHS, SDValue &RHS,
                             ISD::CondCode CC, unsigned &SPCC) {
  if (isa<ConstantSDNode>(RHS) &&
      cast<ConstantSDNode>(RHS)->getZExtValue() == 0 &&
      CC == ISD::SETNE &&
      (LHS.getOpcode() == PIC16ISD::SELECT_ICC &&
        LHS.getOperand(3).getOpcode() == PIC16ISD::SUBCC) &&
      isa<ConstantSDNode>(LHS.getOperand(0)) &&
      isa<ConstantSDNode>(LHS.getOperand(1)) &&
      cast<ConstantSDNode>(LHS.getOperand(0))->getZExtValue() == 1 &&
      cast<ConstantSDNode>(LHS.getOperand(1))->getZExtValue() == 0) {
    SDValue CMPCC = LHS.getOperand(3);
    SPCC = cast<ConstantSDNode>(LHS.getOperand(2))->getZExtValue();
    LHS = CMPCC.getOperand(0);
    RHS = CMPCC.getOperand(1);
  }
}

// Returns appropriate CMP insn and corresponding condition code in PIC16CC
SDValue PIC16TargetLowering::getPIC16Cmp(SDValue LHS, SDValue RHS, 
                                         unsigned CC, SDValue &PIC16CC, 
                                         SelectionDAG &DAG, DebugLoc dl) {
  PIC16CC::CondCodes CondCode = (PIC16CC::CondCodes) CC;

  // PIC16 sub is literal - W. So Swap the operands and condition if needed.
  // i.e. a < 12 can be rewritten as 12 > a.
  if (RHS.getOpcode() == ISD::Constant) {

    SDValue Tmp = LHS;
    LHS = RHS;
    RHS = Tmp;

    switch (CondCode) {
    default: break;
    case PIC16CC::LT:
      CondCode = PIC16CC::GT; 
      break;
    case PIC16CC::GT:
      CondCode = PIC16CC::LT; 
      break;
    case PIC16CC::ULT:
      CondCode = PIC16CC::UGT; 
      break;
    case PIC16CC::UGT:
      CondCode = PIC16CC::ULT; 
      break;
    case PIC16CC::GE:
      CondCode = PIC16CC::LE; 
      break;
    case PIC16CC::LE:
      CondCode = PIC16CC::GE;
      break;
    case PIC16CC::ULE:
      CondCode = PIC16CC::UGE;
      break;
    case PIC16CC::UGE:
      CondCode = PIC16CC::ULE;
      break;
    }
  }

  PIC16CC = DAG.getConstant(CondCode, MVT::i8);

  // These are signed comparisons. 
  SDValue Mask = DAG.getConstant(128, MVT::i8);
  if (isSignedComparison(CondCode)) {
    LHS = DAG.getNode (ISD::XOR, dl, MVT::i8, LHS, Mask);
    RHS = DAG.getNode (ISD::XOR, dl, MVT::i8, RHS, Mask); 
  }

  SDVTList VTs = DAG.getVTList (MVT::i8, MVT::Flag);
  // We can use a subtract operation to set the condition codes. But
  // we need to put one operand in memory if required.
  // Nothing to do if the first operand is already a valid type (direct load 
  // for subwf and literal for sublw) and it is used by this operation only. 
  if ((LHS.getOpcode() == ISD::Constant || isDirectLoad(LHS)) 
      && LHS.hasOneUse())
    return DAG.getNode(PIC16ISD::SUBCC, dl, VTs, LHS, RHS);

  // else convert the first operand to mem.
  LHS = ConvertToMemOperand (LHS, DAG, dl);
  return DAG.getNode(PIC16ISD::SUBCC, dl, VTs, LHS, RHS);
}


SDValue PIC16TargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  unsigned ORIGCC = ~0;
  DebugLoc dl = Op.getDebugLoc();

  // If this is a select_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  // i.e.
  // A setcc: lhs, rhs, cc is expanded by llvm to 
  // select_cc: result of setcc, 0, 1, 0, setne
  // We can think of it as:
  // select_cc: lhs, rhs, 1, 0, cc
  LookThroughSetCC(LHS, RHS, CC, ORIGCC);
  if (ORIGCC == ~0U) ORIGCC = IntCCToPIC16CC (CC);

  SDValue PIC16CC;
  SDValue Cmp = getPIC16Cmp(LHS, RHS, ORIGCC, PIC16CC, DAG, dl);

  return DAG.getNode (PIC16ISD::SELECT_ICC, dl, TrueVal.getValueType(), TrueVal,
                      FalseVal, PIC16CC, Cmp.getValue(1)); 
}

MachineBasicBlock *
PIC16TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *getTargetMachine().getInstrInfo();
  unsigned CC = (PIC16CC::CondCodes)MI->getOperand(3).getImm();
  DebugLoc dl = MI->getDebugLoc();

  // To "insert" a SELECT_CC instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It;

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   [f]bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);
  BuildMI(BB, dl, TII.get(PIC16::pic16brcond)).addMBB(sinkMBB).addImm(CC);
  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);

  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  sinkMBB->transferSuccessors(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, dl, TII.get(PIC16::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);

  F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
  return BB;
}


SDValue PIC16TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);   // LHS of the condition.
  SDValue RHS = Op.getOperand(3);   // RHS of the condition.
  SDValue Dest = Op.getOperand(4);  // BB to jump to
  unsigned ORIGCC = ~0;
  DebugLoc dl = Op.getDebugLoc();

  // If this is a br_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  LookThroughSetCC(LHS, RHS, CC, ORIGCC);
  if (ORIGCC == ~0U) ORIGCC = IntCCToPIC16CC (CC);

  // Get the Compare insn and condition code.
  SDValue PIC16CC;
  SDValue Cmp = getPIC16Cmp(LHS, RHS, ORIGCC, PIC16CC, DAG, dl);

  return DAG.getNode(PIC16ISD::BRCOND, dl, MVT::Other, Chain, Dest, PIC16CC, 
                     Cmp.getValue(1));
}

