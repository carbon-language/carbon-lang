//===- AlphaISelPattern.cpp - A pattern matching inst selector for Alpha --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for Alpha.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaRegisterInfo.h"
#include "llvm/Constants.h"                   // FIXME: REMOVE
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineConstantPool.h" // FIXME: REMOVE
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include <set>
#include <algorithm>
using namespace llvm;

namespace llvm {
  cl::opt<bool> EnableAlphaIDIV("enable-alpha-intfpdiv",
    cl::desc("Use the FP div instruction for integer div when possible"),
                             cl::Hidden);
  cl::opt<bool> EnableAlphaFTOI("enable-alpha-FTOI",
    cl::desc("Enable use of ftoi* and itof* instructions (ev6 and higher)"),
                             cl::Hidden);
  cl::opt<bool> EnableAlphaCT("enable-alpha-CT",
    cl::desc("Enable use of the ctpop, ctlz, and cttz instructions"),
                              cl::Hidden);
  cl::opt<bool> EnableAlphaCount("enable-alpha-count",
    cl::desc("Print estimates on live ins and outs"),
    cl::Hidden);
  cl::opt<bool> EnableAlphaLSMark("enable-alpha-lsmark",
    cl::desc("Emit symbols to correlate Mem ops to LLVM Values"),
    cl::Hidden);
}

namespace {
  // Alpha Specific DAG Nodes
  namespace AlphaISD {
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      //Convert an int bit pattern in an FP reg to a Double or Float
      //Has a dest type and a source
      CVTQ,
      //Move an Ireg to a FPreg
      ITOF,
      //Move a  FPreg to an Ireg
      FTOI, 
    };
  }
}

//===----------------------------------------------------------------------===//
//  AlphaTargetLowering - Alpha Implementation of the TargetLowering interface
namespace {
  class AlphaTargetLowering : public TargetLowering {
    int VarArgsOffset;  // What is the offset to the first vaarg
    int VarArgsBase;    // What is the base FrameIndex
    unsigned GP; //GOT vreg
    unsigned RA; //Return Address
  public:
    AlphaTargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Set up the TargetLowering object.
      //I am having problems with shr n ubyte 1
      setShiftAmountType(MVT::i64);
      setSetCCResultType(MVT::i64);
      setSetCCResultContents(ZeroOrOneSetCCResult);

      addRegisterClass(MVT::i64, Alpha::GPRCRegisterClass);
      addRegisterClass(MVT::f64, Alpha::FPRCRegisterClass);
      addRegisterClass(MVT::f32, Alpha::FPRCRegisterClass);

      setOperationAction(ISD::BRCONDTWOWAY, MVT::Other, Expand);

      setOperationAction(ISD::EXTLOAD, MVT::i1,  Promote);
      setOperationAction(ISD::EXTLOAD, MVT::f32, Promote);

      setOperationAction(ISD::ZEXTLOAD, MVT::i1   , Expand);
      setOperationAction(ISD::ZEXTLOAD, MVT::i32  , Expand);

      setOperationAction(ISD::SEXTLOAD, MVT::i1,  Expand);
      setOperationAction(ISD::SEXTLOAD, MVT::i8,  Expand);
      setOperationAction(ISD::SEXTLOAD, MVT::i16, Expand);

      setOperationAction(ISD::SREM, MVT::f32, Expand);
      setOperationAction(ISD::SREM, MVT::f64, Expand);

      setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);

      if (!EnableAlphaCT) {
        setOperationAction(ISD::CTPOP    , MVT::i64  , Expand);
        setOperationAction(ISD::CTTZ     , MVT::i64  , Expand);
        setOperationAction(ISD::CTLZ     , MVT::i64  , Expand);
      }

      //If this didn't legalize into a div....
      //      setOperationAction(ISD::SREM     , MVT::i64, Expand);
      //      setOperationAction(ISD::UREM     , MVT::i64, Expand);

      setOperationAction(ISD::MEMMOVE  , MVT::Other, Expand);
      setOperationAction(ISD::MEMSET   , MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY   , MVT::Other, Expand);

      // We don't support sin/cos/sqrt
      setOperationAction(ISD::FSIN , MVT::f64, Expand);
      setOperationAction(ISD::FCOS , MVT::f64, Expand);
      setOperationAction(ISD::FSQRT, MVT::f64, Expand);
      setOperationAction(ISD::FSIN , MVT::f32, Expand);
      setOperationAction(ISD::FCOS , MVT::f32, Expand);
      setOperationAction(ISD::FSQRT, MVT::f32, Expand);

      //Doesn't work yet
      setOperationAction(ISD::SETCC, MVT::f32,   Promote);

      //Try a couple things with a custom expander
      //setOperationAction(ISD::SINT_TO_FP       , MVT::i64  , Custom);

      computeRegisterProperties();

      addLegalFPImmediate(+0.0); //F31
      addLegalFPImmediate(-0.0); //-F31
    }

    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerVAStart(SDOperand Chain, SelectionDAG &DAG, SDOperand Dest);

    virtual std::pair<SDOperand,SDOperand>
    LowerVAArgNext(SDOperand Chain, SDOperand VAList,
                   const Type *ArgTy, SelectionDAG &DAG);

    std::pair<SDOperand,SDOperand>
    LowerVACopy(SDOperand Chain, SDOperand Src, SDOperand Dest, 
                                SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    void restoreGP(MachineBasicBlock* BB)
    {
      BuildMI(BB, Alpha::BIS, 2, Alpha::R29).addReg(GP).addReg(GP);
    }
    void restoreRA(MachineBasicBlock* BB)
    {
      BuildMI(BB, Alpha::BIS, 2, Alpha::R26).addReg(RA).addReg(RA);
    }
    unsigned getRA()
    {
      return RA;
    }

  };
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand AlphaTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
    MachineFunction &MF = DAG.getMachineFunction();
    switch (Op.getOpcode()) {
    default: assert(0 && "Should not custom lower this!");
#if 0
    case ISD::SINT_TO_FP:
      {
        assert (Op.getOperand(0).getValueType() == MVT::i64
                && "only quads can be loaded from");
        SDOperand SRC;
        if (EnableAlphaFTOI)
        {
          std::vector<MVT::ValueType> RTs;
          RTs.push_back(Op.getValueType());
          std::vector<SDOperand> Ops;
          Ops.push_back(Op.getOperand(0));
          SRC = DAG.getNode(AlphaISD::ITOF, RTs, Ops);
        } else {
          int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
          SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
          SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, 
                                        DAG.getEntryNode(), Op.getOperand(0), 
                                        StackSlot, DAG.getSrcValue(NULL));
          SRC = DAG.getLoad(Op.getValueType(), Store.getValue(0), StackSlot,
                            DAG.getSrcValue(NULL));
        }
        std::vector<MVT::ValueType> RTs;
        RTs.push_back(Op.getValueType());
        std::vector<SDOperand> Ops;
        Ops.push_back(SRC);
        return DAG.getNode(AlphaISD::CVTQ, RTs, Ops);
      }
#endif
    }
    return SDOperand();
}


/// AddLiveIn - This helper function adds the specified physical register to the
/// MachineFunction as a live in value.  It also creates a corresponding virtual
/// register for it.
static unsigned AddLiveIn(MachineFunction &MF, unsigned PReg,
                          TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getSSARegMap()->createVirtualRegister(RC);
  MF.addLiveIn(PReg, VReg);
  return VReg;
}

//http://www.cs.arizona.edu/computer.help/policy/DIGITAL_unix/AA-PY8AC-TET1_html/callCH3.html#BLOCK21

//For now, just use variable size stack frame format

//In a standard call, the first six items are passed in registers $16
//- $21 and/or registers $f16 - $f21. (See Section 4.1.2 for details
//of argument-to-register correspondence.) The remaining items are
//collected in a memory argument list that is a naturally aligned
//array of quadwords. In a standard call, this list, if present, must
//be passed at 0(SP).
//7 ... n         0(SP) ... (n-7)*8(SP)

// //#define FP    $15
// //#define RA    $26
// //#define PV    $27
// //#define GP    $29
// //#define SP    $30

std::vector<SDOperand>
AlphaTargetLowering::LowerArguments(Function &F, SelectionDAG &DAG)
{
  std::vector<SDOperand> ArgValues;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo*MFI = MF.getFrameInfo();

  MachineBasicBlock& BB = MF.front();

  unsigned args_int[] = {Alpha::R16, Alpha::R17, Alpha::R18,
                         Alpha::R19, Alpha::R20, Alpha::R21};
  unsigned args_float[] = {Alpha::F16, Alpha::F17, Alpha::F18,
                           Alpha::F19, Alpha::F20, Alpha::F21};
  int count = 0;

  GP = AddLiveIn(MF, Alpha::R29, getRegClassFor(MVT::i64));
  RA = AddLiveIn(MF, Alpha::R26, getRegClassFor(MVT::i64));

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
  {
    SDOperand argt;
    if (count  < 6) {
      unsigned Vreg;
      MVT::ValueType VT = getValueType(I->getType());
      switch (VT) {
      default:
        std::cerr << "Unknown Type " << VT << "\n";
        abort();
      case MVT::f64:
      case MVT::f32:
        args_float[count] = AddLiveIn(MF,args_float[count], getRegClassFor(VT));
        argt = DAG.getCopyFromReg(args_float[count], VT, DAG.getRoot());
        break;
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::i64:
        args_int[count] = AddLiveIn(MF, args_int[count], 
                                    getRegClassFor(MVT::i64));
        argt = DAG.getCopyFromReg(args_int[count], VT, DAG.getRoot());
        if (VT != MVT::i64)
          argt = DAG.getNode(ISD::TRUNCATE, VT, argt);
        break;
      }
      DAG.setRoot(argt.getValue(1));
    } else { //more args
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(8, 8 * (count - 6));

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
      argt = DAG.getLoad(getValueType(I->getType()),
                         DAG.getEntryNode(), FIN, DAG.getSrcValue(NULL));
    }
    ++count;
    ArgValues.push_back(argt);
  }

  // If the functions takes variable number of arguments, copy all regs to stack
  if (F.isVarArg()) {
    VarArgsOffset = count * 8;
    std::vector<SDOperand> LS;
    for (int i = 0; i < 6; ++i) {
      if (args_int[i] < 1024)
        args_int[i] = AddLiveIn(MF,args_int[i], getRegClassFor(MVT::i64));
      SDOperand argt = DAG.getCopyFromReg(args_int[i], MVT::i64, DAG.getRoot());
      int FI = MFI->CreateFixedObject(8, -8 * (6 - i));
      if (i == 0) VarArgsBase = FI;
      SDOperand SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getNode(ISD::STORE, MVT::Other, DAG.getRoot(), argt, 
                               SDFI, DAG.getSrcValue(NULL)));
      
      if (args_float[i] < 1024)
        args_float[i] = AddLiveIn(MF,args_float[i], getRegClassFor(MVT::f64));
      argt = DAG.getCopyFromReg(args_float[i], MVT::f64, DAG.getRoot());
      FI = MFI->CreateFixedObject(8, - 8 * (12 - i));
      SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getNode(ISD::STORE, MVT::Other, DAG.getRoot(), argt, 
                               SDFI, DAG.getSrcValue(NULL)));
    }

    //Set up a token factor with all the stack traffic
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, LS));
  }

  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
    MF.addLiveOut(Alpha::R0);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(Alpha::F0);
    break;
  }

  //return the arguments
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerCallTo(SDOperand Chain,
                                 const Type *RetTy, bool isVarArg,
                                 unsigned CallingConv, bool isTailCall,
                                 SDOperand Callee, ArgListTy &Args, 
                                 SelectionDAG &DAG) {
  int NumBytes = 0;
  if (Args.size() > 6)
    NumBytes = (Args.size() - 6) * 8;

  Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  std::vector<SDOperand> args_to_use;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
  {
    switch (getValueType(Args[i].second)) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      // Promote the integer to 64 bits.  If the input type is signed use a
      // sign extend, otherwise use a zero extend.
      if (Args[i].second->isSigned())
        Args[i].first = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64, Args[i].first);
      else
        Args[i].first = DAG.getNode(ISD::ZERO_EXTEND, MVT::i64, Args[i].first);
      break;
    case MVT::i64:
    case MVT::f64:
    case MVT::f32:
      break;
    }
    args_to_use.push_back(Args[i].first);
  }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(RetTyVT);
  RetVals.push_back(MVT::Other);

  SDOperand TheCall = SDOperand(DAG.getCall(RetVals,
                                            Chain, Callee, args_to_use), 0);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG, 
                                  SDOperand Dest) {
  // vastart just stores the address of the VarArgsBase and VarArgsOffset
  SDOperand FR  = DAG.getFrameIndex(VarArgsBase, MVT::i64);
  SDOperand S1  = DAG.getNode(ISD::STORE, MVT::Other, Chain, FR, Dest, 
                              DAG.getSrcValue(NULL));
  SDOperand SA2 = DAG.getNode(ISD::ADD, MVT::i64, Dest, 
                              DAG.getConstant(8, MVT::i64));
  SDOperand S2  = DAG.getNode(ISD::TRUNCSTORE, MVT::Other,   S1, 
                              DAG.getConstant(VarArgsOffset, MVT::i64), SA2, 
                              DAG.getSrcValue(NULL), MVT::i32);
  return std::make_pair(S2, S2);
}

std::pair<SDOperand,SDOperand> AlphaTargetLowering::
LowerVAArgNext(SDOperand Chain, SDOperand VAList,
               const Type *ArgTy, SelectionDAG &DAG) {
  SDOperand Base = DAG.getLoad(MVT::i64, Chain, VAList, DAG.getSrcValue(NULL));
  SDOperand Tmp = DAG.getNode(ISD::ADD, MVT::i64, VAList, 
                              DAG.getConstant(8, MVT::i64));
  SDOperand Offset = DAG.getNode(ISD::SEXTLOAD, MVT::i64, Base.getValue(1), 
                                 Tmp, DAG.getSrcValue(NULL), MVT::i32);
  SDOperand DataPtr = DAG.getNode(ISD::ADD, MVT::i64, Base, Offset);
  if (ArgTy->isFloatingPoint())
  {
    //if fp && Offset < 6*8, then subtract 6*8 from DataPtr
      SDOperand FPDataPtr = DAG.getNode(ISD::SUB, MVT::i64, DataPtr,
                                    DAG.getConstant(8*6, MVT::i64));
      SDOperand CC = DAG.getSetCC(ISD::SETLT, MVT::i64, 
                                  Offset, DAG.getConstant(8*6, MVT::i64));
      DataPtr = DAG.getNode(ISD::SELECT, MVT::i64, CC, FPDataPtr, DataPtr);
  }

  SDOperand Result;
  if (ArgTy == Type::IntTy)
    Result = DAG.getNode(ISD::SEXTLOAD, MVT::i64, Offset.getValue(1), DataPtr, 
                         DAG.getSrcValue(NULL), MVT::i32);
  else if (ArgTy == Type::UIntTy)
    Result = DAG.getNode(ISD::ZEXTLOAD, MVT::i64, Offset.getValue(1), DataPtr, 
                         DAG.getSrcValue(NULL), MVT::i32);
  else
    Result = DAG.getLoad(getValueType(ArgTy), Offset.getValue(1), DataPtr, 
                         DAG.getSrcValue(NULL));

  SDOperand NewOffset = DAG.getNode(ISD::ADD, MVT::i64, Offset, 
                                    DAG.getConstant(8, MVT::i64));
  SDOperand Update = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, 
                                 Result.getValue(1), NewOffset, 
                                 Tmp, DAG.getSrcValue(NULL), MVT::i32);
  Result = DAG.getNode(ISD::TRUNCATE, getValueType(ArgTy), Result);

  return std::make_pair(Result, Update);
}

std::pair<SDOperand,SDOperand> AlphaTargetLowering::
LowerVACopy(SDOperand Chain, SDOperand Src, SDOperand Dest, 
            SelectionDAG &DAG) {
  //Default to returning the input list
  SDOperand Val = DAG.getLoad(getPointerTy(), Chain, Src, 
                              DAG.getSrcValue(NULL));
  SDOperand Result = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                 Val, Dest, DAG.getSrcValue(NULL));
  SDOperand NP = DAG.getNode(ISD::ADD, MVT::i64, Src, 
                             DAG.getConstant(8, MVT::i64));
  Val = DAG.getNode(ISD::SEXTLOAD, MVT::i64, Result, NP, DAG.getSrcValue(NULL), 
                    MVT::i32);
  SDOperand NPD = DAG.getNode(ISD::ADD, MVT::i64, Dest, 
                             DAG.getConstant(8, MVT::i64));
  Result = DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Val.getValue(1),
                       Val, NPD, DAG.getSrcValue(NULL), MVT::i32);
  return std::make_pair(Result, Result);
}

std::pair<SDOperand, SDOperand> AlphaTargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  abort();
}





namespace {

//===--------------------------------------------------------------------===//
/// ISel - Alpha specific code to select Alpha machine instructions for
/// SelectionDAG operations.
//===--------------------------------------------------------------------===//
class AlphaISel : public SelectionDAGISel {

  /// AlphaLowering - This object fully describes how to lower LLVM code to an
  /// Alpha-specific SelectionDAG.
  AlphaTargetLowering AlphaLowering;

  SelectionDAG *ISelDAG;  // Hack to support us having a dag->dag transform
                          // for sdiv and udiv until it is put into the future
                          // dag combiner.

  /// ExprMap - As shared expressions are codegen'd, we keep track of which
  /// vreg the value is produced in, so we only emit one copy of each compiled
  /// tree.
  static const unsigned notIn = (unsigned)(-1);
  std::map<SDOperand, unsigned> ExprMap;

  //CCInvMap sometimes (SetNE) we have the inverse CC code for free
  std::map<SDOperand, unsigned> CCInvMap;

  int count_ins;
  int count_outs;
  bool has_sym;
  int max_depth;

public:
  AlphaISel(TargetMachine &TM) : SelectionDAGISel(AlphaLowering), 
    AlphaLowering(TM)
  {}

  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
    DEBUG(BB->dump());
    count_ins = 0;
    count_outs = 0;
    max_depth = 0;
    has_sym = false;

    // Codegen the basic block.
    ISelDAG = &DAG;
    max_depth = DAG.getRoot().getNodeDepth();
    Select(DAG.getRoot());

    if(has_sym)
      ++count_ins;
    if(EnableAlphaCount)
      std::cerr << "COUNT: " 
                << BB->getParent()->getFunction ()->getName() << " " 
                << BB->getNumber() << " " 
                << max_depth << " "
                << count_ins << " "
                << count_outs << "\n";

    // Clear state used for selection.
    ExprMap.clear();
    CCInvMap.clear();
  }
  
  virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF);

  unsigned SelectExpr(SDOperand N);
  unsigned SelectExprFP(SDOperand N, unsigned Result);
  void Select(SDOperand N);

  void SelectAddr(SDOperand N, unsigned& Reg, long& offset);
  void SelectBranchCC(SDOperand N);
  void MoveFP2Int(unsigned src, unsigned dst, bool isDouble);
  void MoveInt2FP(unsigned src, unsigned dst, bool isDouble);
  //returns whether the sense of the comparison was inverted
  bool SelectFPSetCC(SDOperand N, unsigned dst);

  // dag -> dag expanders for integer divide by constant
  SDOperand BuildSDIVSequence(SDOperand N);
  SDOperand BuildUDIVSequence(SDOperand N);

};
}

void AlphaISel::EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {
  // If this function has live-in values, emit the copies from pregs to vregs at
  // the top of the function, before anything else.
  MachineBasicBlock *BB = MF.begin();
  if (MF.livein_begin() != MF.livein_end()) {
    SSARegMap *RegMap = MF.getSSARegMap();
    for (MachineFunction::livein_iterator LI = MF.livein_begin(),
           E = MF.livein_end(); LI != E; ++LI) {
      const TargetRegisterClass *RC = RegMap->getRegClass(LI->second);
      if (RC == Alpha::GPRCRegisterClass) {
        BuildMI(BB, Alpha::BIS, 2, LI->second).addReg(LI->first)
          .addReg(LI->first);
      } else if (RC == Alpha::FPRCRegisterClass) {
        BuildMI(BB, Alpha::CPYS, 2, LI->second).addReg(LI->first)
          .addReg(LI->first);
      } else {
        assert(0 && "Unknown regclass!");
      }
    }
  }
}

//Find the offset of the arg in it's parent's function
static int getValueOffset(const Value* v)
{
  static int uniqneg = -1;
  if (v == NULL)
    return uniqneg--;

  const Instruction* itarget = dyn_cast<Instruction>(v);
  const BasicBlock* btarget = itarget->getParent();
  const Function* ftarget = btarget->getParent();

  //offset due to earlier BBs
  int i = 0;
  for(Function::const_iterator ii = ftarget->begin(); &*ii != btarget; ++ii)
    i += ii->size();

  for(BasicBlock::const_iterator ii = btarget->begin(); &*ii != itarget; ++ii)
    ++i;

  return i;
}
//Find the offset of the function in it's module
static int getFunctionOffset(const Function* fun)
{
  const Module* M = fun->getParent();

  //offset due to earlier BBs
  int i = 0;
  for(Module::const_iterator ii = M->begin(); &*ii != fun; ++ii)
    ++i;

  return i;
}

static int getUID()
{
  static int id = 0;
  return ++id;
}

//Factorize a number using the list of constants
static bool factorize(int v[], int res[], int size, uint64_t c)
{
  bool cont = true;
  while (c != 1 && cont)
  {
    cont = false;
    for(int i = 0; i < size; ++i)
    {
      if (c % v[i] == 0)
      {
        c /= v[i];
        ++res[i];
        cont=true;
      }
    }
  }
  return c == 1;
}


//Shamelessly adapted from PPC32
// Structure used to return the necessary information to codegen an SDIV as
// a multiply.
struct ms {
  int64_t m; // magic number
  int64_t s; // shift amount
};

struct mu {
  uint64_t m; // magic number
  int64_t a;          // add indicator
  int64_t s;          // shift amount
};

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static struct ms magic(int64_t d) {
  int64_t p;
  uint64_t ad, anc, delta, q1, r1, q2, r2, t;
  const uint64_t two63 = 9223372036854775808ULL; // 2^63
  struct ms mag;

  ad = abs(d);
  t = two63 + ((uint64_t)d >> 63);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 63;               // initialize p
  q1 = two63/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two63 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two63/ad;        // initialize q2 = 2p/abs(d)
  r2 = two63 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));

  mag.m = q2 + 1;
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 64;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static struct mu magicu(uint64_t d)
{
  int64_t p;
  uint64_t nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 63;                   // initialize p
  q1 = 0x8000000000000000ull/nc;       // initialize q1 = 2p/nc
  r1 = 0x8000000000000000ull - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFFFFFFFFFFull/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFFFFFFFFFFull - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFFFFFFFFFFull) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x8000000000000000ull) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 64;  // resulting shift
  return magu;
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand AlphaISel::BuildSDIVSequence(SDOperand N) {
  int64_t d = (int64_t)cast<ConstantSDNode>(N.getOperand(1))->getSignExtended();
  ms magics = magic(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = ISelDAG->getNode(ISD::MULHS, MVT::i64, N.getOperand(0),
                                 ISelDAG->getConstant(magics.m, MVT::i64));
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0)
    Q = ISelDAG->getNode(ISD::ADD, MVT::i64, Q, N.getOperand(0));
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0)
    Q = ISelDAG->getNode(ISD::SUB, MVT::i64, Q, N.getOperand(0));
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0)
    Q = ISelDAG->getNode(ISD::SRA, MVT::i64, Q,
                         ISelDAG->getConstant(magics.s, MVT::i64));
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    ISelDAG->getNode(ISD::SRL, MVT::i64, Q, ISelDAG->getConstant(63, MVT::i64));
  return ISelDAG->getNode(ISD::ADD, MVT::i64, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand AlphaISel::BuildUDIVSequence(SDOperand N) {
  unsigned d =
    (unsigned)cast<ConstantSDNode>(N.getOperand(1))->getSignExtended();
  mu magics = magicu(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = ISelDAG->getNode(ISD::MULHU, MVT::i64, N.getOperand(0),
                                 ISelDAG->getConstant(magics.m, MVT::i64));
  if (magics.a == 0) {
    Q = ISelDAG->getNode(ISD::SRL, MVT::i64, Q,
                         ISelDAG->getConstant(magics.s, MVT::i64));
  } else {
    SDOperand NPQ = ISelDAG->getNode(ISD::SUB, MVT::i64, N.getOperand(0), Q);
    NPQ = ISelDAG->getNode(ISD::SRL, MVT::i64, NPQ,
                           ISelDAG->getConstant(1, MVT::i64));
    NPQ = ISelDAG->getNode(ISD::ADD, MVT::i64, NPQ, Q);
    Q = ISelDAG->getNode(ISD::SRL, MVT::i64, NPQ,
                           ISelDAG->getConstant(magics.s-1, MVT::i64));
  }
  return Q;
}

//From PPC32
/// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
/// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(uint64_t Val) {
  if (Val == 0 || (Val & (Val-1))) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    Val >>= 1;
    ++Count;
  }
  return Count;
}


//These describe LDAx
static const int IMM_LOW  = -32768;
static const int IMM_HIGH = 32767;
static const int IMM_MULT = 65536;

static long getUpper16(long l)
{
  long y = l / IMM_MULT;
  if (l % IMM_MULT > IMM_HIGH)
    ++y;
  return y;
}

static long getLower16(long l)
{
  long h = getUpper16(l);
  return l - h * IMM_MULT;
}

static unsigned GetSymVersion(unsigned opcode)
{
  switch (opcode) {
  default: assert(0 && "unknown load or store"); return 0;
  case Alpha::LDQ: return Alpha::LDQ_SYM;
  case Alpha::LDS: return Alpha::LDS_SYM;
  case Alpha::LDT: return Alpha::LDT_SYM;
  case Alpha::LDL: return Alpha::LDL_SYM;
  case Alpha::LDBU: return Alpha::LDBU_SYM;
  case Alpha::LDWU: return Alpha::LDWU_SYM;
  case Alpha::LDW: return Alpha::LDW_SYM;
  case Alpha::LDB: return Alpha::LDB_SYM;
  case Alpha::STQ: return Alpha::STQ_SYM;
  case Alpha::STS: return Alpha::STS_SYM;
  case Alpha::STT: return Alpha::STT_SYM;
  case Alpha::STL: return Alpha::STL_SYM;
  case Alpha::STW: return Alpha::STW_SYM;
  case Alpha::STB: return Alpha::STB_SYM;
  }
}
static unsigned GetRelVersion(unsigned opcode)
{
  switch (opcode) {
  default: assert(0 && "unknown load or store"); return 0;
  case Alpha::LDQ: return Alpha::LDQr;
  case Alpha::LDS: return Alpha::LDSr;
  case Alpha::LDT: return Alpha::LDTr;
  case Alpha::LDL: return Alpha::LDLr;
  case Alpha::LDBU: return Alpha::LDBUr;
  case Alpha::LDWU: return Alpha::LDWUr;
  }
}

void AlphaISel::MoveFP2Int(unsigned src, unsigned dst, bool isDouble)
{
  unsigned Opc;
  if (EnableAlphaFTOI) {
    Opc = isDouble ? Alpha::FTOIT : Alpha::FTOIS;
    BuildMI(BB, Opc, 1, dst).addReg(src);
  } else {
    //The hard way:
    // Spill the integer to memory and reload it from there.
    unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
    MachineFunction *F = BB->getParent();
    int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);

    Opc = isDouble ? Alpha::STT : Alpha::STS;
    BuildMI(BB, Opc, 3).addReg(src).addFrameIndex(FrameIdx).addReg(Alpha::F31);
    Opc = isDouble ? Alpha::LDQ : Alpha::LDL;
    BuildMI(BB, Alpha::LDQ, 2, dst).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  }
}

void AlphaISel::MoveInt2FP(unsigned src, unsigned dst, bool isDouble)
{
  unsigned Opc;
  if (EnableAlphaFTOI) {
    Opc = isDouble?Alpha::ITOFT:Alpha::ITOFS;
    BuildMI(BB, Opc, 1, dst).addReg(src);
  } else {
    //The hard way:
    // Spill the integer to memory and reload it from there.
    unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
    MachineFunction *F = BB->getParent();
    int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);

    Opc = isDouble ? Alpha::STQ : Alpha::STL;
    BuildMI(BB, Opc, 3).addReg(src).addFrameIndex(FrameIdx).addReg(Alpha::F31);
    Opc = isDouble ? Alpha::LDT : Alpha::LDS;
    BuildMI(BB, Opc, 2, dst).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  }
}

bool AlphaISel::SelectFPSetCC(SDOperand N, unsigned dst)
{
  SDNode *Node = N.Val;
  unsigned Opc, Tmp1, Tmp2, Tmp3;
  SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node);

  bool rev = false;
  bool inv = false;

  switch (SetCC->getCondition()) {
  default: Node->dump(); assert(0 && "Unknown FP comparison!");
  case ISD::SETEQ: Opc = Alpha::CMPTEQ; break;
  case ISD::SETLT: Opc = Alpha::CMPTLT; break;
  case ISD::SETLE: Opc = Alpha::CMPTLE; break;
  case ISD::SETGT: Opc = Alpha::CMPTLT; rev = true; break;
  case ISD::SETGE: Opc = Alpha::CMPTLE; rev = true; break;
  case ISD::SETNE: Opc = Alpha::CMPTEQ; inv = true; break;
  }

  ConstantFPSDNode *CN;
  if ((CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(0)))
      && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
    Tmp1 = Alpha::F31;
  else
    Tmp1 = SelectExpr(N.getOperand(0));

  if ((CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(1)))
      && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
    Tmp2 = Alpha::F31;
  else
    Tmp2 = SelectExpr(N.getOperand(1));

  //Can only compare doubles, and dag won't promote for me
  if (SetCC->getOperand(0).getValueType() == MVT::f32)
    {
      //assert(0 && "Setcc On float?\n");
      std::cerr << "Setcc on float!\n";
      Tmp3 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::CVTST, 1, Tmp3).addReg(Tmp1);
      Tmp1 = Tmp3;
    }
  if (SetCC->getOperand(1).getValueType() == MVT::f32)
    {
      //assert (0 && "Setcc On float?\n");
      std::cerr << "Setcc on float!\n";
      Tmp3 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::CVTST, 1, Tmp3).addReg(Tmp2);
      Tmp2 = Tmp3;
    }

  if (rev) std::swap(Tmp1, Tmp2);
  //do the comparison
  BuildMI(BB, Opc, 2, dst).addReg(Tmp1).addReg(Tmp2);
  return inv;
}

//Check to see if the load is a constant offset from a base register
void AlphaISel::SelectAddr(SDOperand N, unsigned& Reg, long& offset)
{
  unsigned opcode = N.getOpcode();
  if (opcode == ISD::ADD && N.getOperand(1).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 32767)
  { //Normal imm add
    Reg = SelectExpr(N.getOperand(0));
    offset = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    return;
  }
  Reg = SelectExpr(N);
  offset = 0;
  return;
}

void AlphaISel::SelectBranchCC(SDOperand N)
{
  assert(N.getOpcode() == ISD::BRCOND && "Not a BranchCC???");
  MachineBasicBlock *Dest =
    cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();
  unsigned Opc = Alpha::WTF;

  Select(N.getOperand(0));  //chain
  SDOperand CC = N.getOperand(1);

  if (CC.getOpcode() == ISD::SETCC)
  {
    SetCCSDNode* SetCC = dyn_cast<SetCCSDNode>(CC.Val);
    if (MVT::isInteger(SetCC->getOperand(0).getValueType())) {
      //Dropping the CC is only useful if we are comparing to 0
      bool RightZero = SetCC->getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(SetCC->getOperand(1))->getValue() == 0;
      bool isNE = false;

      //Fix up CC
      ISD::CondCode cCode= SetCC->getCondition();

      if(cCode == ISD::SETNE)
        isNE = true;

      if (RightZero) {
        switch (cCode) {
        default: CC.Val->dump(); assert(0 && "Unknown integer comparison!");
        case ISD::SETEQ:  Opc = Alpha::BEQ; break;
        case ISD::SETLT:  Opc = Alpha::BLT; break;
        case ISD::SETLE:  Opc = Alpha::BLE; break;
        case ISD::SETGT:  Opc = Alpha::BGT; break;
        case ISD::SETGE:  Opc = Alpha::BGE; break;
        case ISD::SETULT: assert(0 && "x (unsigned) < 0 is never true"); break;
        case ISD::SETUGT: Opc = Alpha::BNE; break;
        //Technically you could have this CC
        case ISD::SETULE: Opc = Alpha::BEQ; break;
        case ISD::SETUGE: assert(0 && "x (unsgined >= 0 is always true"); break;
        case ISD::SETNE:  Opc = Alpha::BNE; break;
        }
        unsigned Tmp1 = SelectExpr(SetCC->getOperand(0)); //Cond
        BuildMI(BB, Opc, 2).addReg(Tmp1).addMBB(Dest);
        return;
      } else {
        unsigned Tmp1 = SelectExpr(CC);
        if (isNE)
          BuildMI(BB, Alpha::BEQ, 2).addReg(CCInvMap[CC]).addMBB(Dest);
        else
          BuildMI(BB, Alpha::BNE, 2).addReg(Tmp1).addMBB(Dest);
        return;
      }
    } else { //FP
      //Any comparison between 2 values should be codegened as an folded 
      //branch, as moving CC to the integer register is very expensive
      //for a cmp b: c = a - b;
      //a = b: c = 0
      //a < b: c < 0
      //a > b: c > 0

      bool invTest = false;
      unsigned Tmp3;

      ConstantFPSDNode *CN;
      if ((CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(1)))
          && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
        Tmp3 = SelectExpr(SetCC->getOperand(0));
      else if ((CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(0)))
          && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
      {
        Tmp3 = SelectExpr(SetCC->getOperand(1));
        invTest = true;
      }
      else
      {
        unsigned Tmp1 = SelectExpr(SetCC->getOperand(0));
        unsigned Tmp2 = SelectExpr(SetCC->getOperand(1));
        bool isD = SetCC->getOperand(0).getValueType() == MVT::f64;
        Tmp3 = MakeReg(isD ? MVT::f64 : MVT::f32);
        BuildMI(BB, isD ? Alpha::SUBT : Alpha::SUBS, 2, Tmp3)
          .addReg(Tmp1).addReg(Tmp2);
      }

      switch (SetCC->getCondition()) {
      default: CC.Val->dump(); assert(0 && "Unknown FP comparison!");
      case ISD::SETEQ: Opc = invTest ? Alpha::FBNE : Alpha::FBEQ; break;
      case ISD::SETLT: Opc = invTest ? Alpha::FBGT : Alpha::FBLT; break;
      case ISD::SETLE: Opc = invTest ? Alpha::FBGE : Alpha::FBLE; break;
      case ISD::SETGT: Opc = invTest ? Alpha::FBLT : Alpha::FBGT; break;
      case ISD::SETGE: Opc = invTest ? Alpha::FBLE : Alpha::FBGE; break;
      case ISD::SETNE: Opc = invTest ? Alpha::FBEQ : Alpha::FBNE; break;
      }
      BuildMI(BB, Opc, 2).addReg(Tmp3).addMBB(Dest);
      return;
    }
    abort(); //Should never be reached
  } else {
    //Giveup and do the stupid thing
    unsigned Tmp1 = SelectExpr(CC);
    BuildMI(BB, Alpha::BNE, 2).addReg(Tmp1).addMBB(Dest);
    return;
  }
  abort(); //Should never be reached
}

unsigned AlphaISel::SelectExprFP(SDOperand N, unsigned Result)
{
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();
  unsigned opcode = N.getOpcode();

  switch (opcode) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");

  case ISD::UNDEF: {
    BuildMI(BB, Alpha::IDEF, 0, Result);
    return Result;
  }

  case ISD::FNEG:
    if(ISD::FABS == N.getOperand(0).getOpcode())
      {
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        BuildMI(BB, Alpha::CPYSN, 2, Result).addReg(Alpha::F31).addReg(Tmp1);
      } else {
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Alpha::CPYSN, 2, Result).addReg(Tmp1).addReg(Tmp1);
      }
    return Result;

  case ISD::FABS:
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Alpha::CPYS, 2, Result).addReg(Alpha::F31).addReg(Tmp1);
    return Result;

  case ISD::SELECT:
    {
      //Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      unsigned TV = SelectExpr(N.getOperand(1)); //Use if TRUE
      unsigned FV = SelectExpr(N.getOperand(2)); //Use if FALSE

      SDOperand CC = N.getOperand(0);
      SetCCSDNode* SetCC = dyn_cast<SetCCSDNode>(CC.Val);

      if (CC.getOpcode() == ISD::SETCC &&
          !MVT::isInteger(SetCC->getOperand(0).getValueType()))
      { //FP Setcc -> Select yay!


        //for a cmp b: c = a - b;
        //a = b: c = 0
        //a < b: c < 0
        //a > b: c > 0

        bool invTest = false;
        unsigned Tmp3;

        ConstantFPSDNode *CN;
        if ((CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(1)))
            && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
          Tmp3 = SelectExpr(SetCC->getOperand(0));
        else if ((CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(0)))
                 && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
        {
          Tmp3 = SelectExpr(SetCC->getOperand(1));
          invTest = true;
        }
        else
        {
          unsigned Tmp1 = SelectExpr(SetCC->getOperand(0));
          unsigned Tmp2 = SelectExpr(SetCC->getOperand(1));
          bool isD = SetCC->getOperand(0).getValueType() == MVT::f64;
          Tmp3 = MakeReg(isD ? MVT::f64 : MVT::f32);
          BuildMI(BB, isD ? Alpha::SUBT : Alpha::SUBS, 2, Tmp3)
            .addReg(Tmp1).addReg(Tmp2);
        }

        switch (SetCC->getCondition()) {
        default: CC.Val->dump(); assert(0 && "Unknown FP comparison!");
        case ISD::SETEQ: Opc = invTest ? Alpha::FCMOVNE : Alpha::FCMOVEQ; break;
        case ISD::SETLT: Opc = invTest ? Alpha::FCMOVGT : Alpha::FCMOVLT; break;
        case ISD::SETLE: Opc = invTest ? Alpha::FCMOVGE : Alpha::FCMOVLE; break;
        case ISD::SETGT: Opc = invTest ? Alpha::FCMOVLT : Alpha::FCMOVGT; break;
        case ISD::SETGE: Opc = invTest ? Alpha::FCMOVLE : Alpha::FCMOVGE; break;
        case ISD::SETNE: Opc = invTest ? Alpha::FCMOVEQ : Alpha::FCMOVNE; break;
        }
        BuildMI(BB, Opc, 3, Result).addReg(FV).addReg(TV).addReg(Tmp3);
        return Result;
      }
      else
      {
        Tmp1 = SelectExpr(N.getOperand(0)); //Cond
        BuildMI(BB, Alpha::FCMOVEQ_INT, 3, Result).addReg(TV).addReg(FV)
          .addReg(Tmp1);
//         // Spill the cond to memory and reload it from there.
//         unsigned Tmp4 = MakeReg(MVT::f64);
//         MoveIntFP(Tmp1, Tmp4, true);
//         //now ideally, we don't have to do anything to the flag...
//         // Get the condition into the zero flag.
//         BuildMI(BB, Alpha::FCMOVEQ, 3, Result).addReg(TV).addReg(FV).addReg(Tmp4);
        return Result;
      }
    }

  case ISD::FP_ROUND:
    assert (DestType == MVT::f32 &&
            N.getOperand(0).getValueType() == MVT::f64 &&
            "only f64 to f32 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Alpha::CVTTS, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::FP_EXTEND:
    assert (DestType == MVT::f64 &&
            N.getOperand(0).getValueType() == MVT::f32 &&
            "only f32 to f64 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Alpha::CVTST, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::CopyFromReg:
    {
      // Make sure we generate both values.
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      SDOperand Chain   = N.getOperand(0);

      Select(Chain);
      unsigned r = dyn_cast<RegSDNode>(Node)->getReg();
      //std::cerr << "CopyFromReg " << Result << " = " << r << "\n";
      BuildMI(BB, Alpha::CPYS, 2, Result).addReg(r).addReg(r);
      return Result;
    }

  case ISD::LOAD:
    {
      // Make sure we generate both values.
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      DestType = N.getValue(0).getValueType();

      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);
      Select(Chain);
      Opc = DestType == MVT::f64 ? Alpha::LDT : Alpha::LDS;

      if (EnableAlphaLSMark)
      {
        int i = getValueOffset(dyn_cast<SrcValueSDNode>(N.getOperand(2))
                                  ->getValue());
        int j = getFunctionOffset(BB->getParent()->getFunction());
        BuildMI(BB, Alpha::MEMLABEL, 3).addImm(j).addImm(i).addImm(getUID());
      }
      
      if (Address.getOpcode() == ISD::GlobalAddress) {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        has_sym = true;
        BuildMI(BB, Opc, 1, Result)
          .addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
        AlphaLowering.restoreGP(BB);
        Opc = GetRelVersion(Opc);
        has_sym = true;
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp1).addConstantPoolIndex(CP->getIndex())
          .addReg(Alpha::R29);
        BuildMI(BB, Opc, 2, Result).addConstantPoolIndex(CP->getIndex())
          .addReg(Tmp1);
      }
      else if(Address.getOpcode() == ISD::FrameIndex) {
        BuildMI(BB, Opc, 2, Result)
          .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex())
          .addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp1, offset);
        BuildMI(BB, Opc, 2, Result).addImm(offset).addReg(Tmp1);
      }
      return Result;
    }
  case ISD::ConstantFP:
    if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N)) {
      if (CN->isExactlyValue(+0.0)) {
        BuildMI(BB, Alpha::CPYS, 2, Result).addReg(Alpha::F31)
          .addReg(Alpha::F31);
      } else if ( CN->isExactlyValue(-0.0)) {
        BuildMI(BB, Alpha::CPYSN, 2, Result).addReg(Alpha::F31)
          .addReg(Alpha::F31);
      } else {
        abort();
      }
    }
    return Result;

  case ISD::SDIV:
  case ISD::MUL:
  case ISD::ADD:
  case ISD::SUB:
  switch( opcode ) {
      case ISD::MUL: Opc = DestType == MVT::f64 ? Alpha::MULT : Alpha::MULS; 
      break;
      case ISD::ADD: Opc = DestType == MVT::f64 ? Alpha::ADDT : Alpha::ADDS; 
      break;
      case ISD::SUB: Opc = DestType == MVT::f64 ? Alpha::SUBT : Alpha::SUBS; 
      break;
      case ISD::SDIV: Opc = DestType == MVT::f64 ? Alpha::DIVT : Alpha::DIVS;
      break;
    };

    ConstantFPSDNode *CN;
    if (opcode == ISD::SUB
        && (CN = dyn_cast<ConstantFPSDNode>(N.getOperand(0)))
        && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
    {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, Alpha::CPYSN, 2, Result).addReg(Tmp2).addReg(Tmp2);
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::EXTLOAD:
    {
      //include a conversion sequence for float loads to double
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      Tmp1 = MakeReg(MVT::f32);

      assert(cast<MVTSDNode>(Node)->getExtraValueType() == MVT::f32 &&
             "EXTLOAD not from f32");
      assert(Node->getValueType(0) == MVT::f64 && "EXTLOAD not to f64");

      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);
      Select(Chain);

      if (EnableAlphaLSMark)
      {
        int i = getValueOffset(dyn_cast<SrcValueSDNode>(N.getOperand(2))
                                  ->getValue());
        int j = getFunctionOffset(BB->getParent()->getFunction());
        BuildMI(BB, Alpha::MEMLABEL, 3).addImm(j).addImm(i).addImm(getUID());
      }

      if (Address.getOpcode() == ISD::GlobalAddress) {
        AlphaLowering.restoreGP(BB);
        has_sym = true;
        BuildMI(BB, Alpha::LDS_SYM, 1, Tmp1)
          .addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if (ConstantPoolSDNode *CP =
               dyn_cast<ConstantPoolSDNode>(N.getOperand(1)))
      {
        AlphaLowering.restoreGP(BB);
        has_sym = true;
        Tmp2 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp2).addConstantPoolIndex(CP->getIndex())
          .addReg(Alpha::R29);
        BuildMI(BB, Alpha::LDSr, 2, Tmp1).addConstantPoolIndex(CP->getIndex())
          .addReg(Tmp2);
      }
      else if(Address.getOpcode() == ISD::FrameIndex) {
        Tmp2 = cast<FrameIndexSDNode>(Address)->getIndex();
        BuildMI(BB, Alpha::LDS, 2, Tmp1)
          .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex())
          .addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp2, offset);
        BuildMI(BB, Alpha::LDS, 1, Tmp1).addImm(offset).addReg(Tmp2);
      }
      BuildMI(BB, Alpha::CVTST, 1, Result).addReg(Tmp1);
      return Result;
    }

  case ISD::SINT_TO_FP:
    {
      assert (N.getOperand(0).getValueType() == MVT::i64
              && "only quads can be loaded from");
      Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
      Tmp2 = MakeReg(MVT::f64);
      MoveInt2FP(Tmp1, Tmp2, true);
      Opc = DestType == MVT::f64 ? Alpha::CVTQT : Alpha::CVTQS;
      BuildMI(BB, Opc, 1, Result).addReg(Tmp2);
      return Result;
    }
  }
  assert(0 && "should not get here");
  return 0;
}

unsigned AlphaISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2 = 0, Tmp3;
  unsigned Opc = 0;
  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  if (N.getOpcode() != ISD::CALL && N.getOpcode() != ISD::TAILCALL)
    Reg = Result = (N.getValueType() != MVT::Other) ?
      MakeReg(N.getValueType()) : notIn;
  else {
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    if (Node->getNumValues() == 1)
      Reg = Result = notIn;  // Void call, just a chain.
    else {
      Result = MakeReg(Node->getValueType(0));
      ExprMap[N.getValue(0)] = Result;
      for (unsigned i = 1, e = N.Val->getNumValues()-1; i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
      ExprMap[SDOperand(Node, Node->getNumValues()-1)] = notIn;
    }
  }

  if ((DestType == MVT::f64 || DestType == MVT::f32 ||
       (
        (opcode == ISD::LOAD || opcode == ISD::CopyFromReg ||
         opcode == ISD::EXTLOAD) &&
        (N.getValue(0).getValueType() == MVT::f32 ||
         N.getValue(0).getValueType() == MVT::f64)
       ))
      && opcode != ISD::CALL && opcode != ISD::TAILCALL
      )
    return SelectExprFP(N, Result);

  switch (opcode) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");

  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ:
    Opc = opcode == ISD::CTPOP ? Alpha::CTPOP :
    (opcode == ISD::CTTZ ? Alpha::CTTZ : Alpha::CTLZ);
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::MULHU:
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Alpha::UMULH, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  case ISD::MULHS:
    {
      //MULHU - Ra<63>*Rb - Rb<63>*Ra
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      Tmp3 = MakeReg(MVT::i64);
      BuildMI(BB, Alpha::UMULH, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
      unsigned V1 = MakeReg(MVT::i64);
      unsigned V2 = MakeReg(MVT::i64);
      BuildMI(BB, Alpha::CMOVGE, 3, V1).addReg(Tmp2).addReg(Alpha::R31)
        .addReg(Tmp1);
      BuildMI(BB, Alpha::CMOVGE, 3, V2).addReg(Tmp1).addReg(Alpha::R31)
        .addReg(Tmp2);
      unsigned IRes = MakeReg(MVT::i64);
      BuildMI(BB, Alpha::SUBQ, 2, IRes).addReg(Tmp3).addReg(V1);
      BuildMI(BB, Alpha::SUBQ, 2, Result).addReg(IRes).addReg(V2);
      return Result;
    }
  case ISD::UNDEF: {
    BuildMI(BB, Alpha::IDEF, 0, Result);
    return Result;
  }

  case ISD::DYNAMIC_STACKALLOC:
    // Generate both result values.
    if (Result != notIn)
      ExprMap[N.getValue(1)] = notIn;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    // FIXME: We are currently ignoring the requested alignment for handling
    // greater than the stack alignment.  This will need to be revisited at some
    // point.  Align = N.getOperand(2);

    if (!isa<ConstantSDNode>(N.getOperand(2)) ||
        cast<ConstantSDNode>(N.getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }

    Select(N.getOperand(0));
    if (ConstantSDNode* CN = dyn_cast<ConstantSDNode>(N.getOperand(1)))
    {
      if (CN->getValue() < 32000)
      {
        BuildMI(BB, Alpha::LDA, 2, Alpha::R30)
          .addImm(-CN->getValue()).addReg(Alpha::R30);
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        // Subtract size from stack pointer, thereby allocating some space.
        BuildMI(BB, Alpha::SUBQ, 2, Alpha::R30).addReg(Alpha::R30).addReg(Tmp1);
      }
    } else {
      Tmp1 = SelectExpr(N.getOperand(1));
      // Subtract size from stack pointer, thereby allocating some space.
      BuildMI(BB, Alpha::SUBQ, 2, Alpha::R30).addReg(Alpha::R30).addReg(Tmp1);
    }

    // Put a pointer to the space into the result register, by copying the stack
    // pointer.
    BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R30).addReg(Alpha::R30);
    return Result;

  case ISD::ConstantPool:
    Tmp1 = cast<ConstantPoolSDNode>(N)->getIndex();
    AlphaLowering.restoreGP(BB);
    Tmp2 = MakeReg(MVT::i64);
    BuildMI(BB, Alpha::LDAHr, 2, Tmp2).addConstantPoolIndex(Tmp1)
      .addReg(Alpha::R29);
    BuildMI(BB, Alpha::LDAr, 2, Result).addConstantPoolIndex(Tmp1)
      .addReg(Tmp2);
    return Result;

  case ISD::FrameIndex:
    BuildMI(BB, Alpha::LDA, 2, Result)
      .addFrameIndex(cast<FrameIndexSDNode>(N)->getIndex())
      .addReg(Alpha::F31);
    return Result;

  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::LOAD:
    {
      // Make sure we generate both values.
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);
      Select(Chain);

      assert(Node->getValueType(0) == MVT::i64 &&
             "Unknown type to sign extend to.");
      if (opcode == ISD::LOAD)
        Opc = Alpha::LDQ;
      else
        switch (cast<MVTSDNode>(Node)->getExtraValueType()) {
        default: Node->dump(); assert(0 && "Bad sign extend!");
        case MVT::i32: Opc = Alpha::LDL;
          assert(opcode != ISD::ZEXTLOAD && "Not sext"); break;
        case MVT::i16: Opc = Alpha::LDWU;
          assert(opcode != ISD::SEXTLOAD && "Not zext"); break;
        case MVT::i1: //FIXME: Treat i1 as i8 since there are problems otherwise
        case MVT::i8: Opc = Alpha::LDBU;
          assert(opcode != ISD::SEXTLOAD && "Not zext"); break;
        }

      if (EnableAlphaLSMark)
      {
        int i = getValueOffset(dyn_cast<SrcValueSDNode>(N.getOperand(2))
                                 ->getValue());
        int j = getFunctionOffset(BB->getParent()->getFunction());
        BuildMI(BB, Alpha::MEMLABEL, 3).addImm(j).addImm(i).addImm(getUID());
      }

      if (Address.getOpcode() == ISD::GlobalAddress) {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        has_sym = true;
        BuildMI(BB, Opc, 1, Result)
          .addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
        AlphaLowering.restoreGP(BB);
        Opc = GetRelVersion(Opc);
        has_sym = true;
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp1).addConstantPoolIndex(CP->getIndex())
          .addReg(Alpha::R29);
        BuildMI(BB, Opc, 2, Result).addConstantPoolIndex(CP->getIndex())
          .addReg(Tmp1);
      }
      else if(Address.getOpcode() == ISD::FrameIndex) {
        BuildMI(BB, Opc, 2, Result)
          .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex())
          .addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp1, offset);
        BuildMI(BB, Opc, 2, Result).addImm(offset).addReg(Tmp1);
      }
      return Result;
    }

  case ISD::GlobalAddress:
    AlphaLowering.restoreGP(BB);
    has_sym = true;
    BuildMI(BB, Alpha::LOAD_ADDR, 1, Result)
      .addGlobalAddress(cast<GlobalAddressSDNode>(N)->getGlobal());
    return Result;

  case ISD::TAILCALL:
  case ISD::CALL:
    {
      Select(N.getOperand(0));

      // The chain for this call is now lowered.
      ExprMap.insert(std::make_pair(N.getValue(Node->getNumValues()-1), notIn));

      //grab the arguments
      std::vector<unsigned> argvregs;
      //assert(Node->getNumOperands() < 8 && "Only 6 args supported");
      for(int i = 2, e = Node->getNumOperands(); i < e; ++i)
        argvregs.push_back(SelectExpr(N.getOperand(i)));

      //in reg args
      for(int i = 0, e = std::min(6, (int)argvregs.size()); i < e; ++i)
      {
        unsigned args_int[] = {Alpha::R16, Alpha::R17, Alpha::R18,
                               Alpha::R19, Alpha::R20, Alpha::R21};
        unsigned args_float[] = {Alpha::F16, Alpha::F17, Alpha::F18,
                                 Alpha::F19, Alpha::F20, Alpha::F21};
        switch(N.getOperand(i+2).getValueType()) {
        default:
          Node->dump();
          N.getOperand(i).Val->dump();
          std::cerr << "Type for " << i << " is: " <<
            N.getOperand(i+2).getValueType() << "\n";
          assert(0 && "Unknown value type for call");
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
          BuildMI(BB, Alpha::BIS, 2, args_int[i]).addReg(argvregs[i])
            .addReg(argvregs[i]);
          break;
        case MVT::f32:
        case MVT::f64:
          BuildMI(BB, Alpha::CPYS, 2, args_float[i]).addReg(argvregs[i])
            .addReg(argvregs[i]);
          break;
        }
      }
      //in mem args
      for (int i = 6, e = argvregs.size(); i < e; ++i)
      {
        switch(N.getOperand(i+2).getValueType()) {
        default:
          Node->dump();
          N.getOperand(i).Val->dump();
          std::cerr << "Type for " << i << " is: " <<
            N.getOperand(i+2).getValueType() << "\n";
          assert(0 && "Unknown value type for call");
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
          BuildMI(BB, Alpha::STQ, 3).addReg(argvregs[i]).addImm((i - 6) * 8)
            .addReg(Alpha::R30);
          break;
        case MVT::f32:
          BuildMI(BB, Alpha::STS, 3).addReg(argvregs[i]).addImm((i - 6) * 8)
            .addReg(Alpha::R30);
          break;
        case MVT::f64:
          BuildMI(BB, Alpha::STT, 3).addReg(argvregs[i]).addImm((i - 6) * 8)
            .addReg(Alpha::R30);
          break;
        }
      }
      //build the right kind of call
      if (GlobalAddressSDNode *GASD =
          dyn_cast<GlobalAddressSDNode>(N.getOperand(1)))
      {
        if (GASD->getGlobal()->isExternal()) {
          //use safe calling convention
          AlphaLowering.restoreGP(BB);
          has_sym = true;
          BuildMI(BB, Alpha::CALL, 1).addGlobalAddress(GASD->getGlobal());
        } else {
          //use PC relative branch call
          AlphaLowering.restoreGP(BB);
          BuildMI(BB, Alpha::BSR, 1, Alpha::R26)
            .addGlobalAddress(GASD->getGlobal(),true);
        }
      }
      else if (ExternalSymbolSDNode *ESSDN =
               dyn_cast<ExternalSymbolSDNode>(N.getOperand(1)))
      {
        AlphaLowering.restoreGP(BB);
        has_sym = true;
        BuildMI(BB, Alpha::CALL, 1).addExternalSymbol(ESSDN->getSymbol(), true);
      } else {
        //no need to restore GP as we are doing an indirect call
        Tmp1 = SelectExpr(N.getOperand(1));
        BuildMI(BB, Alpha::BIS, 2, Alpha::R27).addReg(Tmp1).addReg(Tmp1);
        BuildMI(BB, Alpha::JSR, 2, Alpha::R26).addReg(Alpha::R27).addImm(0);
      }

      //push the result into a virtual register

      switch (Node->getValueType(0)) {
      default: Node->dump(); assert(0 && "Unknown value type for call result!");
      case MVT::Other: return notIn;
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::i64:
        BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R0).addReg(Alpha::R0);
        break;
      case MVT::f32:
      case MVT::f64:
        BuildMI(BB, Alpha::CPYS, 2, Result).addReg(Alpha::F0).addReg(Alpha::F0);
        break;
      }
      return Result+N.ResNo;
    }

  case ISD::SIGN_EXTEND_INREG:
    {
      //do SDIV opt for all levels of ints if not dividing by a constant
      if (EnableAlphaIDIV && N.getOperand(0).getOpcode() == ISD::SDIV
          && N.getOperand(0).getOperand(1).getOpcode() != ISD::Constant)
      {
        unsigned Tmp4 = MakeReg(MVT::f64);
        unsigned Tmp5 = MakeReg(MVT::f64);
        unsigned Tmp6 = MakeReg(MVT::f64);
        unsigned Tmp7 = MakeReg(MVT::f64);
        unsigned Tmp8 = MakeReg(MVT::f64);
        unsigned Tmp9 = MakeReg(MVT::f64);

        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        MoveInt2FP(Tmp1, Tmp4, true);
        MoveInt2FP(Tmp2, Tmp5, true);
        BuildMI(BB, Alpha::CVTQT, 1, Tmp6).addReg(Tmp4);
        BuildMI(BB, Alpha::CVTQT, 1, Tmp7).addReg(Tmp5);
        BuildMI(BB, Alpha::DIVT, 2, Tmp8).addReg(Tmp6).addReg(Tmp7);
        BuildMI(BB, Alpha::CVTTQ, 1, Tmp9).addReg(Tmp8);
        MoveFP2Int(Tmp9, Result, true);
        return Result;
      }

      //Alpha has instructions for a bunch of signed 32 bit stuff
      if( dyn_cast<MVTSDNode>(Node)->getExtraValueType() == MVT::i32)
      {
        switch (N.getOperand(0).getOpcode()) {
        case ISD::ADD:
        case ISD::SUB:
        case ISD::MUL:
          {
            bool isAdd = N.getOperand(0).getOpcode() == ISD::ADD;
            bool isMul = N.getOperand(0).getOpcode() == ISD::MUL;
            //FIXME: first check for Scaled Adds and Subs!
            ConstantSDNode* CSD = NULL;
            if(!isMul && N.getOperand(0).getOperand(0).getOpcode() == ISD::SHL &&
               (CSD = dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(0).getOperand(1))) &&
               (CSD->getValue() == 2 || CSD->getValue() == 3))
            {
              bool use4 = CSD->getValue() == 2;
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(0));
              Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
              BuildMI(BB, isAdd?(use4?Alpha::S4ADDL:Alpha::S8ADDL):(use4?Alpha::S4SUBL:Alpha::S8SUBL),
                      2,Result).addReg(Tmp1).addReg(Tmp2);
            }
            else if(isAdd && N.getOperand(0).getOperand(1).getOpcode() == ISD::SHL &&
                    (CSD = dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(1).getOperand(1))) &&
                    (CSD->getValue() == 2 || CSD->getValue() == 3))
            {
              bool use4 = CSD->getValue() == 2;
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(1).getOperand(0));
              Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
              BuildMI(BB, use4?Alpha::S4ADDL:Alpha::S8ADDL, 2,Result).addReg(Tmp1).addReg(Tmp2);
            }
            else if(N.getOperand(0).getOperand(1).getOpcode() == ISD::Constant &&
               cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->getValue() <= 255)
            { //Normal imm add/sub
              Opc = isAdd ? Alpha::ADDLi : (isMul ? Alpha::MULLi : Alpha::SUBLi);
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
              Tmp2 = cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->getValue();
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
            }
            else
            { //Normal add/sub
              Opc = isAdd ? Alpha::ADDL : (isMul ? Alpha::MULL : Alpha::SUBL);
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
              Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
            }
            return Result;
          }
        default: break; //Fall Though;
        }
      } //Every thing else fall though too, including unhandled opcodes above
      Tmp1 = SelectExpr(N.getOperand(0));
      MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
      //std::cerr << "SrcT: " << MVN->getExtraValueType() << "\n";
      switch(MVN->getExtraValueType())
      {
      default:
        Node->dump();
        assert(0 && "Sign Extend InReg not there yet");
        break;
      case MVT::i32:
        {
          BuildMI(BB, Alpha::ADDLi, 2, Result).addReg(Tmp1).addImm(0);
          break;
        }
      case MVT::i16:
        BuildMI(BB, Alpha::SEXTW, 1, Result).addReg(Tmp1);
        break;
      case MVT::i8:
        BuildMI(BB, Alpha::SEXTB, 1, Result).addReg(Tmp1);
        break;
      case MVT::i1:
        Tmp2 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::ANDi, 2, Tmp2).addReg(Tmp1).addImm(1);
        BuildMI(BB, Alpha::SUBQ, 2, Result).addReg(Alpha::R31).addReg(Tmp2);
        break;
      }
      return Result;
    }

  case ISD::SETCC:
    {
      if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
        if (MVT::isInteger(SetCC->getOperand(0).getValueType())) {
          bool isConst = false;
          int dir;

          //Tmp1 = SelectExpr(N.getOperand(0));
          if(N.getOperand(1).getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 255)
            isConst = true;

          switch (SetCC->getCondition()) {
          default: Node->dump(); assert(0 && "Unknown integer comparison!");
          case ISD::SETEQ: 
            Opc = isConst ? Alpha::CMPEQi : Alpha::CMPEQ; dir=1; break;
          case ISD::SETLT:
            Opc = isConst ? Alpha::CMPLTi : Alpha::CMPLT; dir = 1; break;
          case ISD::SETLE:
            Opc = isConst ? Alpha::CMPLEi : Alpha::CMPLE; dir = 1; break;
          case ISD::SETGT: Opc = Alpha::CMPLT; dir = 2; break;
          case ISD::SETGE: Opc = Alpha::CMPLE; dir = 2; break;
          case ISD::SETULT:
            Opc = isConst ? Alpha::CMPULTi : Alpha::CMPULT; dir = 1; break;
          case ISD::SETUGT: Opc = Alpha::CMPULT; dir = 2; break;
          case ISD::SETULE:
            Opc = isConst ? Alpha::CMPULEi : Alpha::CMPULE; dir = 1; break;
          case ISD::SETUGE: Opc = Alpha::CMPULE; dir = 2; break;
          case ISD::SETNE: {//Handle this one special
            //std::cerr << "Alpha does not have a setne.\n";
            //abort();
            Tmp1 = SelectExpr(N.getOperand(0));
            Tmp2 = SelectExpr(N.getOperand(1));
            Tmp3 = MakeReg(MVT::i64);
            BuildMI(BB, Alpha::CMPEQ, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
            //Remeber we have the Inv for this CC
            CCInvMap[N] = Tmp3;
            //and invert
            BuildMI(BB, Alpha::CMPEQ, 2, Result).addReg(Alpha::R31).addReg(Tmp3);
            return Result;
          }
          }
          if (dir == 1) {
            Tmp1 = SelectExpr(N.getOperand(0));
            if (isConst) {
              Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
            } else {
              Tmp2 = SelectExpr(N.getOperand(1));
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
            }
          } else { //if (dir == 2) {
            Tmp1 = SelectExpr(N.getOperand(1));
            Tmp2 = SelectExpr(N.getOperand(0));
            BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
          }
        } else {
          //do the comparison
          Tmp1 = MakeReg(MVT::f64);
          bool inv = SelectFPSetCC(N, Tmp1);

          //now arrange for Result (int) to have a 1 or 0
          Tmp2 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::ADDQi, 2, Tmp2).addReg(Alpha::R31).addImm(1);
          Opc = inv?Alpha::CMOVNEi_FP:Alpha::CMOVEQi_FP;
          BuildMI(BB, Opc, 3, Result).addReg(Tmp2).addImm(0).addReg(Tmp1);
        }
      }
      return Result;
    }

  case ISD::CopyFromReg:
    {
      ++count_ins;

      // Make sure we generate both values.
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      SDOperand Chain   = N.getOperand(0);

      Select(Chain);
      unsigned r = dyn_cast<RegSDNode>(Node)->getReg();
      //std::cerr << "CopyFromReg " << Result << " = " << r << "\n";
      BuildMI(BB, Alpha::BIS, 2, Result).addReg(r).addReg(r);
      return Result;
    }

    //Most of the plain arithmetic and logic share the same form, and the same
    //constant immediate test
  case ISD::XOR:
    //Match Not
    if (N.getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(1))->getSignExtended() == -1)
      {
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Alpha::ORNOT, 2, Result).addReg(Alpha::R31).addReg(Tmp1);
        return Result;
      }
    //Fall through
  case ISD::AND:
    //handle zap
    if (opcode == ISD::AND && N.getOperand(1).getOpcode() == ISD::Constant)
    {
      uint64_t k = cast<ConstantSDNode>(N.getOperand(1))->getValue();
      unsigned int build = 0;
      for(int i = 0; i < 8; ++i)
      {
        if ((k & 0x00FF) == 0x00FF)
          build |= 1 << i;
        else if ((k & 0x00FF) != 0)
        { build = 0; break; }
        k >>= 8;
      }
      if (build)
      {
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Alpha::ZAPNOTi, 2, Result).addReg(Tmp1).addImm(build);
        return Result;
      }
    }
  case ISD::OR:
    //Check operand(0) == Not
    if (N.getOperand(0).getOpcode() == ISD::XOR &&
        N.getOperand(0).getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->getSignExtended() 
        == -1) {
      switch(opcode) {
        case ISD::AND: Opc = Alpha::BIC; break;
        case ISD::OR:  Opc = Alpha::ORNOT; break;
        case ISD::XOR: Opc = Alpha::EQV; break;
      }
      Tmp1 = SelectExpr(N.getOperand(1));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    //Check operand(1) == Not
    if (N.getOperand(1).getOpcode() == ISD::XOR &&
        N.getOperand(1).getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(1).getOperand(1))->getSignExtended()
        == -1) {
      switch(opcode) {
        case ISD::AND: Opc = Alpha::BIC; break;
        case ISD::OR:  Opc = Alpha::ORNOT; break;
        case ISD::XOR: Opc = Alpha::EQV; break;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1).getOperand(0));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    //Fall through
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::MUL:
    assert (DestType == MVT::i64 && "Only do arithmetic on i64s!");
    if(N.getOperand(1).getOpcode() == ISD::Constant &&
       cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 255)
    {
      switch(opcode) {
      case ISD::AND: Opc = Alpha::ANDi; break;
      case ISD::OR:  Opc = Alpha::BISi; break;
      case ISD::XOR: Opc = Alpha::XORi; break;
      case ISD::SHL: Opc = Alpha::SLi; break;
      case ISD::SRL: Opc = Alpha::SRLi; break;
      case ISD::SRA: Opc = Alpha::SRAi; break;
      case ISD::MUL: Opc = Alpha::MULQi; break;
      };
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      switch(opcode) {
      case ISD::AND: Opc = Alpha::AND; break;
      case ISD::OR:  Opc = Alpha::BIS; break;
      case ISD::XOR: Opc = Alpha::XOR; break;
      case ISD::SHL: Opc = Alpha::SL; break;
      case ISD::SRL: Opc = Alpha::SRL; break;
      case ISD::SRA: Opc = Alpha::SRA; break;
      case ISD::MUL: Opc = Alpha::MULQ; break;
      };
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::ADD:
  case ISD::SUB:
    {
      bool isAdd = opcode == ISD::ADD;

      //first check for Scaled Adds and Subs!
      //Valid for add and sub
      ConstantSDNode* CSD = NULL;
      if(N.getOperand(0).getOpcode() == ISD::SHL &&
         (CSD = dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(1))) &&
         (CSD->getValue() == 2 || CSD->getValue() == 3))
      {
        bool use4 = CSD->getValue() == 2;
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
        if ((CSD = dyn_cast<ConstantSDNode>(N.getOperand(1))) && CSD->getValue() <= 255)
          BuildMI(BB, isAdd?(use4?Alpha::S4ADDQi:Alpha::S8ADDQi):(use4?Alpha::S4SUBQi:Alpha::S8SUBQi),
                  2, Result).addReg(Tmp2).addImm(CSD->getValue());
        else {
          Tmp1 = SelectExpr(N.getOperand(1));
          BuildMI(BB, isAdd?(use4?Alpha::S4ADDQi:Alpha::S8ADDQi):(use4?Alpha::S4SUBQi:Alpha::S8SUBQi),
                  2, Result).addReg(Tmp2).addReg(Tmp1);
        }
      }
      //Position prevents subs
      else if(N.getOperand(1).getOpcode() == ISD::SHL && isAdd &&
              (CSD = dyn_cast<ConstantSDNode>(N.getOperand(1).getOperand(1))) &&
              (CSD->getValue() == 2 || CSD->getValue() == 3))
      {
        bool use4 = CSD->getValue() == 2;
        Tmp2 = SelectExpr(N.getOperand(1).getOperand(0));
        if ((CSD = dyn_cast<ConstantSDNode>(N.getOperand(0))) && CSD->getValue() <= 255)
          BuildMI(BB, use4?Alpha::S4ADDQi:Alpha::S8ADDQi, 2, Result).addReg(Tmp2)
            .addImm(CSD->getValue());
        else {
          Tmp1 = SelectExpr(N.getOperand(0));
          BuildMI(BB, use4?Alpha::S4ADDQ:Alpha::S8ADDQ, 2, Result).addReg(Tmp2).addReg(Tmp1);
        }
      }
      //small addi
      else if((CSD = dyn_cast<ConstantSDNode>(N.getOperand(1))) &&
              CSD->getValue() <= 255)
      { //Normal imm add/sub
        Opc = isAdd ? Alpha::ADDQi : Alpha::SUBQi;
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(CSD->getValue());
      }
      //larger addi
      else if((CSD = dyn_cast<ConstantSDNode>(N.getOperand(1))) &&
              CSD->getSignExtended() <= 32767 &&
              CSD->getSignExtended() >= -32767)
      { //LDA
        Tmp1 = SelectExpr(N.getOperand(0));
        Tmp2 = (long)CSD->getSignExtended();
        if (!isAdd)
          Tmp2 = -Tmp2;
        BuildMI(BB, Alpha::LDA, 2, Result).addImm(Tmp2).addReg(Tmp1);
      }
      //give up and do the operation
      else {
        //Normal add/sub
        Opc = isAdd ? Alpha::ADDQ : Alpha::SUBQ;
        Tmp1 = SelectExpr(N.getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(1));
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      }
      return Result;
    }

  case ISD::SDIV:
    {
      ConstantSDNode* CSD;
      //check if we can convert into a shift!
      if ((CSD = dyn_cast<ConstantSDNode>(N.getOperand(1).Val)) &&
          (int64_t)CSD->getSignExtended() != 0 &&
          ExactLog2(abs((int64_t)CSD->getSignExtended())) != 0)
      {
        unsigned k = ExactLog2(abs(CSD->getSignExtended()));
        Tmp1 = SelectExpr(N.getOperand(0));
        if (k == 1)
          Tmp2 = Tmp1;
        else
        {
          Tmp2 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::SRAi, 2, Tmp2).addReg(Tmp1).addImm(k - 1);
        }
        Tmp3 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::SRLi, 2, Tmp3).addReg(Tmp2).addImm(64-k);
        unsigned Tmp4 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::ADDQ, 2, Tmp4).addReg(Tmp3).addReg(Tmp1);
        if ((int64_t)CSD->getSignExtended() > 0)
          BuildMI(BB, Alpha::SRAi, 2, Result).addReg(Tmp4).addImm(k);
        else
        {
          unsigned Tmp5 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::SRAi, 2, Tmp5).addReg(Tmp4).addImm(k);
          BuildMI(BB, Alpha::SUBQ, 2, Result).addReg(Alpha::R31).addReg(Tmp5);
        }
        return Result;
      }
    }
    //Else fall through

  case ISD::UDIV:
    {
      ConstantSDNode* CSD;
      if ((CSD = dyn_cast<ConstantSDNode>(N.getOperand(1).Val)) &&
          ((int64_t)CSD->getSignExtended() >= 2 ||
           (int64_t)CSD->getSignExtended() <= -2))
      {
        // If this is a divide by constant, we can emit code using some magic
        // constants to implement it as a multiply instead.
        ExprMap.erase(N);
        if (opcode == ISD::SDIV)
          return SelectExpr(BuildSDIVSequence(N));
        else
          return SelectExpr(BuildUDIVSequence(N));
      }
    }
    //else fall though
  case ISD::UREM:
  case ISD::SREM:
    //FIXME: alpha really doesn't support any of these operations,
    // the ops are expanded into special library calls with
    // special calling conventions
    //Restore GP because it is a call after all...
    switch(opcode) {
    case ISD::UREM: Opc = Alpha::REMQU; break;
    case ISD::SREM: Opc = Alpha::REMQ; break;
    case ISD::UDIV: Opc = Alpha::DIVQU; break;
    case ISD::SDIV: Opc = Alpha::DIVQ; break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    //set up regs explicitly (helps Reg alloc)
    BuildMI(BB, Alpha::BIS, 2, Alpha::R24).addReg(Tmp1).addReg(Tmp1);
    BuildMI(BB, Alpha::BIS, 2, Alpha::R25).addReg(Tmp2).addReg(Tmp2);
    AlphaLowering.restoreGP(BB);
    BuildMI(BB, Opc, 2).addReg(Alpha::R24).addReg(Alpha::R25);
    BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R27).addReg(Alpha::R27);
    return Result;

  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:
    {
      assert (DestType == MVT::i64 && "only quads can be loaded to");
      MVT::ValueType SrcType = N.getOperand(0).getValueType();
      assert (SrcType == MVT::f32 || SrcType == MVT::f64);
      Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
      if (SrcType == MVT::f32)
        {
          Tmp2 = MakeReg(MVT::f64);
          BuildMI(BB, Alpha::CVTST, 1, Tmp2).addReg(Tmp1);
          Tmp1 = Tmp2;
        }
      Tmp2 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::CVTTQ, 1, Tmp2).addReg(Tmp1);
      MoveFP2Int(Tmp2, Result, true);

      return Result;
    }

  case ISD::SELECT:
    {
      //FIXME: look at parent to decide if intCC can be folded, or if setCC(FP)
      //and can save stack use
      //Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      //Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      //Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
      // Get the condition into the zero flag.
      //BuildMI(BB, Alpha::CMOVEQ, 2, Result).addReg(Tmp2).addReg(Tmp3).addReg(Tmp1);

      SDOperand CC = N.getOperand(0);
      SetCCSDNode* SetCC = dyn_cast<SetCCSDNode>(CC.Val);

      if (CC.getOpcode() == ISD::SETCC &&
          !MVT::isInteger(SetCC->getOperand(0).getValueType()))
      { //FP Setcc -> Int Select
        Tmp1 = MakeReg(MVT::f64);
        Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
        Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
        bool inv = SelectFPSetCC(CC, Tmp1);
        BuildMI(BB, inv?Alpha::CMOVNE_FP:Alpha::CMOVEQ_FP, 2, Result)
          .addReg(Tmp2).addReg(Tmp3).addReg(Tmp1);
        return Result;
      }
      if (CC.getOpcode() == ISD::SETCC) {
        //Int SetCC -> Select
        //Dropping the CC is only useful if we are comparing to 0
        if((SetCC->getOperand(1).getOpcode() == ISD::Constant &&
            cast<ConstantSDNode>(SetCC->getOperand(1))->getValue() == 0))
        {
          //figure out a few things
          bool useImm = N.getOperand(2).getOpcode() == ISD::Constant &&
            cast<ConstantSDNode>(N.getOperand(2))->getValue() <= 255;

          //Fix up CC
          ISD::CondCode cCode= SetCC->getCondition();
          if (useImm) //Invert sense to get Imm field right
            cCode = ISD::getSetCCInverse(cCode, true);

          //Choose the CMOV
          switch (cCode) {
          default: CC.Val->dump(); assert(0 && "Unknown integer comparison!");
          case ISD::SETEQ: Opc = useImm?Alpha::CMOVEQi:Alpha::CMOVEQ;     break;
          case ISD::SETLT: Opc = useImm?Alpha::CMOVLTi:Alpha::CMOVLT;     break;
          case ISD::SETLE: Opc = useImm?Alpha::CMOVLEi:Alpha::CMOVLE;     break;
          case ISD::SETGT: Opc = useImm?Alpha::CMOVGTi:Alpha::CMOVGT;     break;
          case ISD::SETGE: Opc = useImm?Alpha::CMOVGEi:Alpha::CMOVGE;     break;
          case ISD::SETULT: assert(0 && "unsigned < 0 is never true"); break;
          case ISD::SETUGT: Opc = useImm?Alpha::CMOVNEi:Alpha::CMOVNE;    break;
          //Technically you could have this CC
          case ISD::SETULE: Opc = useImm?Alpha::CMOVEQi:Alpha::CMOVEQ;    break;
          case ISD::SETUGE: assert(0 && "unsgined >= 0 is always true"); break;
          case ISD::SETNE:  Opc = useImm?Alpha::CMOVNEi:Alpha::CMOVNE;    break;
          }
          Tmp1 = SelectExpr(SetCC->getOperand(0)); //Cond

          if (useImm) {
            Tmp3 = SelectExpr(N.getOperand(1)); //Use if FALSE
            BuildMI(BB, Opc, 2, Result).addReg(Tmp3)
                .addImm(cast<ConstantSDNode>(N.getOperand(2))->getValue())
                .addReg(Tmp1);
          } else {
            Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
            Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
            BuildMI(BB, Opc, 2, Result).addReg(Tmp3).addReg(Tmp2).addReg(Tmp1);
          }
          return Result;
        }
        //Otherwise, fall though
      }
      Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
      BuildMI(BB, Alpha::CMOVEQ, 2, Result).addReg(Tmp2).addReg(Tmp3)
        .addReg(Tmp1);

      return Result;
    }

  case ISD::Constant:
    {
      int64_t val = (int64_t)cast<ConstantSDNode>(N)->getValue();
      if (val <= IMM_HIGH && val >= IMM_LOW) {
        BuildMI(BB, Alpha::LDA, 2, Result).addImm(val).addReg(Alpha::R31);
      }
      else if (val <= (int64_t)IMM_HIGH +(int64_t)IMM_HIGH* (int64_t)IMM_MULT &&
               val >= (int64_t)IMM_LOW + (int64_t)IMM_LOW * (int64_t)IMM_MULT) {
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAH, 2, Tmp1).addImm(getUpper16(val))
          .addReg(Alpha::R31);
        BuildMI(BB, Alpha::LDA, 2, Result).addImm(getLower16(val)).addReg(Tmp1);
      }
      else {
        MachineConstantPool *CP = BB->getParent()->getConstantPool();
        ConstantUInt *C = 
          ConstantUInt::get(Type::getPrimitiveType(Type::ULongTyID) , val);
        unsigned CPI = CP->getConstantPoolIndex(C);
        AlphaLowering.restoreGP(BB);
        has_sym = true;
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp1).addConstantPoolIndex(CPI)
          .addReg(Alpha::R29);
        BuildMI(BB, Alpha::LDQr, 2, Result).addConstantPoolIndex(CPI)
          .addReg(Tmp1);
      }
      return Result;
    }
  }

  return 0;
}

void AlphaISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;
  unsigned opcode = N.getOpcode();

  if (!ExprMap.insert(std::make_pair(N, notIn)).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (opcode) {

  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");

  case ISD::BRCOND: {
    SelectBranchCC(N);
    return;
  }

  case ISD::BR: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();

    Select(N.getOperand(0));
    BuildMI(BB, Alpha::BR, 1, Alpha::R31).addMBB(Dest);
    return;
  }

  case ISD::ImplicitDef:
    ++count_ins;
    Select(N.getOperand(0));
    BuildMI(BB, Alpha::IDEF, 0, cast<RegSDNode>(N)->getReg());
    return;

  case ISD::EntryToken: return;  // Noop

  case ISD::TokenFactor:
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));

    //N.Val->dump(); std::cerr << "\n";
    //assert(0 && "Node not handled yet!");

    return;

  case ISD::CopyToReg:
    ++count_outs;
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    Tmp2 = cast<RegSDNode>(N)->getReg();

    if (Tmp1 != Tmp2) {
      if (N.getOperand(1).getValueType() == MVT::f64 ||
          N.getOperand(1).getValueType() == MVT::f32)
        BuildMI(BB, Alpha::CPYS, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
      else
        BuildMI(BB, Alpha::BIS, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
    }
    return;

  case ISD::RET:
    ++count_outs;
    switch (N.getNumOperands()) {
    default:
      std::cerr << N.getNumOperands() << "\n";
      for (unsigned i = 0; i < N.getNumOperands(); ++i)
        std::cerr << N.getOperand(i).getValueType() << "\n";
      Node->dump();
      assert(0 && "Unknown return instruction!");
    case 2:
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      switch (N.getOperand(1).getValueType()) {
      default: Node->dump();
        assert(0 && "All other types should have been promoted!!");
      case MVT::f64:
      case MVT::f32:
        BuildMI(BB, Alpha::CPYS, 2, Alpha::F0).addReg(Tmp1).addReg(Tmp1);
        break;
      case MVT::i32:
      case MVT::i64:
        BuildMI(BB, Alpha::BIS, 2, Alpha::R0).addReg(Tmp1).addReg(Tmp1);
        break;
      }
      break;
    case 1:
      Select(N.getOperand(0));
      break;
    }
    // Just emit a 'ret' instruction
    BuildMI(BB, Alpha::RET, 1, Alpha::R31).addReg(AlphaLowering.getRA());
    return;

  case ISD::TRUNCSTORE:
  case ISD::STORE:
    {
      SDOperand Chain   = N.getOperand(0);
      SDOperand Value = N.getOperand(1);
      SDOperand Address = N.getOperand(2);
      Select(Chain);

      Tmp1 = SelectExpr(Value); //value

      if (opcode == ISD::STORE) {
        switch(Value.getValueType()) {
        default: assert(0 && "unknown Type in store");
        case MVT::i64: Opc = Alpha::STQ; break;
        case MVT::f64: Opc = Alpha::STT; break;
        case MVT::f32: Opc = Alpha::STS; break;
        }
      } else { //ISD::TRUNCSTORE
        switch(cast<MVTSDNode>(Node)->getExtraValueType()) {
        default: assert(0 && "unknown Type in store");
        case MVT::i1: //FIXME: DAG does not promote this load
        case MVT::i8: Opc = Alpha::STB; break;
        case MVT::i16: Opc = Alpha::STW; break;
        case MVT::i32: Opc = Alpha::STL; break;
        }
      }

      if (EnableAlphaLSMark)
      {
        int i = 
          getValueOffset(dyn_cast<SrcValueSDNode>(N.getOperand(3))->getValue());
        int j = getFunctionOffset(BB->getParent()->getFunction());
        BuildMI(BB, Alpha::MEMLABEL, 3).addImm(j).addImm(i).addImm(getUID());
      }

      if (Address.getOpcode() == ISD::GlobalAddress)
      {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        has_sym = true;
        BuildMI(BB, Opc, 2).addReg(Tmp1)
          .addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if(Address.getOpcode() == ISD::FrameIndex)
      {
        BuildMI(BB, Opc, 3).addReg(Tmp1)
          .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex())
          .addReg(Alpha::F31);
      }
      else
      {
        long offset;
        SelectAddr(Address, Tmp2, offset);
        BuildMI(BB, Opc, 3).addReg(Tmp1).addImm(offset).addReg(Tmp2);
      }
      return;
    }

  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD:
  case ISD::CopyFromReg:
  case ISD::TAILCALL:
  case ISD::CALL:
  case ISD::DYNAMIC_STACKALLOC:
    ExprMap.erase(N);
    SelectExpr(N);
    return;

  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();

    Opc = N.getOpcode() == ISD::CALLSEQ_START ? Alpha::ADJUSTSTACKDOWN :
      Alpha::ADJUSTSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;

  case ISD::PCMARKER:
    Select(N.getOperand(0)); //Chain
    BuildMI(BB, Alpha::PCLABEL, 2)
      .addImm( cast<ConstantSDNode>(N.getOperand(1))->getValue());
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createAlphaPatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createAlphaPatternInstructionSelector(TargetMachine &TM) {
  return new AlphaISel(TM);
}

