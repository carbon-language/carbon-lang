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
#include <set>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  AlphaTargetLowering - Alpha Implementation of the TargetLowering interface
namespace {
  class AlphaTargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    unsigned GP; //GOT vreg
  public:
    AlphaTargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Set up the TargetLowering object.
      //I am having problems with shr n ubyte 1
      setShiftAmountType(MVT::i64);
      setSetCCResultType(MVT::i64);
      
      addRegisterClass(MVT::i64, Alpha::GPRCRegisterClass);
      addRegisterClass(MVT::f64, Alpha::FPRCRegisterClass);
      addRegisterClass(MVT::f32, Alpha::FPRCRegisterClass);
      
      setOperationAction(ISD::EXTLOAD          , MVT::i1   , Promote);

      setOperationAction(ISD::ZEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::ZEXTLOAD         , MVT::i32  , Expand);

      setOperationAction(ISD::SEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i8   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i16  , Expand);

      setOperationAction(ISD::SREM             , MVT::f32  , Expand);
      setOperationAction(ISD::SREM             , MVT::f64  , Expand);

      setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
      setOperationAction(ISD::MEMSET           , MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY           , MVT::Other, Expand);

      computeRegisterProperties();
      
      addLegalFPImmediate(+0.0); //F31
      addLegalFPImmediate(-0.0); //-F31
    }

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);
    
    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, SDOperand Callee,
                ArgListTy &Args, SelectionDAG &DAG);
    
    virtual std::pair<SDOperand, SDOperand>
    LowerVAStart(SDOperand Chain, SelectionDAG &DAG);
    
    virtual std::pair<SDOperand,SDOperand>
    LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
                   const Type *ArgTy, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    void restoreGP(MachineBasicBlock* BB)
    {
      BuildMI(BB, Alpha::BIS, 2, Alpha::R29).addReg(GP).addReg(GP);
    }
  };
}

//http://www.cs.arizona.edu/computer.help/policy/DIGITAL_unix/AA-PY8AC-TET1_html/callCH3.html#BLOCK21

//For now, just use variable size stack frame format

//In a standard call, the first six items are passed in registers $16
//- $21 and/or registers $f16 - $f21. (See Section 4.1.2 for details
//of argument-to-register correspondence.) The remaining items are
//collected in a memory argument list that is a naturally aligned
//array of quadwords. In a standard call, this list, if present, must
//be passed at 0(SP).
//7 ... n  	  	  	0(SP) ... (n-7)*8(SP)

std::vector<SDOperand>
AlphaTargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) 
{
  std::vector<SDOperand> ArgValues;
  
  // //#define FP    $15
  // //#define RA    $26
  // //#define PV    $27
  // //#define GP    $29
  // //#define SP    $30
  
  //  assert(0 && "TODO");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo*MFI = MF.getFrameInfo();

  GP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  MachineBasicBlock& BB = MF.front();

  //Handle the return address
  //BuildMI(&BB, Alpha::IDEF, 0, Alpha::R26);

  unsigned args_int[] = {Alpha::R16, Alpha::R17, Alpha::R18, 
			 Alpha::R19, Alpha::R20, Alpha::R21};
  unsigned args_float[] = {Alpha::F16, Alpha::F17, Alpha::F18, 
			   Alpha::F19, Alpha::F20, Alpha::F21};
  unsigned argVreg[6];
  unsigned argPreg[6];
  unsigned argOpc[6];

  int count = 0;

  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
  {
    SDOperand newroot, argt;
    if (count  < 6) {
      switch (getValueType(I->getType())) {
      default: 
        std::cerr << "Unknown Type " << getValueType(I->getType()) << "\n"; 
        abort();
      case MVT::f64:
      case MVT::f32:
        BuildMI(&BB, Alpha::IDEF, 0, args_float[count]);
        argVreg[count] = 
          MF.getSSARegMap()->createVirtualRegister(
                             getRegClassFor(getValueType(I->getType())));
        argPreg[count] = args_float[count];
        argOpc[count] = Alpha::CPYS;
        argt = newroot = DAG.getCopyFromReg(argVreg[count], 
                                            getValueType(I->getType()), 
                                            DAG.getRoot());
        break;
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::i64:
        BuildMI(&BB, Alpha::IDEF, 0, args_int[count]);
        argVreg[count] = 
          MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
        argPreg[count] = args_int[count];
        argOpc[count] = Alpha::BIS;
        argt = newroot = 
          DAG.getCopyFromReg(argVreg[count], MVT::i64, DAG.getRoot());
        if (getValueType(I->getType()) != MVT::i64)
          argt = 
            DAG.getNode(ISD::TRUNCATE, getValueType(I->getType()), newroot);
        break;
      }
      ++count;
    } else { //more args
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(8, 8 * (count - 6));
        
      // Create the SelectionDAG nodes corresponding to a load 
      //from this parameter
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
      argt = newroot = DAG.getLoad(getValueType(I->getType()), 
                                   DAG.getEntryNode(), FIN);
    }
    DAG.setRoot(newroot.getValue(1));
    ArgValues.push_back(argt);
  }

  BuildMI(&BB, Alpha::IDEF, 0, Alpha::R29);
  BuildMI(&BB, Alpha::BIS, 2, GP).addReg(Alpha::R29).addReg(Alpha::R29);
  for (int i = 0; i < count; ++i) {
    if (argPreg[i] == Alpha::F16 || argPreg[i] == Alpha::F17 || 
        argPreg[i] == Alpha::F18 || argPreg[i] == Alpha::F19 || 
        argPreg[i] == Alpha::F20 || argPreg[i] == Alpha::F21)
    {
      assert(argOpc[i] == Alpha::CPYS && "Using BIS for a float??");
    }
    BuildMI(&BB, argOpc[i], 2, 
            argVreg[i]).addReg(argPreg[i]).addReg(argPreg[i]);
  }
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerCallTo(SDOperand Chain,
				 const Type *RetTy, SDOperand Callee,
				 ArgListTy &Args, SelectionDAG &DAG) {
  int NumBytes = 0;
  if (Args.size() > 6)
    NumBytes = (Args.size() - 6) * 8;

  Chain = DAG.getNode(ISD::ADJCALLSTACKDOWN, MVT::Other, Chain,
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
  Chain = DAG.getNode(ISD::ADJCALLSTACKUP, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG) {
  //vastart just returns the address of the VarArgsFrameIndex slot.
  return std::make_pair(DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64), Chain);
}

std::pair<SDOperand,SDOperand> AlphaTargetLowering::
LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
               const Type *ArgTy, SelectionDAG &DAG) {
  abort();
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
class ISel : public SelectionDAGISel {
  
  /// AlphaLowering - This object fully describes how to lower LLVM code to an
  /// Alpha-specific SelectionDAG.
  AlphaTargetLowering AlphaLowering;
  
  
  /// ExprMap - As shared expressions are codegen'd, we keep track of which
  /// vreg the value is produced in, so we only emit one copy of each compiled
  /// tree.
  static const unsigned notIn = (unsigned)(-1);
  std::map<SDOperand, unsigned> ExprMap;
  
  //CCInvMap sometimes (SetNE) we have the inverse CC code for free
  std::map<SDOperand, unsigned> CCInvMap;
  
public:
  ISel(TargetMachine &TM) : SelectionDAGISel(AlphaLowering), AlphaLowering(TM) 
  {}
  
  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
    // Codegen the basic block.
    Select(DAG.getRoot());
    
    // Clear state used for selection.
    ExprMap.clear();
    CCInvMap.clear();
  }
  
  unsigned SelectExpr(SDOperand N);
  unsigned SelectExprFP(SDOperand N, unsigned Result);
  void Select(SDOperand N);
  
  void SelectAddr(SDOperand N, unsigned& Reg, long& offset);
  void SelectBranchCC(SDOperand N);
};
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

//Check to see if the load is a constant offset from a base register
void ISel::SelectAddr(SDOperand N, unsigned& Reg, long& offset)
{
  unsigned opcode = N.getOpcode();
  if (opcode == ISD::ADD) {
    if(N.getOperand(1).getOpcode() == ISD::Constant && 
       cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 32767)
    { //Normal imm add
      Reg = SelectExpr(N.getOperand(0));
      offset = cast<ConstantSDNode>(N.getOperand(1))->getValue();
      return;
    }
    else if(N.getOperand(0).getOpcode() == ISD::Constant && 
            cast<ConstantSDNode>(N.getOperand(0))->getValue() <= 32767)
    {
      Reg = SelectExpr(N.getOperand(1));
      offset = cast<ConstantSDNode>(N.getOperand(0))->getValue();
      return;
    }
  }
  Reg = SelectExpr(N);
  offset = 0;
  return;
}

void ISel::SelectBranchCC(SDOperand N)
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
      bool isZero0 = false;
      bool isZero1 = false;
      bool isNE = false;

      if(SetCC->getOperand(0).getOpcode() == ISD::Constant &&
         cast<ConstantSDNode>(SetCC->getOperand(0))->getValue() == 0)
        isZero0 = true;
      if(SetCC->getOperand(1).getOpcode() == ISD::Constant &&
         cast<ConstantSDNode>(SetCC->getOperand(1))->getValue() == 0)
        isZero1 = true;
      if(SetCC->getCondition() == ISD::SETNE)
        isNE = true;

      if (isZero0) {
        switch (SetCC->getCondition()) {
        default: CC.Val->dump(); assert(0 && "Unknown integer comparison!");
        case ISD::SETEQ:  Opc = Alpha::BEQ; break;
        case ISD::SETLT:  Opc = Alpha::BGT; break;
        case ISD::SETLE:  Opc = Alpha::BGE; break;
        case ISD::SETGT:  Opc = Alpha::BLT; break;
        case ISD::SETGE:  Opc = Alpha::BLE; break;
        case ISD::SETULT: Opc = Alpha::BNE; break;
        case ISD::SETUGT: assert(0 && "0 > (unsigned) x is never true"); break;
        case ISD::SETULE: assert(0 && "0 <= (unsigned) x is always true"); break;
        case ISD::SETUGE: Opc = Alpha::BEQ; break; //Technically you could have this CC
        case ISD::SETNE:  Opc = Alpha::BNE; break;
        }
        unsigned Tmp1 = SelectExpr(SetCC->getOperand(1));
        BuildMI(BB, Opc, 2).addReg(Tmp1).addMBB(Dest);
        return;
      } else if (isZero1) {
        switch (SetCC->getCondition()) {
        default: CC.Val->dump(); assert(0 && "Unknown integer comparison!");
        case ISD::SETEQ:  Opc = Alpha::BEQ; break;
        case ISD::SETLT:  Opc = Alpha::BLT; break;
        case ISD::SETLE:  Opc = Alpha::BLE; break;
        case ISD::SETGT:  Opc = Alpha::BGT; break;
        case ISD::SETGE:  Opc = Alpha::BGE; break;
        case ISD::SETULT: assert(0 && "x (unsigned) < 0 is never true"); break;
        case ISD::SETUGT: Opc = Alpha::BNE; break;
        case ISD::SETULE: Opc = Alpha::BEQ; break; //Technically you could have this CC
        case ISD::SETUGE: assert(0 && "x (unsgined >= 0 is always true"); break;
        case ISD::SETNE:  Opc = Alpha::BNE; break;
        }
        unsigned Tmp1 = SelectExpr(SetCC->getOperand(0));
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
      //Any comparison between 2 values should be codegened as an folded branch, as moving
      //CC to the integer register is very expensive
      //for a cmp b: c = a - b;
      //a = b: c = 0
      //a < b: c < 0
      //a > b: c > 0
      unsigned Tmp1 = SelectExpr(SetCC->getOperand(0));
      unsigned Tmp2 = SelectExpr(SetCC->getOperand(1));
      unsigned Tmp3 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::SUBT, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);

      switch (SetCC->getCondition()) {
      default: CC.Val->dump(); assert(0 && "Unknown FP comparison!");
      case ISD::SETEQ: Opc = Alpha::FBEQ; break;
      case ISD::SETLT: Opc = Alpha::FBLT; break;
      case ISD::SETLE: Opc = Alpha::FBLE; break;
      case ISD::SETGT: Opc = Alpha::FBGT; break;
      case ISD::SETGE: Opc = Alpha::FBGE; break;
      case ISD::SETNE: Opc = Alpha::FBNE; break;
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

unsigned ISel::SelectExprFP(SDOperand N, unsigned Result)
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

  case ISD::SELECT:
    {
      Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE


      // Spill the cond to memory and reload it from there.
      unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
      MachineFunction *F = BB->getParent();
      int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);
      unsigned Tmp4 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::STQ, 3).addReg(Tmp1).addFrameIndex(FrameIdx).addReg(Alpha::F31);
      BuildMI(BB, Alpha::LDT, 2, Tmp4).addFrameIndex(FrameIdx).addReg(Alpha::F31);
      //now ideally, we don't have to do anything to the flag...
      // Get the condition into the zero flag.
      BuildMI(BB, Alpha::FCMOVEQ, 2, Result).addReg(Tmp2).addReg(Tmp3).addReg(Tmp4);
      return Result;
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

      if (Address.getOpcode() == ISD::GlobalAddress) {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        BuildMI(BB, Opc, 1, Result).addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        BuildMI(BB, Opc, 1, Result).addConstantPoolIndex(CP->getIndex());
      }
      else if(Address.getOpcode() == ISD::FrameIndex) {
        Tmp1 = cast<FrameIndexSDNode>(Address)->getIndex();
        BuildMI(BB, Opc, 2, Result).addFrameIndex(Tmp1).addReg(Alpha::F31);
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
        BuildMI(BB, Alpha::CPYS, 2, Result).addReg(Alpha::F31).addReg(Alpha::F31);
      } else if ( CN->isExactlyValue(-0.0)) {
        BuildMI(BB, Alpha::CPYSN, 2, Result).addReg(Alpha::F31).addReg(Alpha::F31);
      } else {
        abort();
      }
    }
    return Result;
    
  case ISD::MUL:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::SDIV:
    switch( opcode ) {
    case ISD::MUL: Opc = DestType == MVT::f64 ? Alpha::MULT : Alpha::MULS; break;
    case ISD::ADD: Opc = DestType == MVT::f64 ? Alpha::ADDT : Alpha::ADDS; break;
    case ISD::SUB: Opc = DestType == MVT::f64 ? Alpha::SUBT : Alpha::SUBS; break;
    case ISD::SDIV: Opc = DestType == MVT::f64 ? Alpha::DIVT : Alpha::DIVS; break;
    };
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
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
      
      if (Address.getOpcode() == ISD::GlobalAddress) {
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::LDS_SYM, 1, Tmp1).addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if (ConstantPoolSDNode *CP = 
               dyn_cast<ConstantPoolSDNode>(N.getOperand(1))) 
      {
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::LDS_SYM, 1, Tmp1).addConstantPoolIndex(CP->getIndex());
      }
      else if(Address.getOpcode() == ISD::FrameIndex) {
        Tmp2 = cast<FrameIndexSDNode>(Address)->getIndex();
        BuildMI(BB, Alpha::LDS, 2, Tmp1).addFrameIndex(Tmp2).addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp2, offset);
        BuildMI(BB, Alpha::LDS, 1, Tmp1).addImm(offset).addReg(Tmp2);
      }
      BuildMI(BB, Alpha::CVTST, 1, Result).addReg(Tmp1);
      return Result;
    }

  case ISD::UINT_TO_FP:
  case ISD::SINT_TO_FP:
    {
      assert (N.getOperand(0).getValueType() == MVT::i64 
              && "only quads can be loaded from");
      Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
      Tmp2 = MakeReg(MVT::f64);

      //The hard way:
      // Spill the integer to memory and reload it from there.
      unsigned Size = MVT::getSizeInBits(MVT::i64)/8;
      MachineFunction *F = BB->getParent();
      int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, Size);

      BuildMI(BB, Alpha::STQ, 3).addReg(Tmp1).addFrameIndex(FrameIdx).addReg(Alpha::F31);
      BuildMI(BB, Alpha::LDT, 2, Tmp2).addFrameIndex(FrameIdx).addReg(Alpha::F31);
      Opc = DestType == MVT::f64 ? Alpha::CVTQT : Alpha::CVTQS;
      BuildMI(BB, Opc, 1, Result).addReg(Tmp2);

      //The easy way: doesn't work
      //       //so these instructions are not supported on ev56
      //       Opc = DestType == MVT::f64 ? Alpha::ITOFT : Alpha::ITOFS;
      //       BuildMI(BB,  Opc, 1, Tmp2).addReg(Tmp1);
      //       Opc = DestType == MVT::f64 ? Alpha::CVTQT : Alpha::CVTQS;
      //       BuildMI(BB, Opc, 1, Result).addReg(Tmp1);

      return Result;
    }
  }
  assert(0 && "should not get here");
  return 0;
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  if (N.getOpcode() != ISD::CALL)
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

  if (DestType == MVT::f64 || DestType == MVT::f32 ||
      (
       (opcode == ISD::LOAD || opcode == ISD::CopyFromReg || 
        opcode == ISD::EXTLOAD) &&
       (N.getValue(0).getValueType() == MVT::f32 || 
        N.getValue(0).getValueType() == MVT::f64)
       )
      )
    return SelectExprFP(N, Result);

  switch (opcode) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");
 
  case ISD::ConstantPool:
    Tmp1 = cast<ConstantPoolSDNode>(N)->getIndex();
    AlphaLowering.restoreGP(BB);
    BuildMI(BB, Alpha::LDQ_SYM, 1, Result).addConstantPoolIndex(Tmp1);
    return Result;

  case ISD::FrameIndex:
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    BuildMI(BB, Alpha::LDA, 2, Result).addFrameIndex(Tmp1).addReg(Alpha::F31);
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

      if (Address.getOpcode() == ISD::GlobalAddress) {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        BuildMI(BB, Opc, 1, Result).addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        BuildMI(BB, Opc, 1, Result).addConstantPoolIndex(CP->getIndex());
      }
      else if(Address.getOpcode() == ISD::FrameIndex) {
        Tmp1 = cast<FrameIndexSDNode>(Address)->getIndex();
        BuildMI(BB, Opc, 2, Result).addFrameIndex(Tmp1).addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp1, offset);
        BuildMI(BB, Opc, 2, Result).addImm(offset).addReg(Tmp1);
      }
      return Result;
    }

  case ISD::GlobalAddress:
    AlphaLowering.restoreGP(BB);
    BuildMI(BB, Alpha::LOAD_ADDR, 1, Result)
      .addGlobalAddress(cast<GlobalAddressSDNode>(N)->getGlobal());
    return Result;

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
          BuildMI(BB, Alpha::BIS, 2, args_int[i]).addReg(argvregs[i]).addReg(argvregs[i]);
          break;
        case MVT::f32:
        case MVT::f64:
          BuildMI(BB, Alpha::CPYS, 2, args_float[i]).addReg(argvregs[i]).addReg(argvregs[i]);
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
          BuildMI(BB, Alpha::STQ, 3).addReg(argvregs[i]).addImm((i - 6) * 8).addReg(Alpha::R30);
          break;
        case MVT::f32:
          BuildMI(BB, Alpha::STS, 3).addReg(argvregs[i]).addImm((i - 6) * 8).addReg(Alpha::R30);
          break;
        case MVT::f64:
          BuildMI(BB, Alpha::STT, 3).addReg(argvregs[i]).addImm((i - 6) * 8).addReg(Alpha::R30);
          break;
        }
      }
      //build the right kind of call
      if (GlobalAddressSDNode *GASD =
          dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) 
      {
        //if (GASD->getGlobal()->isExternal()) {
          //use safe calling convention
          AlphaLowering.restoreGP(BB);
          BuildMI(BB, Alpha::CALL, 1).addGlobalAddress(GASD->getGlobal(),true);
          //} else {
          //use PC relative branch call
          //BuildMI(BB, Alpha::BSR, 1, Alpha::R26).addGlobalAddress(GASD->getGlobal(),true);
          //}
      } 
      else if (ExternalSymbolSDNode *ESSDN =
               dyn_cast<ExternalSymbolSDNode>(N.getOperand(1))) 
      {
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::CALL, 0).addExternalSymbol(ESSDN->getSymbol(), true);
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
    
  case ISD::SIGN_EXTEND:
    abort();
    
  case ISD::SIGN_EXTEND_INREG:
    {
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
            if(N.getOperand(0).getOperand(1).getOpcode() == ISD::Constant &&
               cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->getValue() <= 255)
            { //Normal imm add/sub
              Opc = isAdd ? Alpha::ADDLi : (isMul ? Alpha::MULLi : Alpha::SUBLi);
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
              Tmp2 = cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->getValue();
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
            }
            else
            { //Normal add/sub
              Opc = isAdd ? Alpha::ADDL : (isMul ? Alpha::MULLi : Alpha::SUBL);
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
        BuildMI(BB, Alpha::SUB, 2, Result).addReg(Alpha::F31).addReg(Tmp2);
        break;
      }
      return Result;
    }
  case ISD::ZERO_EXTEND_INREG:
    {
      Tmp1 = SelectExpr(N.getOperand(0));
      MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
      //std::cerr << "SrcT: " << MVN->getExtraValueType() << "\n";
      switch(MVN->getExtraValueType())
      {
      default:
        Node->dump();
        assert(0 && "Zero Extend InReg not there yet");
        break;
      case MVT::i32: Tmp2 = 0xf0; break;
      case MVT::i16: Tmp2 = 0xfc; break;
      case MVT::i8: Tmp2 = 0xfe; break;
      case MVT::i1: //handle this one special
        BuildMI(BB, Alpha::ANDi, 2, Result).addReg(Tmp1).addImm(1);
        return Result;
      }
      BuildMI(BB, Alpha::ZAPi, 2, Result).addReg(Tmp1).addImm(Tmp2);
      return Result;
    }
    
  case ISD::SETCC:
    {
      if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
        if (MVT::isInteger(SetCC->getOperand(0).getValueType())) {
          bool isConst1 = false;
          bool isConst2 = false;
          int dir;
	  
          //Tmp1 = SelectExpr(N.getOperand(0));
          if(N.getOperand(0).getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(N.getOperand(0))->getValue() <= 255)
            isConst1 = true;
          if(N.getOperand(1).getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 255)
            isConst2 = true;

          switch (SetCC->getCondition()) {
          default: Node->dump(); assert(0 && "Unknown integer comparison!");
          case ISD::SETEQ: Opc = Alpha::CMPEQ; dir=0; break;
          case ISD::SETLT: 
            Opc = isConst2 ? Alpha::CMPLTi : Alpha::CMPLT; dir = 1; break;
          case ISD::SETLE: 
            Opc = isConst2 ? Alpha::CMPLEi : Alpha::CMPLE; dir = 1; break;
          case ISD::SETGT: 
            Opc = isConst1 ? Alpha::CMPLTi : Alpha::CMPLT; dir = 2; break;
          case ISD::SETGE: 
            Opc = isConst1 ? Alpha::CMPLEi : Alpha::CMPLE; dir = 2; break;
          case ISD::SETULT: 
            Opc = isConst2 ? Alpha::CMPULTi : Alpha::CMPULT; dir = 1; break;
          case ISD::SETUGT: 
            Opc = isConst1 ? Alpha::CMPULTi : Alpha::CMPULT; dir = 2; break;
          case ISD::SETULE: 
            Opc = isConst2 ? Alpha::CMPULEi : Alpha::CMPULE; dir = 1; break;
          case ISD::SETUGE: 
            Opc = isConst1 ? Alpha::CMPULEi : Alpha::CMPULE; dir = 2; break;
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
            if (isConst2) {
              Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
            } else {
              Tmp2 = SelectExpr(N.getOperand(1));
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
            }
          } else if (dir == 2) {
            Tmp1 = SelectExpr(N.getOperand(1));
            if (isConst1) {
              Tmp2 = cast<ConstantSDNode>(N.getOperand(0))->getValue();
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
            } else {
              Tmp2 = SelectExpr(N.getOperand(0));
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
            }
          } else { //dir == 0
            if (isConst1) {
              Tmp1 = cast<ConstantSDNode>(N.getOperand(0))->getValue();
              Tmp2 = SelectExpr(N.getOperand(1));
              BuildMI(BB, Alpha::CMPEQi, 2, Result).addReg(Tmp2).addImm(Tmp1);
            } else if (isConst2) {
              Tmp1 = SelectExpr(N.getOperand(0));
              Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
              BuildMI(BB, Alpha::CMPEQi, 2, Result).addReg(Tmp1).addImm(Tmp2);
            } else {
              Tmp1 = SelectExpr(N.getOperand(0));
              Tmp2 = SelectExpr(N.getOperand(1));
              BuildMI(BB, Alpha::CMPEQ, 2, Result).addReg(Tmp1).addReg(Tmp2);
            }
          }
        } else {
          //assert(SetCC->getOperand(0).getValueType() != MVT::f32 && "SetCC f32 should have been promoted");
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
          
          Tmp1 = SelectExpr(N.getOperand(0));
          Tmp2 = SelectExpr(N.getOperand(1));
          //Can only compare doubles, and dag won't promote for me
          if (SetCC->getOperand(0).getValueType() == MVT::f32)
          {
            Tmp3 = MakeReg(MVT::f64);
            BuildMI(BB, Alpha::CVTST, 1, Tmp3).addReg(Tmp1);
            Tmp1 = Tmp3;
          }
          if (SetCC->getOperand(1).getValueType() == MVT::f32)
          {
            Tmp3 = MakeReg(MVT::f64);
            BuildMI(BB, Alpha::CVTST, 1, Tmp3).addReg(Tmp2);
            Tmp1 = Tmp2;
          }

          if (rev) std::swap(Tmp1, Tmp2);
          Tmp3 = MakeReg(MVT::f64);
          //do the comparison
          BuildMI(BB, Opc, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
          
          //now arrange for Result (int) to have a 1 or 0
          
          // Spill the FP to memory and reload it from there.
          unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
          MachineFunction *F = BB->getParent();
          int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);
          unsigned Tmp4 = MakeReg(MVT::f64);
          BuildMI(BB, Alpha::CVTTQ, 1, Tmp4).addReg(Tmp3);
          BuildMI(BB, Alpha::STT, 3).addReg(Tmp4).addFrameIndex(FrameIdx).addReg(Alpha::F31);
          unsigned Tmp5 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::LDQ, 2, Tmp5).addFrameIndex(FrameIdx).addReg(Alpha::F31);
	  
          //now, set result based on Tmp5
          //Set Tmp6 if fp cmp was false
          unsigned Tmp6 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::CMPEQ, 2, Tmp6).addReg(Tmp5).addReg(Alpha::R31);
          //and invert
          BuildMI(BB, Alpha::CMPEQ, 2, Result).addReg(Tmp6).addReg(Alpha::R31);
          
        }
        //       else
        //         {
        //           Node->dump();
        //           assert(0 && "Not a setcc in setcc");
        //         }
      }
      return Result;
    }
    
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
      BuildMI(BB, Alpha::BIS, 2, Result).addReg(r).addReg(r);
      return Result;
    }

    //Most of the plain arithmetic and logic share the same form, and the same 
    //constant immediate test
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
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

      //FIXME: first check for Scaled Adds and Subs!
      if(N.getOperand(1).getOpcode() == ISD::Constant &&
         cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 255)
      { //Normal imm add/sub
        Opc = isAdd ? Alpha::ADDQi : Alpha::SUBQi;
        Tmp1 = SelectExpr(N.getOperand(0));
        Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
      }
      else if(N.getOperand(1).getOpcode() == ISD::Constant &&
              cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 32767)
      { //LDA  //FIXME: expand the above condition a bit
        Tmp1 = SelectExpr(N.getOperand(0));
        Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
        if (!isAdd)
          Tmp2 = -Tmp2;
        BuildMI(BB, Alpha::LDA, 2, Result).addImm(Tmp2).addReg(Tmp1);
      } else {
        //Normal add/sub
        Opc = isAdd ? Alpha::ADDQ : Alpha::SUBQ;
        Tmp1 = SelectExpr(N.getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(1));
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      }
      return Result;
    }

  case ISD::UREM:
  case ISD::SREM:
  case ISD::SDIV:
  case ISD::UDIV:
    //FIXME: alpha really doesn't support any of these operations, 
    // the ops are expanded into special library calls with
    // special calling conventions
    //Restore GP because it is a call after all...
    switch(opcode) {
    case ISD::UREM: AlphaLowering.restoreGP(BB); Opc = Alpha::REMQU; break;
    case ISD::SREM: AlphaLowering.restoreGP(BB); Opc = Alpha::REMQ; break;
    case ISD::UDIV: AlphaLowering.restoreGP(BB); Opc = Alpha::DIVQU; break;
    case ISD::SDIV: AlphaLowering.restoreGP(BB); Opc = Alpha::DIVQ; break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:
    {
      assert (DestType == MVT::i64 && "only quads can be loaded to");
      MVT::ValueType SrcType = N.getOperand(0).getValueType();
      assert (SrcType == MVT::f32 || SrcType == MVT::f64);
      Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register

      //The hard way:
      // Spill the integer to memory and reload it from there.
      unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
      MachineFunction *F = BB->getParent();
      int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);

      //CVTTQ STT LDQ
      //CVTST CVTTQ STT LDQ
      if (SrcType == MVT::f32)
      {
        Tmp2 = MakeReg(MVT::f64);
        BuildMI(BB, Alpha::CVTST, 1, Tmp2).addReg(Tmp1);
        Tmp1 = Tmp2;
      }
      Tmp2 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::CVTTQ, 1, Tmp2).addReg(Tmp1);
      BuildMI(BB, Alpha::STT, 3).addReg(Tmp2).addFrameIndex(FrameIdx).addReg(Alpha::F31);
      BuildMI(BB, Alpha::LDQ, 2, Result).addFrameIndex(FrameIdx).addReg(Alpha::F31);
      
      return Result;
    }

    //     //  case ISD::FP_TO_UINT: 
 
  case ISD::SELECT:
    {
      Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
      // Get the condition into the zero flag.
      BuildMI(BB, Alpha::CMOVEQ, 2, Result).addReg(Tmp2).addReg(Tmp3).addReg(Tmp1);
      return Result;
    }

  case ISD::Constant:
    {
      unsigned long val = cast<ConstantSDNode>(N)->getValue();
      if (val < 32000 && (long)val > -32000)
        BuildMI(BB, Alpha::LOAD_IMM, 1, Result).addImm(val);
      else {
        MachineConstantPool *CP = BB->getParent()->getConstantPool();
        ConstantUInt *C = ConstantUInt::get(Type::getPrimitiveType(Type::ULongTyID) , val);
        unsigned CPI = CP->getConstantPoolIndex(C);
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::LDQ_SYM, 1, Result).addConstantPoolIndex(CPI);
      }
      return Result;
    }
  }

  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;
  unsigned opcode = N.getOpcode();

  // FIXME: Disable for our current expansion model!
  if (/*!N->hasOneUse() &&*/ !ExprMap.insert(std::make_pair(N, notIn)).second)
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
    //Tmp2 = AlphaLowering.getRetAddr();
    //BuildMI(BB, Alpha::BIS, 2, Alpha::R26).addReg(Tmp2).addReg(Tmp2);
    BuildMI(BB, Alpha::RETURN, 0); // Just emit a 'ret' instruction
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

      if (Address.getOpcode() == ISD::GlobalAddress)
      {
        AlphaLowering.restoreGP(BB);
        Opc = GetSymVersion(Opc);
        BuildMI(BB, Opc, 2).addReg(Tmp1).addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
      }
      else if(Address.getOpcode() == ISD::FrameIndex)
      {
        Tmp2 = cast<FrameIndexSDNode>(Address)->getIndex();
        BuildMI(BB, Opc, 3).addReg(Tmp1).addFrameIndex(Tmp2).addReg(Alpha::F31);
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
  case ISD::CALL:
    //   case ISD::DYNAMIC_STACKALLOC:
    ExprMap.erase(N);
    SelectExpr(N);
    return;

  case ISD::ADJCALLSTACKDOWN:
  case ISD::ADJCALLSTACKUP:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    
    Opc = N.getOpcode() == ISD::ADJCALLSTACKDOWN ? Alpha::ADJUSTSTACKDOWN :
      Alpha::ADJUSTSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createAlphaPatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createAlphaPatternInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);  
}
