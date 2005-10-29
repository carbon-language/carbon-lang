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
#include "AlphaTargetMachine.h"
#include "AlphaISelLowering.h"
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
  cl::opt<bool> EnableAlphaCount("enable-alpha-count",
    cl::desc("Print estimates on live ins and outs"),
    cl::Hidden);
  cl::opt<bool> EnableAlphaLSMark("enable-alpha-lsmark",
    cl::desc("Emit symbols to correlate Mem ops to LLVM Values"),
    cl::Hidden);
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

    virtual const char *getPassName() const {
      return "Alpha Pattern Instruction Selection";
    } 

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

  unsigned SelectExpr(SDOperand N);
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

static bool isSIntImmediate(SDOperand N, int64_t& Imm) {
  // test for constant
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // retrieve value
    Imm = CN->getSignExtended();
    // passes muster
    return true;
  }
  // not a constant
  return false;
}

// isSIntImmediateBounded - This method tests to see if a constant operand
// bounded s.t. low <= Imm <= high
// If so Imm will receive the 64 bit value.
static bool isSIntImmediateBounded(SDOperand N, int64_t& Imm, 
                                   int64_t low, int64_t high) {
  if (isSIntImmediate(N, Imm) && Imm <= high && Imm >= low)
    return true;
  return false;
}
static bool isUIntImmediate(SDOperand N, uint64_t& Imm) {
  // test for constant
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // retrieve value
    Imm = (uint64_t)CN->getValue();
    // passes muster
    return true;
  }
  // not a constant
  return false;
}

static bool isUIntImmediateBounded(SDOperand N, uint64_t& Imm, 
                                   uint64_t low, uint64_t high) {
  if (isUIntImmediate(N, Imm) && Imm <= high && Imm >= low)
    return true;
  return false;
}

static void getValueInfo(const Value* v, int& type, int& fun, int& offset)
{
  fun = type = offset = 0;
  if (v == NULL) {
    type = 0;
  } else if (const GlobalValue* GV = dyn_cast<GlobalValue>(v)) {
    type = 1;
    const Module* M = GV->getParent();
    for(Module::const_global_iterator ii = M->global_begin(); &*ii != GV; ++ii)
      ++offset;
  } else if (const Argument* Arg = dyn_cast<Argument>(v)) {
    type = 2;
    const Function* F = Arg->getParent();
    const Module* M = F->getParent();
    for(Module::const_iterator ii = M->begin(); &*ii != F; ++ii)
      ++fun;
    for(Function::const_arg_iterator ii = F->arg_begin(); &*ii != Arg; ++ii)
      ++offset;
  } else if (const Instruction* I = dyn_cast<Instruction>(v)) {
    assert(dyn_cast<PointerType>(I->getType()));
    type = 3;
    const BasicBlock* bb = I->getParent();
    const Function* F = bb->getParent();
    const Module* M = F->getParent();
    for(Module::const_iterator ii = M->begin(); &*ii != F; ++ii)
      ++fun;
    for(Function::const_iterator ii = F->begin(); &*ii != bb; ++ii)
      offset += ii->size();
    for(BasicBlock::const_iterator ii = bb->begin(); &*ii != I; ++ii)
      ++offset;
  } else if (const Constant* C = dyn_cast<Constant>(v)) {
    //Don't know how to look these up yet
    type = 0;
  } else {
    assert(0 && "Error in value marking");
  }
  //type = 4: register spilling
  //type = 5: global address loading or constant loading
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
  case Alpha::STB: return Alpha::STBr;
  case Alpha::STW: return Alpha::STWr;
  case Alpha::STL: return Alpha::STLr;
  case Alpha::STQ: return Alpha::STQr;
  case Alpha::STS: return Alpha::STSr;
  case Alpha::STT: return Alpha::STTr;

  }
}

void AlphaISel::MoveFP2Int(unsigned src, unsigned dst, bool isDouble)
{
  unsigned Opc;
  if (TLI.getTargetMachine().getSubtarget<AlphaSubtarget>().hasF2I()) {
    Opc = isDouble ? Alpha::FTOIT : Alpha::FTOIS;
    BuildMI(BB, Opc, 1, dst).addReg(src).addReg(Alpha::F31);
  } else {
    //The hard way:
    // Spill the integer to memory and reload it from there.
    unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
    MachineFunction *F = BB->getParent();
    int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);

    if (EnableAlphaLSMark)
      BuildMI(BB, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(0)
        .addImm(getUID());
    Opc = isDouble ? Alpha::STT : Alpha::STS;
    BuildMI(BB, Opc, 3).addReg(src).addFrameIndex(FrameIdx).addReg(Alpha::F31);

    if (EnableAlphaLSMark)
      BuildMI(BB, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(0)
        .addImm(getUID());
    Opc = isDouble ? Alpha::LDQ : Alpha::LDL;
    BuildMI(BB, Alpha::LDQ, 2, dst).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  }
}

void AlphaISel::MoveInt2FP(unsigned src, unsigned dst, bool isDouble)
{
  unsigned Opc;
  if (TLI.getTargetMachine().getSubtarget<AlphaSubtarget>().hasF2I()) {
    Opc = isDouble?Alpha::ITOFT:Alpha::ITOFS;
    BuildMI(BB, Opc, 1, dst).addReg(src).addReg(Alpha::R31);
  } else {
    //The hard way:
    // Spill the integer to memory and reload it from there.
    unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
    MachineFunction *F = BB->getParent();
    int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, 8);

    if (EnableAlphaLSMark)
      BuildMI(BB, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(0)
        .addImm(getUID());
    Opc = isDouble ? Alpha::STQ : Alpha::STL;
    BuildMI(BB, Opc, 3).addReg(src).addFrameIndex(FrameIdx).addReg(Alpha::F31);

    if (EnableAlphaLSMark)
      BuildMI(BB, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(0)
        .addImm(getUID());
    Opc = isDouble ? Alpha::LDT : Alpha::LDS;
    BuildMI(BB, Opc, 2, dst).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  }
}

bool AlphaISel::SelectFPSetCC(SDOperand N, unsigned dst)
{
  SDNode *SetCC = N.Val;
  unsigned Opc, Tmp1, Tmp2, Tmp3;
  ISD::CondCode CC = cast<CondCodeSDNode>(SetCC->getOperand(2))->get();
  bool rev = false;
  bool inv = false;

  switch (CC) {
  default: SetCC->dump(); assert(0 && "Unknown FP comparison!");
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
      BuildMI(BB, Alpha::CVTST, 1, Tmp3).addReg(Alpha::F31).addReg(Tmp1);
      Tmp1 = Tmp3;
    }
  if (SetCC->getOperand(1).getValueType() == MVT::f32)
    {
      //assert (0 && "Setcc On float?\n");
      std::cerr << "Setcc on float!\n";
      Tmp3 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::CVTST, 1, Tmp3).addReg(Alpha::F31).addReg(Tmp2);
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
    ISD::CondCode cCode= cast<CondCodeSDNode>(CC.getOperand(2))->get();
    if (MVT::isInteger(CC.getOperand(0).getValueType())) {
      //Dropping the CC is only useful if we are comparing to 0
      bool RightZero = CC.getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(CC.getOperand(1))->getValue() == 0;
      bool isNE = false;

      //Fix up CC
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
        unsigned Tmp1 = SelectExpr(CC.getOperand(0)); //Cond
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
      if ((CN = dyn_cast<ConstantFPSDNode>(CC.getOperand(1)))
          && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
        Tmp3 = SelectExpr(CC.getOperand(0));
      else if ((CN = dyn_cast<ConstantFPSDNode>(CC.getOperand(0)))
          && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
      {
        Tmp3 = SelectExpr(CC.getOperand(1));
        invTest = true;
      }
      else
      {
        unsigned Tmp1 = SelectExpr(CC.getOperand(0));
        unsigned Tmp2 = SelectExpr(CC.getOperand(1));
        bool isD = CC.getOperand(0).getValueType() == MVT::f64;
        Tmp3 = MakeReg(isD ? MVT::f64 : MVT::f32);
        BuildMI(BB, isD ? Alpha::SUBT : Alpha::SUBS, 2, Tmp3)
          .addReg(Tmp1).addReg(Tmp2);
      }

      switch (cCode) {
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

unsigned AlphaISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2 = 0, Tmp3;
  unsigned Opc = 0;
  unsigned opcode = N.getOpcode();
  int64_t SImm = 0;
  uint64_t UImm;

  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();
  bool isFP = DestType == MVT::f64 || DestType == MVT::f32;

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  switch(N.getOpcode()) {
  default:
    Reg = Result = (N.getValueType() != MVT::Other) ?
      MakeReg(N.getValueType()) : notIn;
      break;
  case ISD::AssertSext:
  case ISD::AssertZext:
    return Reg = SelectExpr(N.getOperand(0));
  case ISD::CALL:
  case ISD::TAILCALL:
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
    break;
  }

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
    if (isSIntImmediateBounded(N.getOperand(1), SImm, 0, 32767))
      BuildMI(BB, Alpha::LDA, 2, Alpha::R30).addImm(-SImm).addReg(Alpha::R30);
    else {
      Tmp1 = SelectExpr(N.getOperand(1));
      // Subtract size from stack pointer, thereby allocating some space.
      BuildMI(BB, Alpha::SUBQ, 2, Alpha::R30).addReg(Alpha::R30).addReg(Tmp1);
    }

    // Put a pointer to the space into the result register, by copying the stack
    // pointer.
    BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R30).addReg(Alpha::R30);
    return Result;

  case ISD::ConstantPool:
    Tmp1 = BB->getParent()->getConstantPool()->
       getConstantPoolIndex(cast<ConstantPoolSDNode>(N)->get());
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

      bool fpext = true;

      if (opcode == ISD::LOAD)
        switch (Node->getValueType(0)) {
        default: Node->dump(); assert(0 && "Bad load!");
        case MVT::i64: Opc = Alpha::LDQ; break;
        case MVT::f64: Opc = Alpha::LDT; break;
        case MVT::f32: Opc = Alpha::LDS; break;
        }
      else
        switch (cast<VTSDNode>(Node->getOperand(3))->getVT()) {
        default: Node->dump(); assert(0 && "Bad sign extend!");
        case MVT::i32: Opc = Alpha::LDL;
          assert(opcode != ISD::ZEXTLOAD && "Not sext"); break;
        case MVT::i16: Opc = Alpha::LDWU;
          assert(opcode != ISD::SEXTLOAD && "Not zext"); break;
        case MVT::i1: //FIXME: Treat i1 as i8 since there are problems otherwise
        case MVT::i8: Opc = Alpha::LDBU;
          assert(opcode != ISD::SEXTLOAD && "Not zext"); break;
        }

      int i, j, k;
      if (EnableAlphaLSMark)
        getValueInfo(dyn_cast<SrcValueSDNode>(N.getOperand(2))->getValue(),
                     i, j, k);

      GlobalAddressSDNode *GASD = dyn_cast<GlobalAddressSDNode>(Address);
      if (GASD && !GASD->getGlobal()->isExternal()) {
        Tmp1 = MakeReg(MVT::i64);
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp1)
          .addGlobalAddress(GASD->getGlobal()).addReg(Alpha::R29);
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
        BuildMI(BB, GetRelVersion(Opc), 2, Result)
          .addGlobalAddress(GASD->getGlobal()).addReg(Tmp1);
      } else if (ConstantPoolSDNode *CP =
                     dyn_cast<ConstantPoolSDNode>(Address)) {
        unsigned CPIdx = BB->getParent()->getConstantPool()->
             getConstantPoolIndex(CP->get());
        AlphaLowering.restoreGP(BB);
        has_sym = true;
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp1).addConstantPoolIndex(CPIdx)
          .addReg(Alpha::R29);
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
        BuildMI(BB, GetRelVersion(Opc), 2, Result)
          .addConstantPoolIndex(CPIdx).addReg(Tmp1);
      } else if(Address.getOpcode() == ISD::FrameIndex) {
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
        BuildMI(BB, Opc, 2, Result)
          .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex())
          .addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp1, offset);
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
        BuildMI(BB, Opc, 2, Result).addImm(offset).addReg(Tmp1);
      }
      return Result;
    }

  case ISD::GlobalAddress:
    AlphaLowering.restoreGP(BB);
    has_sym = true;

    Reg = Result = MakeReg(MVT::i64);

    if (EnableAlphaLSMark)
      BuildMI(BB, Alpha::MEMLABEL, 4).addImm(5).addImm(0).addImm(0)
        .addImm(getUID());

    BuildMI(BB, Alpha::LDQl, 2, Result)
      .addGlobalAddress(cast<GlobalAddressSDNode>(N)->getGlobal())
      .addReg(Alpha::R29);
    return Result;

  case ISD::ExternalSymbol:
    AlphaLowering.restoreGP(BB);
    has_sym = true;

    Reg = Result = MakeReg(MVT::i64);

    if (EnableAlphaLSMark)
      BuildMI(BB, Alpha::MEMLABEL, 4).addImm(5).addImm(0).addImm(0)
        .addImm(getUID());

    BuildMI(BB, Alpha::LDQl, 2, Result)
      .addExternalSymbol(cast<ExternalSymbolSDNode>(N)->getSymbol())
      .addReg(Alpha::R29);
    return Result;

  case ISD::TAILCALL:
  case ISD::CALL:
    {
      Select(N.getOperand(0));

      // The chain for this call is now lowered.
      ExprMap[N.getValue(Node->getNumValues()-1)] = notIn;

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
      GlobalAddressSDNode *GASD = dyn_cast<GlobalAddressSDNode>(N.getOperand(1));
      if (GASD && !GASD->getGlobal()->isExternal()) {
        //use PC relative branch call
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::BSR, 1, Alpha::R26)
          .addGlobalAddress(GASD->getGlobal(),true);
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
        BuildMI(BB, Alpha::CVTQT, 1, Tmp6).addReg(Alpha::F31).addReg(Tmp4);
        BuildMI(BB, Alpha::CVTQT, 1, Tmp7).addReg(Alpha::F31).addReg(Tmp5);
        BuildMI(BB, Alpha::DIVT, 2, Tmp8).addReg(Tmp6).addReg(Tmp7);
        BuildMI(BB, Alpha::CVTTQ, 1, Tmp9).addReg(Alpha::F31).addReg(Tmp8);
        MoveFP2Int(Tmp9, Result, true);
        return Result;
      }

      //Alpha has instructions for a bunch of signed 32 bit stuff
      if(cast<VTSDNode>(Node->getOperand(1))->getVT() == MVT::i32) {
        switch (N.getOperand(0).getOpcode()) {
        case ISD::ADD:
        case ISD::SUB:
        case ISD::MUL:
          {
            bool isAdd = N.getOperand(0).getOpcode() == ISD::ADD;
            bool isMul = N.getOperand(0).getOpcode() == ISD::MUL;
            //FIXME: first check for Scaled Adds and Subs!
            if(!isMul && N.getOperand(0).getOperand(0).getOpcode() == ISD::SHL &&
               isSIntImmediateBounded(N.getOperand(0).getOperand(0).getOperand(1), SImm, 2, 3))
            {
              bool use4 = SImm == 2;
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(0));
              Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
              BuildMI(BB, isAdd?(use4?Alpha::S4ADDL:Alpha::S8ADDL):(use4?Alpha::S4SUBL:Alpha::S8SUBL),
                      2,Result).addReg(Tmp1).addReg(Tmp2);
            }
            else if(isAdd && N.getOperand(0).getOperand(1).getOpcode() == ISD::SHL &&
                    isSIntImmediateBounded(N.getOperand(0).getOperand(1).getOperand(1), SImm, 2, 3))
            {
              bool use4 = SImm == 2;
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(1).getOperand(0));
              Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
              BuildMI(BB, use4?Alpha::S4ADDL:Alpha::S8ADDL, 2,Result).addReg(Tmp1).addReg(Tmp2);
            }
            else if(isSIntImmediateBounded(N.getOperand(0).getOperand(1), SImm, 0, 255)) 
            { //Normal imm add/sub
              Opc = isAdd ? Alpha::ADDLi : (isMul ? Alpha::MULLi : Alpha::SUBLi);
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(SImm);
            }
            else if(!isMul && isSIntImmediate(N.getOperand(0).getOperand(1), SImm) &&
                    (((SImm << 32) >> 32) >= -255) && (((SImm << 32) >> 32) <= 0))
            { //handle canonicalization
              Opc = isAdd ? Alpha::SUBLi : Alpha::ADDLi;
              Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
              SImm = 0 - ((SImm << 32) >> 32);
              assert(SImm >= 0 && SImm <= 255);
              BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(SImm);
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
      //std::cerr << "SrcT: " << MVN->getExtraValueType() << "\n";
      switch(cast<VTSDNode>(Node->getOperand(1))->getVT()) {
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
      ISD::CondCode CC = cast<CondCodeSDNode>(N.getOperand(2))->get();
      if (MVT::isInteger(N.getOperand(0).getValueType())) {
        bool isConst = false;
        int dir;

        //Tmp1 = SelectExpr(N.getOperand(0));
        if(isSIntImmediate(N.getOperand(1), SImm) && SImm <= 255 && SImm >= 0)
          isConst = true;

        switch (CC) {
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
            BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(SImm);
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
      unsigned r = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      //std::cerr << "CopyFromReg " << Result << " = " << r << "\n";
      if (MVT::isFloatingPoint(N.getValue(0).getValueType()))
        BuildMI(BB, Alpha::CPYS, 2, Result).addReg(r).addReg(r);
      else
        BuildMI(BB, Alpha::BIS, 2, Result).addReg(r).addReg(r);
      return Result;
    }

    //Most of the plain arithmetic and logic share the same form, and the same
    //constant immediate test
  case ISD::XOR:
    //Match Not
    if (isSIntImmediate(N.getOperand(1), SImm) && SImm == -1) {
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, Alpha::ORNOT, 2, Result).addReg(Alpha::R31).addReg(Tmp1);
      return Result;
    }
    //Fall through
  case ISD::AND:
    //handle zap
    if (opcode == ISD::AND && isUIntImmediate(N.getOperand(1), UImm))
    {
      unsigned int build = 0;
      for(int i = 0; i < 8; ++i)
      {
        if ((UImm & 0x00FF) == 0x00FF)
          build |= 1 << i;
        else if ((UImm & 0x00FF) != 0)
        { build = 0; break; }
        UImm >>= 8;
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
        isSIntImmediate(N.getOperand(0).getOperand(1), SImm) && SImm == -1) {
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
        isSIntImmediate(N.getOperand(1).getOperand(1), SImm) && SImm == -1) {
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
    if(isSIntImmediateBounded(N.getOperand(1), SImm, 0, 255)) {
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
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(SImm);
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
      if(N.getOperand(0).getOpcode() == ISD::SHL && 
         isSIntImmediate(N.getOperand(0).getOperand(1), SImm) &&
         (SImm == 2 || SImm == 3)) {
        bool use4 = SImm == 2;
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
        if (isSIntImmediateBounded(N.getOperand(1), SImm, 0, 255))
          BuildMI(BB, isAdd?(use4?Alpha::S4ADDQi:Alpha::S8ADDQi):(use4?Alpha::S4SUBQi:Alpha::S8SUBQi),
                  2, Result).addReg(Tmp2).addImm(SImm);
        else {
          Tmp1 = SelectExpr(N.getOperand(1));
          BuildMI(BB, isAdd?(use4?Alpha::S4ADDQi:Alpha::S8ADDQi):(use4?Alpha::S4SUBQi:Alpha::S8SUBQi),
                  2, Result).addReg(Tmp2).addReg(Tmp1);
        }
      }
      //Position prevents subs
      else if(N.getOperand(1).getOpcode() == ISD::SHL && isAdd &&
              isSIntImmediate(N.getOperand(1).getOperand(1), SImm) &&
              (SImm == 2 || SImm == 3)) {
        bool use4 = SImm == 2;
        Tmp2 = SelectExpr(N.getOperand(1).getOperand(0));
        if (isSIntImmediateBounded(N.getOperand(0), SImm, 0, 255))
          BuildMI(BB, use4?Alpha::S4ADDQi:Alpha::S8ADDQi, 2, Result).addReg(Tmp2).addImm(SImm);
        else {
          Tmp1 = SelectExpr(N.getOperand(0));
          BuildMI(BB, use4?Alpha::S4ADDQ:Alpha::S8ADDQ, 2, Result).addReg(Tmp2).addReg(Tmp1);
        }
      }
      //small addi
      else if(isSIntImmediateBounded(N.getOperand(1), SImm, 0, 255))
      { //Normal imm add/sub
        Opc = isAdd ? Alpha::ADDQi : Alpha::SUBQi;
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(SImm);
      }
      else if(isSIntImmediateBounded(N.getOperand(1), SImm, -255, 0))
      { //inverted imm add/sub
        Opc = isAdd ? Alpha::SUBQi : Alpha::ADDQi;
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(-SImm);
      }
      //larger addi
      else if(isSIntImmediateBounded(N.getOperand(1), SImm, -32767, 32767))
      { //LDA
        Tmp1 = SelectExpr(N.getOperand(0));
        if (!isAdd)
          SImm = -SImm;
        BuildMI(BB, Alpha::LDA, 2, Result).addImm(SImm).addReg(Tmp1);
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
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV: {
    if (opcode == ISD::FADD)
      Opc = DestType == MVT::f64 ? Alpha::ADDT : Alpha::ADDS;
    else if (opcode == ISD::FSUB)
      Opc = DestType == MVT::f64 ? Alpha::SUBT : Alpha::SUBS;
    else if (opcode == ISD::FMUL)
      Opc = DestType == MVT::f64 ? Alpha::MULT : Alpha::MULS;
    else
      Opc = DestType == MVT::f64 ? Alpha::DIVT : Alpha::DIVS;
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }
  case ISD::SDIV:
    {
      //check if we can convert into a shift!
      if (isSIntImmediate(N.getOperand(1), SImm) &&
          SImm != 0 && isPowerOf2_64(llabs(SImm))) {
        unsigned k = Log2_64(llabs(SImm));
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
        if (SImm > 0)
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
    //else fall though
  case ISD::UREM:
  case ISD::SREM: {
    const char* opstr = 0;
    switch(opcode) {
    case ISD::UREM: opstr = "__remqu"; break;
    case ISD::SREM: opstr = "__remq";  break;
    case ISD::UDIV: opstr = "__divqu"; break;
    case ISD::SDIV: opstr = "__divq";  break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    SDOperand Addr =
      ISelDAG->getExternalSymbol(opstr, AlphaLowering.getPointerTy());
    Tmp3 = SelectExpr(Addr);
    //set up regs explicitly (helps Reg alloc)
    BuildMI(BB, Alpha::BIS, 2, Alpha::R24).addReg(Tmp1).addReg(Tmp1);
    BuildMI(BB, Alpha::BIS, 2, Alpha::R25).addReg(Tmp2).addReg(Tmp2);
    BuildMI(BB, Alpha::BIS, 2, Alpha::R27).addReg(Tmp3).addReg(Tmp3);
    BuildMI(BB, Alpha::JSRs, 2, Alpha::R23).addReg(Alpha::R27).addImm(0);
    BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R27).addReg(Alpha::R27);
    return Result;
  }

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
          BuildMI(BB, Alpha::CVTST, 1, Tmp2).addReg(Alpha::F31).addReg(Tmp1);
          Tmp1 = Tmp2;
        }
      Tmp2 = MakeReg(MVT::f64);
      BuildMI(BB, Alpha::CVTTQ, 1, Tmp2).addReg(Alpha::F31).addReg(Tmp1);
      MoveFP2Int(Tmp2, Result, true);

      return Result;
    }

  case ISD::SELECT:
    if (isFP) {
      //Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      unsigned TV = SelectExpr(N.getOperand(1)); //Use if TRUE
      unsigned FV = SelectExpr(N.getOperand(2)); //Use if FALSE

      SDOperand CC = N.getOperand(0);

      if (CC.getOpcode() == ISD::SETCC &&
          !MVT::isInteger(CC.getOperand(0).getValueType())) {
        //FP Setcc -> Select yay!


        //for a cmp b: c = a - b;
        //a = b: c = 0
        //a < b: c < 0
        //a > b: c > 0

        bool invTest = false;
        unsigned Tmp3;

        ConstantFPSDNode *CN;
        if ((CN = dyn_cast<ConstantFPSDNode>(CC.getOperand(1)))
            && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
          Tmp3 = SelectExpr(CC.getOperand(0));
        else if ((CN = dyn_cast<ConstantFPSDNode>(CC.getOperand(0)))
                 && (CN->isExactlyValue(+0.0) || CN->isExactlyValue(-0.0)))
        {
          Tmp3 = SelectExpr(CC.getOperand(1));
          invTest = true;
        }
        else
        {
          unsigned Tmp1 = SelectExpr(CC.getOperand(0));
          unsigned Tmp2 = SelectExpr(CC.getOperand(1));
          bool isD = CC.getOperand(0).getValueType() == MVT::f64;
          Tmp3 = MakeReg(isD ? MVT::f64 : MVT::f32);
          BuildMI(BB, isD ? Alpha::SUBT : Alpha::SUBS, 2, Tmp3)
            .addReg(Tmp1).addReg(Tmp2);
        }

        switch (cast<CondCodeSDNode>(CC.getOperand(2))->get()) {
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
    } else {
      //FIXME: look at parent to decide if intCC can be folded, or if setCC(FP)
      //and can save stack use
      //Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      //Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      //Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
      // Get the condition into the zero flag.
      //BuildMI(BB, Alpha::CMOVEQ, 2, Result).addReg(Tmp2).addReg(Tmp3).addReg(Tmp1);

      SDOperand CC = N.getOperand(0);

      if (CC.getOpcode() == ISD::SETCC &&
          !MVT::isInteger(CC.getOperand(0).getValueType()))
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
        if(isSIntImmediateBounded(CC.getOperand(1), SImm, 0, 0)) {
          //figure out a few things
          bool useImm = isSIntImmediateBounded(N.getOperand(2), SImm, 0, 255);

          //Fix up CC
          ISD::CondCode cCode= cast<CondCodeSDNode>(CC.getOperand(2))->get();
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
          Tmp1 = SelectExpr(CC.getOperand(0)); //Cond

          if (useImm) {
            Tmp3 = SelectExpr(N.getOperand(1)); //Use if FALSE
            BuildMI(BB, Opc, 2, Result).addReg(Tmp3).addImm(SImm).addReg(Tmp1);
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
      int zero_extend_top = 0;
      if (val > 0 && (val & 0xFFFFFFFF00000000ULL) == 0 &&
          ((int32_t)val < 0)) {
        //try a small load and zero extend
        val = (int32_t)val;
        zero_extend_top = 15;
      }

      if (val <= IMM_HIGH && val >= IMM_LOW) {
        if(!zero_extend_top)
          BuildMI(BB, Alpha::LDA, 2, Result).addImm(val).addReg(Alpha::R31);
        else {
          Tmp1 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::LDA, 2, Tmp1).addImm(val).addReg(Alpha::R31);
          BuildMI(BB, Alpha::ZAPNOT, 2, Result).addReg(Tmp1).addImm(zero_extend_top);
        }
      }
      else if (val <= (int64_t)IMM_HIGH +(int64_t)IMM_HIGH* (int64_t)IMM_MULT &&
               val >= (int64_t)IMM_LOW + (int64_t)IMM_LOW * (int64_t)IMM_MULT) {
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAH, 2, Tmp1).addImm(getUpper16(val))
          .addReg(Alpha::R31);
        if (!zero_extend_top)
          BuildMI(BB, Alpha::LDA, 2, Result).addImm(getLower16(val)).addReg(Tmp1);
        else {
          Tmp3 = MakeReg(MVT::i64);
          BuildMI(BB, Alpha::LDA, 2, Tmp3).addImm(getLower16(val)).addReg(Tmp1);
          BuildMI(BB, Alpha::ZAPNOT, 2, Result).addReg(Tmp3).addImm(zero_extend_top);
        }
      }
      else {
        //re-get the val since we are going to mem anyway
        val = (int64_t)cast<ConstantSDNode>(N)->getValue();
        MachineConstantPool *CP = BB->getParent()->getConstantPool();
        ConstantUInt *C =
          ConstantUInt::get(Type::getPrimitiveType(Type::ULongTyID) , val);
        unsigned CPI = CP->getConstantPoolIndex(C);
        AlphaLowering.restoreGP(BB);
        has_sym = true;
        Tmp1 = MakeReg(MVT::i64);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp1).addConstantPoolIndex(CPI)
          .addReg(Alpha::R29);
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(5).addImm(0).addImm(0)
            .addImm(getUID());
        BuildMI(BB, Alpha::LDQr, 2, Result).addConstantPoolIndex(CPI)
          .addReg(Tmp1);
      }
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

  case ISD::FP_ROUND:
    assert (DestType == MVT::f32 &&
            N.getOperand(0).getValueType() == MVT::f64 &&
            "only f64 to f32 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Alpha::CVTTS, 1, Result).addReg(Alpha::F31).addReg(Tmp1);
    return Result;

  case ISD::FP_EXTEND:
    assert (DestType == MVT::f64 &&
            N.getOperand(0).getValueType() == MVT::f32 &&
            "only f32 to f64 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Alpha::CVTST, 1, Result).addReg(Alpha::F31).addReg(Tmp1);
    return Result;

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

  case ISD::SINT_TO_FP:
    {
      assert (N.getOperand(0).getValueType() == MVT::i64
              && "only quads can be loaded from");
      Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
      Tmp2 = MakeReg(MVT::f64);
      MoveInt2FP(Tmp1, Tmp2, true);
      Opc = DestType == MVT::f64 ? Alpha::CVTQT : Alpha::CVTQS;
      BuildMI(BB, Opc, 1, Result).addReg(Alpha::F31).addReg(Tmp2);
      return Result;
    }

  case ISD::AssertSext:
  case ISD::AssertZext:
    return SelectExpr(N.getOperand(0));

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
    BuildMI(BB, Alpha::IDEF, 0,
            cast<RegisterSDNode>(N.getOperand(1))->getReg());
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
    Tmp1 = SelectExpr(N.getOperand(2));
    Tmp2 = cast<RegisterSDNode>(N.getOperand(1))->getReg();

    if (Tmp1 != Tmp2) {
      if (N.getOperand(2).getValueType() == MVT::f64 ||
          N.getOperand(2).getValueType() == MVT::f32)
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
    AlphaLowering.restoreRA(BB);
    BuildMI(BB, Alpha::RET, 2, Alpha::R31).addReg(Alpha::R26).addImm(1);
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
        switch(cast<VTSDNode>(Node->getOperand(4))->getVT()) {
        default: assert(0 && "unknown Type in store");
        case MVT::i8: Opc = Alpha::STB; break;
        case MVT::i16: Opc = Alpha::STW; break;
        case MVT::i32: Opc = Alpha::STL; break;
        }
      }

      int i, j, k;
      if (EnableAlphaLSMark)
        getValueInfo(cast<SrcValueSDNode>(N.getOperand(3))->getValue(),
                     i, j, k);

      GlobalAddressSDNode *GASD = dyn_cast<GlobalAddressSDNode>(Address);
      if (GASD && !GASD->getGlobal()->isExternal()) {
        Tmp2 = MakeReg(MVT::i64);
        AlphaLowering.restoreGP(BB);
        BuildMI(BB, Alpha::LDAHr, 2, Tmp2)
          .addGlobalAddress(GASD->getGlobal()).addReg(Alpha::R29);
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
        BuildMI(BB, GetRelVersion(Opc), 3).addReg(Tmp1)
          .addGlobalAddress(GASD->getGlobal()).addReg(Tmp2);
      } else if(Address.getOpcode() == ISD::FrameIndex) {
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
        BuildMI(BB, Opc, 3).addReg(Tmp1)
          .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex())
          .addReg(Alpha::F31);
      } else {
        long offset;
        SelectAddr(Address, Tmp2, offset);
        if (EnableAlphaLSMark)
          BuildMI(BB, Alpha::MEMLABEL, 4).addImm(i).addImm(j).addImm(k)
            .addImm(getUID());
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

