//===-- AlphaISelLowering.cpp - Alpha DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AlphaISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "AlphaISelLowering.h"
#include "AlphaTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>

using namespace llvm;
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

  ad = llabs(d);
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
static SDOperand BuildSDIVSequence(SDOperand N, SelectionDAG* ISelDAG) {
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
static SDOperand BuildUDIVSequence(SDOperand N, SelectionDAG* ISelDAG) {
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

AlphaTargetLowering::AlphaTargetLowering(TargetMachine &TM) : TargetLowering(TM) {
  // Set up the TargetLowering object.
  //I am having problems with shr n ubyte 1
  setShiftAmountType(MVT::i64);
  setSetCCResultType(MVT::i64);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  
  addRegisterClass(MVT::i64, Alpha::GPRCRegisterClass);
  addRegisterClass(MVT::f64, Alpha::F8RCRegisterClass);
  addRegisterClass(MVT::f32, Alpha::F4RCRegisterClass);
  
  setOperationAction(ISD::BR_CC,        MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,    MVT::Other, Expand);
  
  setOperationAction(ISD::EXTLOAD, MVT::i1,  Promote);
  setOperationAction(ISD::EXTLOAD, MVT::f32, Expand);
  
  setOperationAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setOperationAction(ISD::ZEXTLOAD, MVT::i32, Expand);
  
  setOperationAction(ISD::SEXTLOAD, MVT::i1,  Promote);
  setOperationAction(ISD::SEXTLOAD, MVT::i8,  Expand);
  setOperationAction(ISD::SEXTLOAD, MVT::i16, Expand);
  
  setOperationAction(ISD::TRUNCSTORE, MVT::i1, Promote);

  setOperationAction(ISD::FREM, MVT::f32, Expand);
  setOperationAction(ISD::FREM, MVT::f64, Expand);
  
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);

  if (!TM.getSubtarget<AlphaSubtarget>().hasCT()) {
    setOperationAction(ISD::CTPOP    , MVT::i64  , Expand);
    setOperationAction(ISD::CTTZ     , MVT::i64  , Expand);
    setOperationAction(ISD::CTLZ     , MVT::i64  , Expand);
  }
  setOperationAction(ISD::BSWAP    , MVT::i64, Expand);
  setOperationAction(ISD::ROTL     , MVT::i64, Expand);
  setOperationAction(ISD::ROTR     , MVT::i64, Expand);
  
  setOperationAction(ISD::SREM     , MVT::i64, Custom);
  setOperationAction(ISD::UREM     , MVT::i64, Custom);
  setOperationAction(ISD::SDIV     , MVT::i64, Custom);
  setOperationAction(ISD::UDIV     , MVT::i64, Custom);
  
  setOperationAction(ISD::MEMMOVE  , MVT::Other, Expand);
  setOperationAction(ISD::MEMSET   , MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY   , MVT::Other, Expand);
  
  // We don't support sin/cos/sqrt
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);

  setOperationAction(ISD::FSQRT, MVT::f64, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  
  // FIXME: Alpha supports fcopysign natively!?
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  setOperationAction(ISD::SETCC, MVT::f32, Promote);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

  // Not implemented yet.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);

  // We want to legalize GlobalAddress and ConstantPool and
  // ExternalSymbols nodes into the appropriate instructions to
  // materialize the address.
  setOperationAction(ISD::GlobalAddress,  MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool,   MVT::i64, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i64, Custom);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAEND,   MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,  MVT::Other, Custom);
  setOperationAction(ISD::VAARG,   MVT::Other, Custom);
  setOperationAction(ISD::VAARG,   MVT::i32,   Custom);

  setStackPointerRegisterToSaveRestore(Alpha::R30);

  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  addLegalFPImmediate(+0.0); //F31
  addLegalFPImmediate(-0.0); //-F31

  computeRegisterProperties();

  useITOF = TM.getSubtarget<AlphaSubtarget>().hasF2I();
}

const char *AlphaTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case AlphaISD::ITOFT_: return "Alpha::ITOFT_";
  case AlphaISD::FTOIT_: return "Alpha::FTOIT_";
  case AlphaISD::CVTQT_: return "Alpha::CVTQT_";
  case AlphaISD::CVTQS_: return "Alpha::CVTQS_";
  case AlphaISD::CVTTQ_: return "Alpha::CVTTQ_";
  case AlphaISD::GPRelHi: return "Alpha::GPRelHi";
  case AlphaISD::GPRelLo: return "Alpha::GPRelLo";
  case AlphaISD::RelLit: return "Alpha::RelLit";
  case AlphaISD::GlobalBaseReg: return "Alpha::GlobalBaseReg";
  case AlphaISD::CALL:   return "Alpha::CALL";
  case AlphaISD::DivCall: return "Alpha::DivCall";
  }
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
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock& BB = MF.front();
  std::vector<SDOperand> ArgValues;

  unsigned args_int[] = {
    Alpha::R16, Alpha::R17, Alpha::R18, Alpha::R19, Alpha::R20, Alpha::R21};
  unsigned args_float[] = {
    Alpha::F16, Alpha::F17, Alpha::F18, Alpha::F19, Alpha::F20, Alpha::F21};

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
        args_float[count] = AddLiveIn(MF, args_float[count], getRegClassFor(VT));
        argt = DAG.getCopyFromReg(DAG.getRoot(), args_float[count], VT);
        DAG.setRoot(argt.getValue(1));
        break;
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::i64:
        args_int[count] = AddLiveIn(MF, args_int[count], getRegClassFor(MVT::i64));
        argt = DAG.getCopyFromReg(DAG.getRoot(), args_int[count], MVT::i64);
        DAG.setRoot(argt.getValue(1));
        if (VT != MVT::i64) {
          unsigned AssertOp = 
            I->getType()->isSigned() ? ISD::AssertSext : ISD::AssertZext;
          argt = DAG.getNode(AssertOp, MVT::i64, argt, 
                             DAG.getValueType(VT));
          argt = DAG.getNode(ISD::TRUNCATE, VT, argt);
        }
        break;
      }
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
      if (MRegisterInfo::isPhysicalRegister(args_int[i]))
        args_int[i] = AddLiveIn(MF, args_int[i], getRegClassFor(MVT::i64));
      SDOperand argt = DAG.getCopyFromReg(DAG.getRoot(), args_int[i], MVT::i64);
      int FI = MFI->CreateFixedObject(8, -8 * (6 - i));
      if (i == 0) VarArgsBase = FI;
      SDOperand SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getNode(ISD::STORE, MVT::Other, DAG.getRoot(), argt,
                               SDFI, DAG.getSrcValue(NULL)));

      if (MRegisterInfo::isPhysicalRegister(args_float[i]))
        args_float[i] = AddLiveIn(MF, args_float[i], getRegClassFor(MVT::f64));
      argt = DAG.getCopyFromReg(DAG.getRoot(), args_float[i], MVT::f64);
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

  //return the arguments+
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

  Chain = DAG.getCALLSEQ_START(Chain,
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
  MVT::ValueType ActualRetTyVT = RetTyVT;
  if (RetTyVT >= MVT::i1 && RetTyVT <= MVT::i32)
    ActualRetTyVT = MVT::i64;

  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(ActualRetTyVT);
  RetVals.push_back(MVT::Other);

  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.insert(Ops.end(), args_to_use.begin(), args_to_use.end());
  SDOperand TheCall = DAG.getNode(AlphaISD::CALL, RetVals, Ops);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  SDOperand RetVal = TheCall;

  if (RetTyVT != ActualRetTyVT) {
    RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext : ISD::AssertZext,
                         MVT::i64, RetVal, DAG.getValueType(RetTyVT));
    RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
  }

  return std::make_pair(RetVal, Chain);
}

void AlphaTargetLowering::restoreGP(MachineBasicBlock* BB)
{
  BuildMI(BB, Alpha::BIS, 2, Alpha::R29).addReg(GP).addReg(GP);
}
void AlphaTargetLowering::restoreRA(MachineBasicBlock* BB)
{
  BuildMI(BB, Alpha::BIS, 2, Alpha::R26).addReg(RA).addReg(RA);
}

static int getUID()
{
  static int id = 0;
  return ++id;
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand AlphaTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Wasn't expecting to be able to lower this!"); 
  case ISD::SINT_TO_FP: {
    assert(MVT::i64 == Op.getOperand(0).getValueType() && 
           "Unhandled SINT_TO_FP type in custom expander!");
    SDOperand LD;
    bool isDouble = MVT::f64 == Op.getValueType();
    if (useITOF) {
      LD = DAG.getNode(AlphaISD::ITOFT_, MVT::f64, Op.getOperand(0));
    } else {
      int FrameIdx =
        DAG.getMachineFunction().getFrameInfo()->CreateStackObject(8, 8);
      SDOperand FI = DAG.getFrameIndex(FrameIdx, MVT::i64);
      SDOperand ST = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                 Op.getOperand(0), FI, DAG.getSrcValue(0));
      LD = DAG.getLoad(MVT::f64, ST, FI, DAG.getSrcValue(0));
      }
    SDOperand FP = DAG.getNode(isDouble?AlphaISD::CVTQT_:AlphaISD::CVTQS_,
                               isDouble?MVT::f64:MVT::f32, LD);
    return FP;
  }
  case ISD::FP_TO_SINT: {
    bool isDouble = MVT::f64 == Op.getOperand(0).getValueType();
    SDOperand src = Op.getOperand(0);

    if (!isDouble) //Promote
      src = DAG.getNode(ISD::FP_EXTEND, MVT::f64, src);
    
    src = DAG.getNode(AlphaISD::CVTTQ_, MVT::f64, src);

    if (useITOF) {
      return DAG.getNode(AlphaISD::FTOIT_, MVT::i64, src);
    } else {
      int FrameIdx =
        DAG.getMachineFunction().getFrameInfo()->CreateStackObject(8, 8);
      SDOperand FI = DAG.getFrameIndex(FrameIdx, MVT::i64);
      SDOperand ST = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                 src, FI, DAG.getSrcValue(0));
      return DAG.getLoad(MVT::i64, ST, FI, DAG.getSrcValue(0));
      }
  }
  case ISD::ConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
    Constant *C = CP->get();
    SDOperand CPI = DAG.getTargetConstantPool(C, MVT::i64, CP->getAlignment());
    
    SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, CPI,
			       DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
    SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, CPI, Hi);
    return Lo;
  }
  case ISD::GlobalAddress: {
    GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
    GlobalValue *GV = GSDN->getGlobal();
    SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i64, GSDN->getOffset());

    if (!GV->hasWeakLinkage() && !GV->isExternal()) {
      SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, GA,
				 DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
      SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, GA, Hi);
      return Lo;
    } else
      return DAG.getNode(AlphaISD::RelLit, MVT::i64, GA, DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
  }
  case ISD::ExternalSymbol: {
    return DAG.getNode(AlphaISD::RelLit, MVT::i64, 
		       DAG.getTargetExternalSymbol(cast<ExternalSymbolSDNode>(Op)->getSymbol(), MVT::i64),
		       DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
  }

  case ISD::UREM:
  case ISD::SREM:
    //Expand only on constant case
    if (Op.getOperand(1).getOpcode() == ISD::Constant) {
      MVT::ValueType VT = Op.Val->getValueType(0);
      unsigned Opc = Op.Val->getOpcode() == ISD::UREM ? ISD::UDIV : ISD::SDIV;
      SDOperand Tmp1 = Op.Val->getOpcode() == ISD::UREM ?
	BuildUDIVSequence(Op, &DAG) :
	BuildSDIVSequence(Op, &DAG);
      Tmp1 = DAG.getNode(ISD::MUL, VT, Tmp1, Op.getOperand(1));
      Tmp1 = DAG.getNode(ISD::SUB, VT, Op.getOperand(0), Tmp1);
      return Tmp1;
    }
    //fall through
  case ISD::SDIV:
  case ISD::UDIV:
    if (MVT::isInteger(Op.getValueType())) {
      const char* opstr = 0;
      switch(Op.getOpcode()) {
      case ISD::UREM: opstr = "__remqu"; break;
      case ISD::SREM: opstr = "__remq";  break;
      case ISD::UDIV: opstr = "__divqu"; break;
      case ISD::SDIV: opstr = "__divq";  break;
      }
      SDOperand Tmp1 = Op.getOperand(0),
        Tmp2 = Op.getOperand(1),
        Addr = DAG.getExternalSymbol(opstr, MVT::i64);
      return DAG.getNode(AlphaISD::DivCall, MVT::i64, Addr, Tmp1, Tmp2);
    }
    break;

  case ISD::VAARG: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand VAListP = Op.getOperand(1);
    SDOperand VAListS = Op.getOperand(2);
    
    SDOperand Base = DAG.getLoad(MVT::i64, Chain, VAListP, VAListS);
    SDOperand Tmp = DAG.getNode(ISD::ADD, MVT::i64, VAListP,
                                DAG.getConstant(8, MVT::i64));
    SDOperand Offset = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Base.getValue(1),
                                      Tmp, DAG.getSrcValue(0), MVT::i32);
    SDOperand DataPtr = DAG.getNode(ISD::ADD, MVT::i64, Base, Offset);
    if (MVT::isFloatingPoint(Op.getValueType()))
    {
      //if fp && Offset < 6*8, then subtract 6*8 from DataPtr
      SDOperand FPDataPtr = DAG.getNode(ISD::SUB, MVT::i64, DataPtr,
                                        DAG.getConstant(8*6, MVT::i64));
      SDOperand CC = DAG.getSetCC(MVT::i64, Offset,
                                  DAG.getConstant(8*6, MVT::i64), ISD::SETLT);
      DataPtr = DAG.getNode(ISD::SELECT, MVT::i64, CC, FPDataPtr, DataPtr);
    }

    SDOperand NewOffset = DAG.getNode(ISD::ADD, MVT::i64, Offset,
                                      DAG.getConstant(8, MVT::i64));
    SDOperand Update = DAG.getNode(ISD::TRUNCSTORE, MVT::Other,
                                   Offset.getValue(1), NewOffset,
                                   Tmp, DAG.getSrcValue(0),
                                   DAG.getValueType(MVT::i32));
    
    SDOperand Result;
    if (Op.getValueType() == MVT::i32)
      Result = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Update, DataPtr,
                              DAG.getSrcValue(0), MVT::i32);
    else
      Result = DAG.getLoad(Op.getValueType(), Update, DataPtr, 
                           DAG.getSrcValue(0));
    return Result;
  }
  case ISD::VACOPY: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand DestP = Op.getOperand(1);
    SDOperand SrcP = Op.getOperand(2);
    SDOperand DestS = Op.getOperand(3);
    SDOperand SrcS = Op.getOperand(4);
    
    SDOperand Val = DAG.getLoad(getPointerTy(), Chain, SrcP, SrcS);
    SDOperand Result = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1), Val,
                                   DestP, DestS);
    SDOperand NP = DAG.getNode(ISD::ADD, MVT::i64, SrcP, 
                               DAG.getConstant(8, MVT::i64));
    Val = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Result, NP,
                         DAG.getSrcValue(0), MVT::i32);
    SDOperand NPD = DAG.getNode(ISD::ADD, MVT::i64, DestP,
                                DAG.getConstant(8, MVT::i64));
    return DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Val.getValue(1),
                       Val, NPD, DAG.getSrcValue(0),DAG.getValueType(MVT::i32));
  }
  case ISD::VASTART: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand VAListP = Op.getOperand(1);
    SDOperand VAListS = Op.getOperand(2);
    
    // vastart stores the address of the VarArgsBase and VarArgsOffset
    SDOperand FR  = DAG.getFrameIndex(VarArgsBase, MVT::i64);
    SDOperand S1  = DAG.getNode(ISD::STORE, MVT::Other, Chain, FR, VAListP,
                                VAListS);
    SDOperand SA2 = DAG.getNode(ISD::ADD, MVT::i64, VAListP,
                                DAG.getConstant(8, MVT::i64));
    return DAG.getNode(ISD::TRUNCSTORE, MVT::Other, S1,
                       DAG.getConstant(VarArgsOffset, MVT::i64), SA2,
                       DAG.getSrcValue(0), DAG.getValueType(MVT::i32));
  }
  }

  return SDOperand();
}

SDOperand AlphaTargetLowering::CustomPromoteOperation(SDOperand Op, 
                                                      SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i32 && 
         Op.getOpcode() == ISD::VAARG &&
         "Unknown node to custom promote!");
  
  // The code in LowerOperation already handles i32 vaarg
  return LowerOperation(Op, DAG);
}
