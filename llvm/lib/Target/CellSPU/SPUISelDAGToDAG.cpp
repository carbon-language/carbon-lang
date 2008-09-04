//===-- SPUISelDAGToDAG.cpp - CellSPU pattern matching inst selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for the Cell SPU,
// converting from a legalized dag to a SPU-target dag.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPUTargetMachine.h"
#include "SPUISelLowering.h"
#include "SPUHazardRecognizers.h"
#include "SPUFrameInfo.h"
#include "SPURegisterNames.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Compiler.h"
#include <queue>
#include <set>

using namespace llvm;

namespace {
  //! ConstantSDNode predicate for i32 sign-extended, 10-bit immediates
  bool
  isI64IntS10Immediate(ConstantSDNode *CN)
  {
    return isS10Constant(CN->getSignExtended());
  }

  //! ConstantSDNode predicate for i32 sign-extended, 10-bit immediates
  bool
  isI32IntS10Immediate(ConstantSDNode *CN)
  {
    return isS10Constant(CN->getSignExtended());
  }

#if 0
  //! SDNode predicate for sign-extended, 10-bit immediate values
  bool
  isI32IntS10Immediate(SDNode *N)
  {
    return (N->getOpcode() == ISD::Constant
            && isI32IntS10Immediate(cast<ConstantSDNode>(N)));
  }
#endif

  //! ConstantSDNode predicate for i32 unsigned 10-bit immediate values
  bool
  isI32IntU10Immediate(ConstantSDNode *CN)
  {
    return isU10Constant(CN->getSignExtended());
  }

  //! ConstantSDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntS10Immediate(ConstantSDNode *CN)
  {
    return isS10Constant(CN->getSignExtended());
  }

  //! SDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntS10Immediate(SDNode *N)
  {
    return (N->getOpcode() == ISD::Constant
            && isI16IntS10Immediate(cast<ConstantSDNode>(N)));
  }

  //! ConstantSDNode predicate for i16 unsigned 10-bit immediate values
  bool
  isI16IntU10Immediate(ConstantSDNode *CN)
  {
    return isU10Constant((short) CN->getValue());
  }

  //! SDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntU10Immediate(SDNode *N)
  {
    return (N->getOpcode() == ISD::Constant
            && isI16IntU10Immediate(cast<ConstantSDNode>(N)));
  }

  //! ConstantSDNode predicate for signed 16-bit values
  /*!
    \arg CN The constant SelectionDAG node holding the value
    \arg Imm The returned 16-bit value, if returning true

    This predicate tests the value in \a CN to see whether it can be
    represented as a 16-bit, sign-extended quantity. Returns true if
    this is the case.
   */
  bool
  isIntS16Immediate(ConstantSDNode *CN, short &Imm)
  {
    MVT vt = CN->getValueType(0);
    Imm = (short) CN->getValue();
    if (vt.getSimpleVT() >= MVT::i1 && vt.getSimpleVT() <= MVT::i16) {
      return true;
    } else if (vt == MVT::i32) {
      int32_t i_val = (int32_t) CN->getValue();
      short s_val = (short) i_val;
      return i_val == s_val;
    } else {
      int64_t i_val = (int64_t) CN->getValue();
      short s_val = (short) i_val;
      return i_val == s_val;
    }

    return false;
  }

  //! SDNode predicate for signed 16-bit values.
  bool
  isIntS16Immediate(SDNode *N, short &Imm)
  {
    return (N->getOpcode() == ISD::Constant
            && isIntS16Immediate(cast<ConstantSDNode>(N), Imm));
  }

  //! ConstantFPSDNode predicate for representing floats as 16-bit sign ext.
  static bool
  isFPS16Immediate(ConstantFPSDNode *FPN, short &Imm)
  {
    MVT vt = FPN->getValueType(0);
    if (vt == MVT::f32) {
      int val = FloatToBits(FPN->getValueAPF().convertToFloat());
      int sval = (int) ((val << 16) >> 16);
      Imm = (short) val;
      return val == sval;
    }

    return false;
  }

  bool
  isHighLow(const SDValue &Op) 
  {
    return (Op.getOpcode() == SPUISD::IndirectAddr
            && ((Op.getOperand(0).getOpcode() == SPUISD::Hi
                 && Op.getOperand(1).getOpcode() == SPUISD::Lo)
                || (Op.getOperand(0).getOpcode() == SPUISD::Lo
                    && Op.getOperand(1).getOpcode() == SPUISD::Hi)));
  }

  //===------------------------------------------------------------------===//
  //! MVT to "useful stuff" mapping structure:

  struct valtype_map_s {
    MVT VT;
    unsigned ldresult_ins;      /// LDRESULT instruction (0 = undefined)
    bool ldresult_imm;          /// LDRESULT instruction requires immediate?
    int prefslot_byte;          /// Byte offset of the "preferred" slot
  };

  const valtype_map_s valtype_map[] = {
    { MVT::i1,    0,            false, 3 },
    { MVT::i8,    SPU::ORBIr8,  true,  3 },
    { MVT::i16,   SPU::ORHIr16, true,  2 },
    { MVT::i32,   SPU::ORIr32,  true,  0 },
    { MVT::i64,   SPU::ORr64,   false, 0 },
    { MVT::f32,   SPU::ORf32,   false, 0 },
    { MVT::f64,   SPU::ORf64,   false, 0 },
    // vector types... (sigh!)
    { MVT::v16i8, 0,            false, 0 },
    { MVT::v8i16, 0,            false, 0 },
    { MVT::v4i32, 0,            false, 0 },
    { MVT::v2i64, 0,            false, 0 },
    { MVT::v4f32, 0,            false, 0 },
    { MVT::v2f64, 0,            false, 0 }
  };

  const size_t n_valtype_map = sizeof(valtype_map) / sizeof(valtype_map[0]);

  const valtype_map_s *getValueTypeMapEntry(MVT VT)
  {
    const valtype_map_s *retval = 0;
    for (size_t i = 0; i < n_valtype_map; ++i) {
      if (valtype_map[i].VT == VT) {
        retval = valtype_map + i;
        break;
      }
    }


#ifndef NDEBUG
    if (retval == 0) {
      cerr << "SPUISelDAGToDAG.cpp: getValueTypeMapEntry returns NULL for "
           << VT.getMVTString()
           << "\n";
      abort();
    }
#endif

    return retval;
  }
}

namespace {

//===--------------------------------------------------------------------===//
/// SPUDAGToDAGISel - Cell SPU-specific code to select SPU machine
/// instructions for SelectionDAG operations.
///
class SPUDAGToDAGISel :
  public SelectionDAGISel
{
  SPUTargetMachine &TM;
  SPUTargetLowering &SPUtli;
  unsigned GlobalBaseReg;

public:
  explicit SPUDAGToDAGISel(SPUTargetMachine &tm) :
    SelectionDAGISel(*tm.getTargetLowering()),
    TM(tm),
    SPUtli(*tm.getTargetLowering())
  {}
    
  virtual bool runOnFunction(Function &Fn) {
    // Make sure we re-emit a set of the global base reg if necessary
    GlobalBaseReg = 0;
    SelectionDAGISel::runOnFunction(Fn);
    return true;
  }
   
  /// getI32Imm - Return a target constant with the specified value, of type
  /// i32.
  inline SDValue getI32Imm(uint32_t Imm) {
    return CurDAG->getTargetConstant(Imm, MVT::i32);
  }

  /// getI64Imm - Return a target constant with the specified value, of type
  /// i64.
  inline SDValue getI64Imm(uint64_t Imm) {
    return CurDAG->getTargetConstant(Imm, MVT::i64);
  }
    
  /// getSmallIPtrImm - Return a target constant of pointer type.
  inline SDValue getSmallIPtrImm(unsigned Imm) {
    return CurDAG->getTargetConstant(Imm, SPUtli.getPointerTy());
  }

  /// Select - Convert the specified operand from a target-independent to a
  /// target-specific node if it hasn't already been changed.
  SDNode *Select(SDValue Op);

  //! Returns true if the address N is an A-form (local store) address
  bool SelectAFormAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Index);

  //! D-form address predicate
  bool SelectDFormAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Index);

  /// Alternate D-form address using i7 offset predicate
  bool SelectDForm2Addr(SDValue Op, SDValue N, SDValue &Disp,
                        SDValue &Base);

  /// D-form address selection workhorse
  bool DFormAddressPredicate(SDValue Op, SDValue N, SDValue &Disp,
                             SDValue &Base, int minOffset, int maxOffset);

  //! Address predicate if N can be expressed as an indexed [r+r] operation.
  bool SelectXFormAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Index);

  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps) {
    SDValue Op0, Op1;
    switch (ConstraintCode) {
    default: return true;
    case 'm':   // memory
      if (!SelectDFormAddr(Op, Op, Op0, Op1) 
          && !SelectAFormAddr(Op, Op, Op0, Op1))
        SelectXFormAddr(Op, Op, Op0, Op1);
      break;
    case 'o':   // offsetable
      if (!SelectDFormAddr(Op, Op, Op0, Op1)
          && !SelectAFormAddr(Op, Op, Op0, Op1)) {
        Op0 = Op;
        AddToISelQueue(Op0);     // r+0.
        Op1 = getSmallIPtrImm(0);
      }
      break;
    case 'v':   // not offsetable
#if 1
      assert(0 && "InlineAsmMemoryOperand 'v' constraint not handled.");
#else
      SelectAddrIdxOnly(Op, Op, Op0, Op1);
#endif
      break;
    }
      
    OutOps.push_back(Op0);
    OutOps.push_back(Op1);
    return false;
  }

  /// InstructionSelect - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelect();

  virtual const char *getPassName() const {
    return "Cell SPU DAG->DAG Pattern Instruction Selection";
  } 
    
  /// CreateTargetHazardRecognizer - Return the hazard recognizer to use for
  /// this target when scheduling the DAG.
  virtual HazardRecognizer *CreateTargetHazardRecognizer() {
    const TargetInstrInfo *II = TM.getInstrInfo();
    assert(II && "No InstrInfo?");
    return new SPUHazardRecognizer(*II); 
  }

  // Include the pieces autogenerated from the target description.
#include "SPUGenDAGISel.inc"
};

}

/// InstructionSelect - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void
SPUDAGToDAGISel::InstructionSelect()
{
  DEBUG(BB->dump());

  // Select target instructions for the DAG.
  SelectRoot();
  CurDAG->RemoveDeadNodes();
}

/*!
 \arg Op The ISD instructio operand
 \arg N The address to be tested
 \arg Base The base address
 \arg Index The base address index
 */
bool
SPUDAGToDAGISel::SelectAFormAddr(SDValue Op, SDValue N, SDValue &Base,
                    SDValue &Index) {
  // These match the addr256k operand type:
  MVT OffsVT = MVT::i16;
  SDValue Zero = CurDAG->getTargetConstant(0, OffsVT);

  switch (N.getOpcode()) {
  case ISD::Constant:
  case ISD::ConstantPool:
  case ISD::GlobalAddress:
    cerr << "SPU SelectAFormAddr: Constant/Pool/Global not lowered.\n";
    abort();
    /*NOTREACHED*/

  case ISD::TargetConstant:
  case ISD::TargetGlobalAddress:
  case ISD::TargetJumpTable:
    cerr << "SPUSelectAFormAddr: Target Constant/Pool/Global not wrapped as "
         << "A-form address.\n";
    abort();
    /*NOTREACHED*/

  case SPUISD::AFormAddr: 
    // Just load from memory if there's only a single use of the location,
    // otherwise, this will get handled below with D-form offset addresses
    if (N.hasOneUse()) {
      SDValue Op0 = N.getOperand(0);
      switch (Op0.getOpcode()) {
      case ISD::TargetConstantPool:
      case ISD::TargetJumpTable:
        Base = Op0;
        Index = Zero;
        return true;

      case ISD::TargetGlobalAddress: {
        GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op0);
        GlobalValue *GV = GSDN->getGlobal();
        if (GV->getAlignment() == 16) {
          Base = Op0;
          Index = Zero;
          return true;
        }
        break;
      }
      }
    }
    break;
  }
  return false;
}

bool 
SPUDAGToDAGISel::SelectDForm2Addr(SDValue Op, SDValue N, SDValue &Disp,
                                  SDValue &Base) {
  const int minDForm2Offset = -(1 << 7);
  const int maxDForm2Offset = (1 << 7) - 1;
  return DFormAddressPredicate(Op, N, Disp, Base, minDForm2Offset,
                               maxDForm2Offset);
}

/*!
  \arg Op The ISD instruction (ignored)
  \arg N The address to be tested
  \arg Base Base address register/pointer
  \arg Index Base address index

  Examine the input address by a base register plus a signed 10-bit
  displacement, [r+I10] (D-form address).

  \return true if \a N is a D-form address with \a Base and \a Index set
  to non-empty SDValue instances.
*/
bool
SPUDAGToDAGISel::SelectDFormAddr(SDValue Op, SDValue N, SDValue &Base,
                                 SDValue &Index) {
  return DFormAddressPredicate(Op, N, Base, Index,
                              SPUFrameInfo::minFrameOffset(),
                              SPUFrameInfo::maxFrameOffset());
}

bool
SPUDAGToDAGISel::DFormAddressPredicate(SDValue Op, SDValue N, SDValue &Base,
                                      SDValue &Index, int minOffset,
                                      int maxOffset) {
  unsigned Opc = N.getOpcode();
  MVT PtrTy = SPUtli.getPointerTy();

  if (Opc == ISD::FrameIndex) {
    // Stack frame index must be less than 512 (divided by 16):
    FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N);
    int FI = int(FIN->getIndex());
    DEBUG(cerr << "SelectDFormAddr: ISD::FrameIndex = "
               << FI << "\n");
    if (SPUFrameInfo::FItoStackOffset(FI) < maxOffset) {
      Base = CurDAG->getTargetConstant(0, PtrTy);
      Index = CurDAG->getTargetFrameIndex(FI, PtrTy);
      return true;
    }
  } else if (Opc == ISD::ADD) {
    // Generated by getelementptr
    const SDValue Op0 = N.getOperand(0);
    const SDValue Op1 = N.getOperand(1);

    if ((Op0.getOpcode() == SPUISD::Hi && Op1.getOpcode() == SPUISD::Lo)
        || (Op1.getOpcode() == SPUISD::Hi && Op0.getOpcode() == SPUISD::Lo)) {
      Base = CurDAG->getTargetConstant(0, PtrTy);
      Index = N;
      return true;
    } else if (Op1.getOpcode() == ISD::Constant
               || Op1.getOpcode() == ISD::TargetConstant) {
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1);
      int32_t offset = int32_t(CN->getSignExtended());

      if (Op0.getOpcode() == ISD::FrameIndex) {
        FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Op0);
        int FI = int(FIN->getIndex());
        DEBUG(cerr << "SelectDFormAddr: ISD::ADD offset = " << offset
                   << " frame index = " << FI << "\n");

        if (SPUFrameInfo::FItoStackOffset(FI) < maxOffset) {
          Base = CurDAG->getTargetConstant(offset, PtrTy);
          Index = CurDAG->getTargetFrameIndex(FI, PtrTy);
          return true;
        }
      } else if (offset > minOffset && offset < maxOffset) {
        Base = CurDAG->getTargetConstant(offset, PtrTy);
        Index = Op0;
        return true;
      }
    } else if (Op0.getOpcode() == ISD::Constant
               || Op0.getOpcode() == ISD::TargetConstant) {
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op0);
      int32_t offset = int32_t(CN->getSignExtended());

      if (Op1.getOpcode() == ISD::FrameIndex) {
        FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Op1);
        int FI = int(FIN->getIndex());
        DEBUG(cerr << "SelectDFormAddr: ISD::ADD offset = " << offset
                   << " frame index = " << FI << "\n");

        if (SPUFrameInfo::FItoStackOffset(FI) < maxOffset) {
          Base = CurDAG->getTargetConstant(offset, PtrTy);
          Index = CurDAG->getTargetFrameIndex(FI, PtrTy);
          return true;
        }
      } else if (offset > minOffset && offset < maxOffset) {
        Base = CurDAG->getTargetConstant(offset, PtrTy);
        Index = Op1;
        return true;
      }
    }
  } else if (Opc == SPUISD::IndirectAddr) {
    // Indirect with constant offset -> D-Form address
    const SDValue Op0 = N.getOperand(0);
    const SDValue Op1 = N.getOperand(1);

    if (Op0.getOpcode() == SPUISD::Hi
        && Op1.getOpcode() == SPUISD::Lo) {
      // (SPUindirect (SPUhi <arg>, 0), (SPUlo <arg>, 0))
      Base = CurDAG->getTargetConstant(0, PtrTy);
      Index = N;
      return true;
    } else if (isa<ConstantSDNode>(Op0) || isa<ConstantSDNode>(Op1)) {
      int32_t offset = 0;
      SDValue idxOp;

      if (isa<ConstantSDNode>(Op1)) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op1);
        offset = int32_t(CN->getSignExtended());
        idxOp = Op0;
      } else if (isa<ConstantSDNode>(Op0)) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op0);
        offset = int32_t(CN->getSignExtended());
        idxOp = Op1;
      } 

      if (offset >= minOffset && offset <= maxOffset) {
        Base = CurDAG->getTargetConstant(offset, PtrTy);
        Index = idxOp;
        return true;
      }
    }
  } else if (Opc == SPUISD::AFormAddr) {
    Base = CurDAG->getTargetConstant(0, N.getValueType());
    Index = N;
    return true;
  } else if (Opc == SPUISD::LDRESULT) {
    Base = CurDAG->getTargetConstant(0, N.getValueType());
    Index = N;
    return true;
  }
  return false;
}

/*!
  \arg Op The ISD instruction operand
  \arg N The address operand
  \arg Base The base pointer operand
  \arg Index The offset/index operand

  If the address \a N can be expressed as a [r + s10imm] address, returns false.
  Otherwise, creates two operands, Base and Index that will become the [r+r]
  address.
*/
bool
SPUDAGToDAGISel::SelectXFormAddr(SDValue Op, SDValue N, SDValue &Base,
                                 SDValue &Index) {
  if (SelectAFormAddr(Op, N, Base, Index)
      || SelectDFormAddr(Op, N, Base, Index))
    return false;

  // All else fails, punt and use an X-form address:
  Base = N.getOperand(0);
  Index = N.getOperand(1);
  return true;
}

//! Convert the operand from a target-independent to a target-specific node
/*!
 */
SDNode *
SPUDAGToDAGISel::Select(SDValue Op) {
  SDNode *N = Op.getNode();
  unsigned Opc = N->getOpcode();
  int n_ops = -1;
  unsigned NewOpc;
  MVT OpVT = Op.getValueType();
  SDValue Ops[8];

  if (N->isMachineOpcode()) {
    return NULL;   // Already selected.
  } else if (Opc == ISD::FrameIndex) {
    // Selects to (add $sp, FI * stackSlotSize)
    int FI =
      SPUFrameInfo::FItoStackOffset(cast<FrameIndexSDNode>(N)->getIndex());
    MVT PtrVT = SPUtli.getPointerTy();

    // Adjust stack slot to actual offset in frame:
    if (isS10Constant(FI)) {
      DEBUG(cerr << "SPUDAGToDAGISel: Replacing FrameIndex with AIr32 $sp, "
                 << FI
                 << "\n");
      NewOpc = SPU::AIr32;
      Ops[0] = CurDAG->getRegister(SPU::R1, PtrVT);
      Ops[1] = CurDAG->getTargetConstant(FI, PtrVT);
      n_ops = 2;
    } else {
      DEBUG(cerr << "SPUDAGToDAGISel: Replacing FrameIndex with Ar32 $sp, "
                 << FI
                 << "\n");
      NewOpc = SPU::Ar32;
      Ops[0] = CurDAG->getRegister(SPU::R1, PtrVT);
      Ops[1] = CurDAG->getConstant(FI, PtrVT);
      n_ops = 2;

      AddToISelQueue(Ops[1]);
    }
  } else if (Opc == ISD::ZERO_EXTEND) {
    // (zero_extend:i16 (and:i8 <arg>, <const>))
    const SDValue &Op1 = N->getOperand(0);

    if (Op.getValueType() == MVT::i16 && Op1.getValueType() == MVT::i8) {
      if (Op1.getOpcode() == ISD::AND) {
        // Fold this into a single ANDHI. This is often seen in expansions of i1
        // to i8, then i8 to i16 in logical/branching operations.
        DEBUG(cerr << "CellSPU: Coalescing (zero_extend:i16 (and:i8 "
                      "<arg>, <const>))\n");
        NewOpc = SPU::ANDHIi8i16;
        Ops[0] = Op1.getOperand(0);
        Ops[1] = Op1.getOperand(1);
        n_ops = 2;
      }
    }
  } else if (Opc == SPUISD::LDRESULT) {
    // Custom select instructions for LDRESULT
    MVT VT = N->getValueType(0);
    SDValue Arg = N->getOperand(0);
    SDValue Chain = N->getOperand(1);
    SDNode *Result;
    const valtype_map_s *vtm = getValueTypeMapEntry(VT);

    if (vtm->ldresult_ins == 0) {
      cerr << "LDRESULT for unsupported type: "
           << VT.getMVTString()
           << "\n";
      abort();
    }

    AddToISelQueue(Arg);
    Opc = vtm->ldresult_ins;
    if (vtm->ldresult_imm) {
      SDValue Zero = CurDAG->getTargetConstant(0, VT);

      AddToISelQueue(Zero);
      Result = CurDAG->getTargetNode(Opc, VT, MVT::Other, Arg, Zero, Chain);
    } else {
      Result = CurDAG->getTargetNode(Opc, MVT::Other, Arg, Arg, Chain);
    }

    Chain = SDValue(Result, 1);
    AddToISelQueue(Chain);

    return Result;
  } else if (Opc == SPUISD::IndirectAddr) {
    SDValue Op0 = Op.getOperand(0);
    if (Op0.getOpcode() == SPUISD::LDRESULT) {
        /* || Op0.getOpcode() == SPUISD::AFormAddr) */
      // (IndirectAddr (LDRESULT, imm))
      SDValue Op1 = Op.getOperand(1);
      MVT VT = Op.getValueType();

      DEBUG(cerr << "CellSPU: IndirectAddr(LDRESULT, imm):\nOp0 = ");
      DEBUG(Op.getOperand(0).getNode()->dump(CurDAG));
      DEBUG(cerr << "\nOp1 = ");
      DEBUG(Op.getOperand(1).getNode()->dump(CurDAG));
      DEBUG(cerr << "\n");

      if (Op1.getOpcode() == ISD::Constant) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op1);
        Op1 = CurDAG->getTargetConstant(CN->getValue(), VT);
        NewOpc = (isI32IntS10Immediate(CN) ? SPU::AIr32 : SPU::Ar32);
        AddToISelQueue(Op0);
        AddToISelQueue(Op1);
        Ops[0] = Op0;
        Ops[1] = Op1;
        n_ops = 2;
      }
    }
  }
  
  if (n_ops > 0) {
    if (N->hasOneUse())
      return CurDAG->SelectNodeTo(N, NewOpc, OpVT, Ops, n_ops);
    else
      return CurDAG->getTargetNode(NewOpc, OpVT, Ops, n_ops);
  } else
    return SelectCode(Op);
}

/// createPPCISelDag - This pass converts a legalized DAG into a 
/// SPU-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSPUISelDag(SPUTargetMachine &TM) {
  return new SPUDAGToDAGISel(TM);
}
