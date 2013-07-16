//===-- SystemZISelDAGToDAG.cpp - A dag to dag inst selector for SystemZ --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the SystemZ target.
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetMachine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
// Used to build addressing modes.
struct SystemZAddressingMode {
  // The shape of the address.
  enum AddrForm {
    // base+displacement
    FormBD,

    // base+displacement+index for load and store operands
    FormBDXNormal,

    // base+displacement+index for load address operands
    FormBDXLA,

    // base+displacement+index+ADJDYNALLOC
    FormBDXDynAlloc
  };
  AddrForm Form;

  // The type of displacement.  The enum names here correspond directly
  // to the definitions in SystemZOperand.td.  We could split them into
  // flags -- single/pair, 128-bit, etc. -- but it hardly seems worth it.
  enum DispRange {
    Disp12Only,
    Disp12Pair,
    Disp20Only,
    Disp20Only128,
    Disp20Pair
  };
  DispRange DR;

  // The parts of the address.  The address is equivalent to:
  //
  //     Base + Disp + Index + (IncludesDynAlloc ? ADJDYNALLOC : 0)
  SDValue Base;
  int64_t Disp;
  SDValue Index;
  bool IncludesDynAlloc;

  SystemZAddressingMode(AddrForm form, DispRange dr)
    : Form(form), DR(dr), Base(), Disp(0), Index(),
      IncludesDynAlloc(false) {}

  // True if the address can have an index register.
  bool hasIndexField() { return Form != FormBD; }

  // True if the address can (and must) include ADJDYNALLOC.
  bool isDynAlloc() { return Form == FormBDXDynAlloc; }

  void dump() {
    errs() << "SystemZAddressingMode " << this << '\n';

    errs() << " Base ";
    if (Base.getNode() != 0)
      Base.getNode()->dump();
    else
      errs() << "null\n";

    if (hasIndexField()) {
      errs() << " Index ";
      if (Index.getNode() != 0)
        Index.getNode()->dump();
      else
        errs() << "null\n";
    }

    errs() << " Disp " << Disp;
    if (IncludesDynAlloc)
      errs() << " + ADJDYNALLOC";
    errs() << '\n';
  }
};

// Return a mask with Count low bits set.
static uint64_t allOnes(unsigned int Count) {
  return Count == 0 ? 0 : (uint64_t(1) << (Count - 1) << 1) - 1;
}

// Represents operands 2 to 5 of a ROTATE AND ... SELECTED BITS operation.
// The operands are: Input (R2), Start (I3), End (I4) and Rotate (I5).
// The operand value is effectively (and (rotl Input Rotate) Mask) and
// has BitSize bits.
struct RISBGOperands {
  RISBGOperands(SDValue N)
    : BitSize(N.getValueType().getSizeInBits()), Mask(allOnes(BitSize)),
      Input(N), Start(64 - BitSize), End(63), Rotate(0) {}

  unsigned BitSize;
  uint64_t Mask;
  SDValue Input;
  unsigned Start;
  unsigned End;
  unsigned Rotate;
};

class SystemZDAGToDAGISel : public SelectionDAGISel {
  const SystemZTargetLowering &Lowering;
  const SystemZSubtarget &Subtarget;

  // Used by SystemZOperands.td to create integer constants.
  inline SDValue getImm(const SDNode *Node, uint64_t Imm) {
    return CurDAG->getTargetConstant(Imm, Node->getValueType(0));
  }

  // Try to fold more of the base or index of AM into AM, where IsBase
  // selects between the base and index.
  bool expandAddress(SystemZAddressingMode &AM, bool IsBase);

  // Try to describe N in AM, returning true on success.
  bool selectAddress(SDValue N, SystemZAddressingMode &AM);

  // Extract individual target operands from matched address AM.
  void getAddressOperands(const SystemZAddressingMode &AM, EVT VT,
                          SDValue &Base, SDValue &Disp);
  void getAddressOperands(const SystemZAddressingMode &AM, EVT VT,
                          SDValue &Base, SDValue &Disp, SDValue &Index);

  // Try to match Addr as a FormBD address with displacement type DR.
  // Return true on success, storing the base and displacement in
  // Base and Disp respectively.
  bool selectBDAddr(SystemZAddressingMode::DispRange DR, SDValue Addr,
                    SDValue &Base, SDValue &Disp);

  // Try to match Addr as a FormBDX* address of form Form with
  // displacement type DR.  Return true on success, storing the base,
  // displacement and index in Base, Disp and Index respectively.
  bool selectBDXAddr(SystemZAddressingMode::AddrForm Form,
                     SystemZAddressingMode::DispRange DR, SDValue Addr,
                     SDValue &Base, SDValue &Disp, SDValue &Index);

  // PC-relative address matching routines used by SystemZOperands.td.
  bool selectPCRelAddress(SDValue Addr, SDValue &Target) {
    if (Addr.getOpcode() == SystemZISD::PCREL_WRAPPER) {
      Target = Addr.getOperand(0);
      return true;
    }
    return false;
  }

  // BD matching routines used by SystemZOperands.td.
  bool selectBDAddr12Only(SDValue Addr, SDValue &Base, SDValue &Disp) {
    return selectBDAddr(SystemZAddressingMode::Disp12Only, Addr, Base, Disp);
  }
  bool selectBDAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp) {
    return selectBDAddr(SystemZAddressingMode::Disp12Pair, Addr, Base, Disp);
  }
  bool selectBDAddr20Only(SDValue Addr, SDValue &Base, SDValue &Disp) {
    return selectBDAddr(SystemZAddressingMode::Disp20Only, Addr, Base, Disp);
  }
  bool selectBDAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp) {
    return selectBDAddr(SystemZAddressingMode::Disp20Pair, Addr, Base, Disp);
  }

  // BDX matching routines used by SystemZOperands.td.
  bool selectBDXAddr12Only(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp12Only,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp12Pair,
                         Addr, Base, Disp, Index);
  }
  bool selectDynAlloc12Only(SDValue Addr, SDValue &Base, SDValue &Disp,
                            SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXDynAlloc,
                         SystemZAddressingMode::Disp12Only,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr20Only(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp20Only,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr20Only128(SDValue Addr, SDValue &Base, SDValue &Disp,
                              SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp20Only128,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp20Pair,
                         Addr, Base, Disp, Index);
  }
  bool selectLAAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                          SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXLA,
                         SystemZAddressingMode::Disp12Pair,
                         Addr, Base, Disp, Index);
  }
  bool selectLAAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                          SDValue &Index) {
    return selectBDXAddr(SystemZAddressingMode::FormBDXLA,
                         SystemZAddressingMode::Disp20Pair,
                         Addr, Base, Disp, Index);
  }

  // Try to fold some of Ops.Input into other fields of Ops.  Return true
  // on success.
  bool expandRISBG(RISBGOperands &Ops);

  // Return an undefined i64 value.
  SDValue getUNDEF64(SDLoc DL);

  // Convert N to VT, if it isn't already.
  SDValue convertTo(SDLoc DL, EVT VT, SDValue N);

  // Try to implement AND or shift node N using RISBG with the zero flag set.
  // Return the selected node on success, otherwise return null.
  SDNode *tryRISBGZero(SDNode *N);

  // If Op0 is null, then Node is a constant that can be loaded using:
  //
  //   (Opcode UpperVal LowerVal)
  //
  // If Op0 is nonnull, then Node can be implemented using:
  //
  //   (Opcode (Opcode Op0 UpperVal) LowerVal)
  SDNode *splitLargeImmediate(unsigned Opcode, SDNode *Node, SDValue Op0,
                              uint64_t UpperVal, uint64_t LowerVal);

  bool storeLoadCanUseMVC(SDNode *N) const;

public:
  SystemZDAGToDAGISel(SystemZTargetMachine &TM, CodeGenOpt::Level OptLevel)
    : SelectionDAGISel(TM, OptLevel),
      Lowering(*TM.getTargetLowering()),
      Subtarget(*TM.getSubtargetImpl()) { }

  // Override MachineFunctionPass.
  virtual const char *getPassName() const LLVM_OVERRIDE {
    return "SystemZ DAG->DAG Pattern Instruction Selection";
  }

  // Override SelectionDAGISel.
  virtual SDNode *Select(SDNode *Node) LLVM_OVERRIDE;
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps)
    LLVM_OVERRIDE;

  // Include the pieces autogenerated from the target description.
  #include "SystemZGenDAGISel.inc"
};
} // end anonymous namespace

FunctionPass *llvm::createSystemZISelDag(SystemZTargetMachine &TM,
                                         CodeGenOpt::Level OptLevel) {
  return new SystemZDAGToDAGISel(TM, OptLevel);
}

// Return true if Val should be selected as a displacement for an address
// with range DR.  Here we're interested in the range of both the instruction
// described by DR and of any pairing instruction.
static bool selectDisp(SystemZAddressingMode::DispRange DR, int64_t Val) {
  switch (DR) {
  case SystemZAddressingMode::Disp12Only:
    return isUInt<12>(Val);

  case SystemZAddressingMode::Disp12Pair:
  case SystemZAddressingMode::Disp20Only:
  case SystemZAddressingMode::Disp20Pair:
    return isInt<20>(Val);

  case SystemZAddressingMode::Disp20Only128:
    return isInt<20>(Val) && isInt<20>(Val + 8);
  }
  llvm_unreachable("Unhandled displacement range");
}

// Change the base or index in AM to Value, where IsBase selects
// between the base and index.
static void changeComponent(SystemZAddressingMode &AM, bool IsBase,
                            SDValue Value) {
  if (IsBase)
    AM.Base = Value;
  else
    AM.Index = Value;
}

// The base or index of AM is equivalent to Value + ADJDYNALLOC,
// where IsBase selects between the base and index.  Try to fold the
// ADJDYNALLOC into AM.
static bool expandAdjDynAlloc(SystemZAddressingMode &AM, bool IsBase,
                              SDValue Value) {
  if (AM.isDynAlloc() && !AM.IncludesDynAlloc) {
    changeComponent(AM, IsBase, Value);
    AM.IncludesDynAlloc = true;
    return true;
  }
  return false;
}

// The base of AM is equivalent to Base + Index.  Try to use Index as
// the index register.
static bool expandIndex(SystemZAddressingMode &AM, SDValue Base,
                        SDValue Index) {
  if (AM.hasIndexField() && !AM.Index.getNode()) {
    AM.Base = Base;
    AM.Index = Index;
    return true;
  }
  return false;
}

// The base or index of AM is equivalent to Op0 + Op1, where IsBase selects
// between the base and index.  Try to fold Op1 into AM's displacement.
static bool expandDisp(SystemZAddressingMode &AM, bool IsBase,
                       SDValue Op0, ConstantSDNode *Op1) {
  // First try adjusting the displacement.
  int64_t TestDisp = AM.Disp + Op1->getSExtValue();
  if (selectDisp(AM.DR, TestDisp)) {
    changeComponent(AM, IsBase, Op0);
    AM.Disp = TestDisp;
    return true;
  }

  // We could consider forcing the displacement into a register and
  // using it as an index, but it would need to be carefully tuned.
  return false;
}

bool SystemZDAGToDAGISel::expandAddress(SystemZAddressingMode &AM,
                                        bool IsBase) {
  SDValue N = IsBase ? AM.Base : AM.Index;
  unsigned Opcode = N.getOpcode();
  if (Opcode == ISD::TRUNCATE) {
    N = N.getOperand(0);
    Opcode = N.getOpcode();
  }
  if (Opcode == ISD::ADD || CurDAG->isBaseWithConstantOffset(N)) {
    SDValue Op0 = N.getOperand(0);
    SDValue Op1 = N.getOperand(1);

    unsigned Op0Code = Op0->getOpcode();
    unsigned Op1Code = Op1->getOpcode();

    if (Op0Code == SystemZISD::ADJDYNALLOC)
      return expandAdjDynAlloc(AM, IsBase, Op1);
    if (Op1Code == SystemZISD::ADJDYNALLOC)
      return expandAdjDynAlloc(AM, IsBase, Op0);

    if (Op0Code == ISD::Constant)
      return expandDisp(AM, IsBase, Op1, cast<ConstantSDNode>(Op0));
    if (Op1Code == ISD::Constant)
      return expandDisp(AM, IsBase, Op0, cast<ConstantSDNode>(Op1));

    if (IsBase && expandIndex(AM, Op0, Op1))
      return true;
  }
  return false;
}

// Return true if an instruction with displacement range DR should be
// used for displacement value Val.  selectDisp(DR, Val) must already hold.
static bool isValidDisp(SystemZAddressingMode::DispRange DR, int64_t Val) {
  assert(selectDisp(DR, Val) && "Invalid displacement");
  switch (DR) {
  case SystemZAddressingMode::Disp12Only:
  case SystemZAddressingMode::Disp20Only:
  case SystemZAddressingMode::Disp20Only128:
    return true;

  case SystemZAddressingMode::Disp12Pair:
    // Use the other instruction if the displacement is too large.
    return isUInt<12>(Val);

  case SystemZAddressingMode::Disp20Pair:
    // Use the other instruction if the displacement is small enough.
    return !isUInt<12>(Val);
  }
  llvm_unreachable("Unhandled displacement range");
}

// Return true if Base + Disp + Index should be performed by LA(Y).
static bool shouldUseLA(SDNode *Base, int64_t Disp, SDNode *Index) {
  // Don't use LA(Y) for constants.
  if (!Base)
    return false;

  // Always use LA(Y) for frame addresses, since we know that the destination
  // register is almost always (perhaps always) going to be different from
  // the frame register.
  if (Base->getOpcode() == ISD::FrameIndex)
    return true;

  if (Disp) {
    // Always use LA(Y) if there is a base, displacement and index.
    if (Index)
      return true;

    // Always use LA if the displacement is small enough.  It should always
    // be no worse than AGHI (and better if it avoids a move).
    if (isUInt<12>(Disp))
      return true;

    // For similar reasons, always use LAY if the constant is too big for AGHI.
    // LAY should be no worse than AGFI.
    if (!isInt<16>(Disp))
      return true;
  } else {
    // Don't use LA for plain registers.
    if (!Index)
      return false;

    // Don't use LA for plain addition if the index operand is only used
    // once.  It should be a natural two-operand addition in that case.
    if (Index->hasOneUse())
      return false;

    // Prefer addition if the second operation is sign-extended, in the
    // hope of using AGF.
    unsigned IndexOpcode = Index->getOpcode();
    if (IndexOpcode == ISD::SIGN_EXTEND ||
        IndexOpcode == ISD::SIGN_EXTEND_INREG)
      return false;
  }

  // Don't use LA for two-operand addition if either operand is only
  // used once.  The addition instructions are better in that case.
  if (Base->hasOneUse())
    return false;

  return true;
}

// Return true if Addr is suitable for AM, updating AM if so.
bool SystemZDAGToDAGISel::selectAddress(SDValue Addr,
                                        SystemZAddressingMode &AM) {
  // Start out assuming that the address will need to be loaded separately,
  // then try to extend it as much as we can.
  AM.Base = Addr;

  // First try treating the address as a constant.
  if (Addr.getOpcode() == ISD::Constant &&
      expandDisp(AM, true, SDValue(), cast<ConstantSDNode>(Addr)))
    ;
  else
    // Otherwise try expanding each component.
    while (expandAddress(AM, true) ||
           (AM.Index.getNode() && expandAddress(AM, false)))
      continue;

  // Reject cases where it isn't profitable to use LA(Y).
  if (AM.Form == SystemZAddressingMode::FormBDXLA &&
      !shouldUseLA(AM.Base.getNode(), AM.Disp, AM.Index.getNode()))
    return false;

  // Reject cases where the other instruction in a pair should be used.
  if (!isValidDisp(AM.DR, AM.Disp))
    return false;

  // Make sure that ADJDYNALLOC is included where necessary.
  if (AM.isDynAlloc() && !AM.IncludesDynAlloc)
    return false;

  DEBUG(AM.dump());
  return true;
}

// Insert a node into the DAG at least before Pos.  This will reposition
// the node as needed, and will assign it a node ID that is <= Pos's ID.
// Note that this does *not* preserve the uniqueness of node IDs!
// The selection DAG must no longer depend on their uniqueness when this
// function is used.
static void insertDAGNode(SelectionDAG *DAG, SDNode *Pos, SDValue N) {
  if (N.getNode()->getNodeId() == -1 ||
      N.getNode()->getNodeId() > Pos->getNodeId()) {
    DAG->RepositionNode(Pos, N.getNode());
    N.getNode()->setNodeId(Pos->getNodeId());
  }
}

void SystemZDAGToDAGISel::getAddressOperands(const SystemZAddressingMode &AM,
                                             EVT VT, SDValue &Base,
                                             SDValue &Disp) {
  Base = AM.Base;
  if (!Base.getNode())
    // Register 0 means "no base".  This is mostly useful for shifts.
    Base = CurDAG->getRegister(0, VT);
  else if (Base.getOpcode() == ISD::FrameIndex) {
    // Lower a FrameIndex to a TargetFrameIndex.
    int64_t FrameIndex = cast<FrameIndexSDNode>(Base)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FrameIndex, VT);
  } else if (Base.getValueType() != VT) {
    // Truncate values from i64 to i32, for shifts.
    assert(VT == MVT::i32 && Base.getValueType() == MVT::i64 &&
           "Unexpected truncation");
    SDLoc DL(Base);
    SDValue Trunc = CurDAG->getNode(ISD::TRUNCATE, DL, VT, Base);
    insertDAGNode(CurDAG, Base.getNode(), Trunc);
    Base = Trunc;
  }

  // Lower the displacement to a TargetConstant.
  Disp = CurDAG->getTargetConstant(AM.Disp, VT);
}

void SystemZDAGToDAGISel::getAddressOperands(const SystemZAddressingMode &AM,
                                             EVT VT, SDValue &Base,
                                             SDValue &Disp, SDValue &Index) {
  getAddressOperands(AM, VT, Base, Disp);

  Index = AM.Index;
  if (!Index.getNode())
    // Register 0 means "no index".
    Index = CurDAG->getRegister(0, VT);
}

bool SystemZDAGToDAGISel::selectBDAddr(SystemZAddressingMode::DispRange DR,
                                       SDValue Addr, SDValue &Base,
                                       SDValue &Disp) {
  SystemZAddressingMode AM(SystemZAddressingMode::FormBD, DR);
  if (!selectAddress(Addr, AM))
    return false;

  getAddressOperands(AM, Addr.getValueType(), Base, Disp);
  return true;
}

bool SystemZDAGToDAGISel::selectBDXAddr(SystemZAddressingMode::AddrForm Form,
                                        SystemZAddressingMode::DispRange DR,
                                        SDValue Addr, SDValue &Base,
                                        SDValue &Disp, SDValue &Index) {
  SystemZAddressingMode AM(Form, DR);
  if (!selectAddress(Addr, AM))
    return false;

  getAddressOperands(AM, Addr.getValueType(), Base, Disp, Index);
  return true;
}

// Return true if Mask matches the regexp 0*1+0*, given that zero masks
// have already been filtered out.  Store the first set bit in LSB and
// the number of set bits in Length if so.
static bool isStringOfOnes(uint64_t Mask, unsigned &LSB, unsigned &Length) {
  unsigned First = findFirstSet(Mask);
  uint64_t Top = (Mask >> First) + 1;
  if ((Top & -Top) == Top)
    {
      LSB = First;
      Length = findFirstSet(Top);
      return true;
    }
  return false;
}

// Try to update RISBG so that only the bits of Ops.Input in Mask are used.
// Return true on success.
static bool refineRISBGMask(RISBGOperands &RISBG, uint64_t Mask) {
  if (RISBG.Rotate != 0)
    Mask = (Mask << RISBG.Rotate) | (Mask >> (64 - RISBG.Rotate));
  Mask &= RISBG.Mask;

  // Reject trivial all-zero masks.
  if (Mask == 0)
    return false;

  // Handle the 1+0+ or 0+1+0* cases.  Start then specifies the index of
  // the msb and End specifies the index of the lsb.
  unsigned LSB, Length;
  if (isStringOfOnes(Mask, LSB, Length))
    {
      RISBG.Mask = Mask;
      RISBG.Start = 63 - (LSB + Length - 1);
      RISBG.End = 63 - LSB;
      return true;
    }

  // Handle the wrap-around 1+0+1+ cases.  Start then specifies the msb
  // of the low 1s and End specifies the lsb of the high 1s.
  if (isStringOfOnes(Mask ^ allOnes(RISBG.BitSize), LSB, Length))
    {
      assert(LSB > 0 && "Bottom bit must be set");
      assert(LSB + Length < RISBG.BitSize && "Top bit must be set");
      RISBG.Mask = Mask;
      RISBG.Start = 63 - (LSB - 1);
      RISBG.End = 63 - (LSB + Length);
      return true;
    }

  return false;
}

bool SystemZDAGToDAGISel::expandRISBG(RISBGOperands &RISBG) {
  SDValue N = RISBG.Input;
  switch (N.getOpcode()) {
  case ISD::AND: {
    ConstantSDNode *MaskNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!MaskNode)
      return false;

    SDValue Input = N.getOperand(0);
    uint64_t Mask = MaskNode->getZExtValue();
    if (!refineRISBGMask(RISBG, Mask)) {
      // If some bits of Input are already known zeros, those bits will have
      // been removed from the mask.  See if adding them back in makes the
      // mask suitable.
      APInt KnownZero, KnownOne;
      CurDAG->ComputeMaskedBits(Input, KnownZero, KnownOne);
      Mask |= KnownZero.getZExtValue();
      if (!refineRISBGMask(RISBG, Mask))
        return false;
    }
    RISBG.Input = Input;
    return true;
  }

  case ISD::ROTL: {
    // Any 64-bit rotate left can be merged into the RISBG.
    if (RISBG.BitSize != 64)
      return false;
    ConstantSDNode *CountNode
      = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    RISBG.Rotate = (RISBG.Rotate + CountNode->getZExtValue()) & 63;
    RISBG.Input = N.getOperand(0);
    return true;
  }
      
  case ISD::SHL: {
    // Treat (shl X, count) as (and (rotl X, count), ~0<<count).
    ConstantSDNode *CountNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    if (Count < 1 ||
        Count >= RISBG.BitSize ||
        !refineRISBGMask(RISBG, allOnes(RISBG.BitSize - Count) << Count))
      return false;

    RISBG.Rotate = (RISBG.Rotate + Count) & 63;
    RISBG.Input = N.getOperand(0);
    return true;
  }

  case ISD::SRL: {
    // Treat (srl X, count), mask) as (and (rotl X, size-count), ~0>>count),
    // which is similar to SLL above.
    ConstantSDNode *CountNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    if (Count < 1 ||
        Count >= RISBG.BitSize ||
        !refineRISBGMask(RISBG, allOnes(RISBG.BitSize - Count)))
      return false;

    RISBG.Rotate = (RISBG.Rotate - Count) & 63;
    RISBG.Input = N.getOperand(0);
    return true;
  }

  case ISD::SRA: {
    // Treat (sra X, count) as (rotl X, size-count) as long as the top
    // count bits from Ops.Input are ignored.
    ConstantSDNode *CountNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    if (RISBG.Rotate != 0 ||
        Count < 1 ||
        Count >= RISBG.BitSize ||
        RISBG.Start < 64 - (RISBG.BitSize - Count))
      return false;

    RISBG.Rotate = -Count & 63;
    RISBG.Input = N.getOperand(0);
    return true;
  }
  default:
    return false;
  }
}

SDValue SystemZDAGToDAGISel::getUNDEF64(SDLoc DL) {
  SDNode *N = CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, MVT::i64);
  return SDValue(N, 0);
}

SDValue SystemZDAGToDAGISel::convertTo(SDLoc DL, EVT VT, SDValue N) {
  if (N.getValueType() == MVT::i32 && VT == MVT::i64) {
    SDValue Index = CurDAG->getTargetConstant(SystemZ::subreg_32bit, MVT::i64);
    SDNode *Insert = CurDAG->getMachineNode(TargetOpcode::INSERT_SUBREG,
                                            DL, VT, getUNDEF64(DL), N, Index);
    return SDValue(Insert, 0);
  }
  if (N.getValueType() == MVT::i64 && VT == MVT::i32) {
    SDValue Index = CurDAG->getTargetConstant(SystemZ::subreg_32bit, MVT::i64);
    SDNode *Extract = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                             DL, VT, N, Index);
    return SDValue(Extract, 0);
  }
  assert(N.getValueType() == VT && "Unexpected value types");
  return N;
}

SDNode *SystemZDAGToDAGISel::tryRISBGZero(SDNode *N) {
  RISBGOperands RISBG(SDValue(N, 0));
  unsigned Count = 0;
  while (expandRISBG(RISBG))
    Count += 1;
  // Prefer to use normal shift instructions over RISBG, since they can handle
  // all cases and are sometimes shorter.  Prefer to use RISBG for ANDs though,
  // since it is effectively a three-operand instruction in this case,
  // and since it can handle some masks that AND IMMEDIATE can't.
  if (Count < (N->getOpcode() == ISD::AND ? 1 : 2))
    return 0;

  // Prefer register extensions like LLC over RISBG.
  if (RISBG.Rotate == 0 &&
      (RISBG.Start == 32 || RISBG.Start == 48 || RISBG.Start == 56) &&
      RISBG.End == 63)
    return 0;

  EVT VT = N->getValueType(0);
  SDValue Ops[5] = {
    getUNDEF64(SDLoc(N)),
    convertTo(SDLoc(N), MVT::i64, RISBG.Input),
    CurDAG->getTargetConstant(RISBG.Start, MVT::i32),
    CurDAG->getTargetConstant(RISBG.End | 128, MVT::i32),
    CurDAG->getTargetConstant(RISBG.Rotate, MVT::i32)
  };
  N = CurDAG->getMachineNode(SystemZ::RISBG, SDLoc(N), MVT::i64, Ops);
  return convertTo(SDLoc(N), VT, SDValue(N, 0)).getNode();
}

SDNode *SystemZDAGToDAGISel::splitLargeImmediate(unsigned Opcode, SDNode *Node,
                                                 SDValue Op0, uint64_t UpperVal,
                                                 uint64_t LowerVal) {
  EVT VT = Node->getValueType(0);
  SDLoc DL(Node);
  SDValue Upper = CurDAG->getConstant(UpperVal, VT);
  if (Op0.getNode())
    Upper = CurDAG->getNode(Opcode, DL, VT, Op0, Upper);
  Upper = SDValue(Select(Upper.getNode()), 0);

  SDValue Lower = CurDAG->getConstant(LowerVal, VT);
  SDValue Or = CurDAG->getNode(Opcode, DL, VT, Upper, Lower);
  return Or.getNode();
}

// N is a (store (load ...), ...) pattern.  Return true if it can use MVC.
bool SystemZDAGToDAGISel::storeLoadCanUseMVC(SDNode *N) const {
  StoreSDNode *Store = cast<StoreSDNode>(N);
  LoadSDNode *Load = cast<LoadSDNode>(Store->getValue().getNode());

  // MVC is logically a bytewise copy, so can't be used for volatile accesses.
  if (Load->isVolatile() || Store->isVolatile())
    return false;

  // Prefer not to use MVC if either address can use ... RELATIVE LONG
  // instructions.
  assert(Load->getMemoryVT() == Store->getMemoryVT() &&
         "Should already have checked that the types match");
  uint64_t Size = Load->getMemoryVT().getStoreSize();
  if (Size > 1 && Size <= 8) {
    // Prefer LHRL, LRL and LGRL.
    if (Load->getBasePtr().getOpcode() == SystemZISD::PCREL_WRAPPER)
      return false;
    // Prefer STHRL, STRL and STGRL.
    if (Store->getBasePtr().getOpcode() == SystemZISD::PCREL_WRAPPER)
      return false;
  }

  // There's no chance of overlap if the load is invariant.
  if (Load->isInvariant())
    return true;

  // If both operands are aligned, they must be equal or not overlap.
  if (Load->getAlignment() >= Size && Store->getAlignment() >= Size)
    return true;

  // Otherwise we need to check whether there's an alias.
  const Value *V1 = Load->getSrcValue();
  const Value *V2 = Store->getSrcValue();
  if (!V1 || !V2)
    return false;

  int64_t End1 = Load->getSrcValueOffset() + Size;
  int64_t End2 = Store->getSrcValueOffset() + Size;
  return !AA->alias(AliasAnalysis::Location(V1, End1, Load->getTBAAInfo()),
                    AliasAnalysis::Location(V2, End2, Store->getTBAAInfo()));
}

SDNode *SystemZDAGToDAGISel::Select(SDNode *Node) {
  // Dump information about the Node being selected
  DEBUG(errs() << "Selecting: "; Node->dump(CurDAG); errs() << "\n");

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    return 0;
  }

  unsigned Opcode = Node->getOpcode();
  SDNode *ResNode = 0;
  switch (Opcode) {
  case ISD::OR:
  case ISD::XOR:
    // If this is a 64-bit operation in which both 32-bit halves are nonzero,
    // split the operation into two.
    if (Node->getValueType(0) == MVT::i64)
      if (ConstantSDNode *Op1 = dyn_cast<ConstantSDNode>(Node->getOperand(1))) {
        uint64_t Val = Op1->getZExtValue();
        if (!SystemZ::isImmLF(Val) && !SystemZ::isImmHF(Val))
          Node = splitLargeImmediate(Opcode, Node, Node->getOperand(0),
                                     Val - uint32_t(Val), uint32_t(Val));
      }
    break;

  case ISD::AND:
  case ISD::ROTL:
  case ISD::SHL:
  case ISD::SRL:
    ResNode = tryRISBGZero(Node);
    break;

  case ISD::Constant:
    // If this is a 64-bit constant that is out of the range of LLILF,
    // LLIHF and LGFI, split it into two 32-bit pieces.
    if (Node->getValueType(0) == MVT::i64) {
      uint64_t Val = cast<ConstantSDNode>(Node)->getZExtValue();
      if (!SystemZ::isImmLF(Val) && !SystemZ::isImmHF(Val) && !isInt<32>(Val))
        Node = splitLargeImmediate(ISD::OR, Node, SDValue(),
                                   Val - uint32_t(Val), uint32_t(Val));
    }
    break;

  case ISD::ATOMIC_LOAD_SUB:
    // Try to convert subtractions of constants to additions.
    if (ConstantSDNode *Op2 = dyn_cast<ConstantSDNode>(Node->getOperand(2))) {
      uint64_t Value = -Op2->getZExtValue();
      EVT VT = Node->getValueType(0);
      if (VT == MVT::i32 || isInt<32>(Value)) {
        SDValue Ops[] = { Node->getOperand(0), Node->getOperand(1),
                          CurDAG->getConstant(int32_t(Value), VT) };
        Node = CurDAG->MorphNodeTo(Node, ISD::ATOMIC_LOAD_ADD,
                                   Node->getVTList(), Ops, array_lengthof(Ops));
      }
    }
    break;
  }

  // Select the default instruction
  if (!ResNode)
    ResNode = SelectCode(Node);

  DEBUG(errs() << "=> ";
        if (ResNode == NULL || ResNode == Node)
          Node->dump(CurDAG);
        else
          ResNode->dump(CurDAG);
        errs() << "\n";
        );
  return ResNode;
}

bool SystemZDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op,
                             char ConstraintCode,
                             std::vector<SDValue> &OutOps) {
  assert(ConstraintCode == 'm' && "Unexpected constraint code");
  // Accept addresses with short displacements, which are compatible
  // with Q, R, S and T.  But keep the index operand for future expansion.
  SDValue Base, Disp, Index;
  if (!selectBDXAddr(SystemZAddressingMode::FormBD,
                     SystemZAddressingMode::Disp12Only,
                     Op, Base, Disp, Index))
    return true;
  OutOps.push_back(Base);
  OutOps.push_back(Disp);
  OutOps.push_back(Index);
  return false;
}
