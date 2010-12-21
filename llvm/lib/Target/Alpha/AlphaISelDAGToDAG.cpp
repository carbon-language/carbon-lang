//===-- AlphaISelDAGToDAG.cpp - Alpha pattern matching inst selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for Alpha,
// converting from a legalized dag to a Alpha dag.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

namespace {

  //===--------------------------------------------------------------------===//
  /// AlphaDAGToDAGISel - Alpha specific code to select Alpha machine
  /// instructions for SelectionDAG operations.
  class AlphaDAGToDAGISel : public SelectionDAGISel {
    static const int64_t IMM_LOW  = -32768;
    static const int64_t IMM_HIGH = 32767;
    static const int64_t IMM_MULT = 65536;
    static const int64_t IMM_FULLHIGH = IMM_HIGH + IMM_HIGH * IMM_MULT;
    static const int64_t IMM_FULLLOW = IMM_LOW + IMM_LOW  * IMM_MULT;

    static int64_t get_ldah16(int64_t x) {
      int64_t y = x / IMM_MULT;
      if (x % IMM_MULT > IMM_HIGH)
        ++y;
      return y;
    }

    static int64_t get_lda16(int64_t x) {
      return x - get_ldah16(x) * IMM_MULT;
    }

    /// get_zapImm - Return a zap mask if X is a valid immediate for a zapnot
    /// instruction (if not, return 0).  Note that this code accepts partial
    /// zap masks.  For example (and LHS, 1) is a valid zap, as long we know
    /// that the bits 1-7 of LHS are already zero.  If LHS is non-null, we are
    /// in checking mode.  If LHS is null, we assume that the mask has already
    /// been validated before.
    uint64_t get_zapImm(SDValue LHS, uint64_t Constant) const {
      uint64_t BitsToCheck = 0;
      unsigned Result = 0;
      for (unsigned i = 0; i != 8; ++i) {
        if (((Constant >> 8*i) & 0xFF) == 0) {
          // nothing to do.
        } else {
          Result |= 1 << i;
          if (((Constant >> 8*i) & 0xFF) == 0xFF) {
            // If the entire byte is set, zapnot the byte.
          } else if (LHS.getNode() == 0) {
            // Otherwise, if the mask was previously validated, we know its okay
            // to zapnot this entire byte even though all the bits aren't set.
          } else {
            // Otherwise we don't know that the it's okay to zapnot this entire
            // byte.  Only do this iff we can prove that the missing bits are
            // already null, so the bytezap doesn't need to really null them.
            BitsToCheck |= ~Constant & (0xFF << 8*i);
          }
        }
      }
      
      // If there are missing bits in a byte (for example, X & 0xEF00), check to
      // see if the missing bits (0x1000) are already known zero if not, the zap
      // isn't okay to do, as it won't clear all the required bits.
      if (BitsToCheck &&
          !CurDAG->MaskedValueIsZero(LHS,
                                     APInt(LHS.getValueSizeInBits(),
                                           BitsToCheck)))
        return 0;
      
      return Result;
    }
    
    static uint64_t get_zapImm(uint64_t x) {
      unsigned build = 0;
      for(int i = 0; i != 8; ++i) {
        if ((x & 0x00FF) == 0x00FF)
          build |= 1 << i;
        else if ((x & 0x00FF) != 0)
          return 0;
        x >>= 8;
      }
      return build;
    }
      
    
    static uint64_t getNearPower2(uint64_t x) {
      if (!x) return 0;
      unsigned at = CountLeadingZeros_64(x);
      uint64_t complow = 1ULL << (63 - at);
      uint64_t comphigh = 1ULL << (64 - at);
      //cerr << x << ":" << complow << ":" << comphigh << "\n";
      if (abs64(complow - x) <= abs64(comphigh - x))
        return complow;
      else
        return comphigh;
    }

    static bool chkRemNearPower2(uint64_t x, uint64_t r, bool swap) {
      uint64_t y = getNearPower2(x);
      if (swap)
        return (y - x) == r;
      else
        return (x - y) == r;
    }

  public:
    explicit AlphaDAGToDAGISel(AlphaTargetMachine &TM)
      : SelectionDAGISel(TM)
    {}

    /// getI64Imm - Return a target constant with the specified value, of type
    /// i64.
    inline SDValue getI64Imm(int64_t Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i64);
    }

    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    SDNode *Select(SDNode *N);
    
    virtual const char *getPassName() const {
      return "Alpha DAG->DAG Pattern Instruction Selection";
    } 

    /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
    /// inline asm expressions.
    virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                              char ConstraintCode,
                                              std::vector<SDValue> &OutOps) {
      SDValue Op0;
      switch (ConstraintCode) {
      default: return true;
      case 'm':   // memory
        Op0 = Op;
        break;
      }
      
      OutOps.push_back(Op0);
      return false;
    }
    
// Include the pieces autogenerated from the target description.
#include "AlphaGenDAGISel.inc"
    
private:
    /// getTargetMachine - Return a reference to the TargetMachine, casted
    /// to the target-specific type.
    const AlphaTargetMachine &getTargetMachine() {
      return static_cast<const AlphaTargetMachine &>(TM);
    }

    /// getInstrInfo - Return a reference to the TargetInstrInfo, casted
    /// to the target-specific type.
    const AlphaInstrInfo *getInstrInfo() {
      return getTargetMachine().getInstrInfo();
    }

    SDNode *getGlobalBaseReg();
    SDNode *getGlobalRetAddr();
    void SelectCALL(SDNode *Op);

  };
}

/// getGlobalBaseReg - Output the instructions required to put the
/// GOT address into a register.
///
SDNode *AlphaDAGToDAGISel::getGlobalBaseReg() {
  unsigned GlobalBaseReg = getInstrInfo()->getGlobalBaseReg(MF);
  return CurDAG->getRegister(GlobalBaseReg, TLI.getPointerTy()).getNode();
}

/// getGlobalRetAddr - Grab the return address.
///
SDNode *AlphaDAGToDAGISel::getGlobalRetAddr() {
  unsigned GlobalRetAddr = getInstrInfo()->getGlobalRetAddr(MF);
  return CurDAG->getRegister(GlobalRetAddr, TLI.getPointerTy()).getNode();
}

// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
SDNode *AlphaDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode())
    return NULL;   // Already selected.
  DebugLoc dl = N->getDebugLoc();

  switch (N->getOpcode()) {
  default: break;
  case AlphaISD::CALL:
    SelectCALL(N);
    return NULL;

  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    return CurDAG->SelectNodeTo(N, Alpha::LDA, MVT::i64,
                                CurDAG->getTargetFrameIndex(FI, MVT::i32),
                                getI64Imm(0));
  }
  case ISD::GLOBAL_OFFSET_TABLE:
    return getGlobalBaseReg();
  case AlphaISD::GlobalRetAddr:
    return getGlobalRetAddr();
  
  case AlphaISD::DivCall: {
    SDValue Chain = CurDAG->getEntryNode();
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    Chain = CurDAG->getCopyToReg(Chain, dl, Alpha::R24, N1, 
                                 SDValue(0,0));
    Chain = CurDAG->getCopyToReg(Chain, dl, Alpha::R25, N2, 
                                 Chain.getValue(1));
    Chain = CurDAG->getCopyToReg(Chain, dl, Alpha::R27, N0, 
                                 Chain.getValue(1));
    SDNode *CNode =
      CurDAG->getMachineNode(Alpha::JSRs, dl, MVT::Other, MVT::Glue, 
                             Chain, Chain.getValue(1));
    Chain = CurDAG->getCopyFromReg(Chain, dl, Alpha::R27, MVT::i64, 
                                   SDValue(CNode, 1));
    return CurDAG->SelectNodeTo(N, Alpha::BISr, MVT::i64, Chain, Chain);
  }

  case ISD::READCYCLECOUNTER: {
    SDValue Chain = N->getOperand(0);
    return CurDAG->getMachineNode(Alpha::RPCC, dl, MVT::i64, MVT::Other,
                                  Chain);
  }

  case ISD::Constant: {
    uint64_t uval = cast<ConstantSDNode>(N)->getZExtValue();
    
    if (uval == 0) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                Alpha::R31, MVT::i64);
      ReplaceUses(SDValue(N, 0), Result);
      return NULL;
    }

    int64_t val = (int64_t)uval;
    int32_t val32 = (int32_t)val;
    if (val <= IMM_HIGH + IMM_HIGH * IMM_MULT &&
        val >= IMM_LOW  + IMM_LOW  * IMM_MULT)
      break; //(LDAH (LDA))
    if ((uval >> 32) == 0 && //empty upper bits
        val32 <= IMM_HIGH + IMM_HIGH * IMM_MULT)
      // val32 >= IMM_LOW  + IMM_LOW  * IMM_MULT) //always true
      break; //(zext (LDAH (LDA)))
    //Else use the constant pool
    ConstantInt *C = ConstantInt::get(
                                Type::getInt64Ty(*CurDAG->getContext()), uval);
    SDValue CPI = CurDAG->getTargetConstantPool(C, MVT::i64);
    SDNode *Tmp = CurDAG->getMachineNode(Alpha::LDAHr, dl, MVT::i64, CPI,
                                         SDValue(getGlobalBaseReg(), 0));
    return CurDAG->SelectNodeTo(N, Alpha::LDQr, MVT::i64, MVT::Other, 
                                CPI, SDValue(Tmp, 0), CurDAG->getEntryNode());
  }
  case ISD::TargetConstantFP:
  case ISD::ConstantFP: {
    ConstantFPSDNode *CN = cast<ConstantFPSDNode>(N);
    bool isDouble = N->getValueType(0) == MVT::f64;
    EVT T = isDouble ? MVT::f64 : MVT::f32;
    if (CN->getValueAPF().isPosZero()) {
      return CurDAG->SelectNodeTo(N, isDouble ? Alpha::CPYST : Alpha::CPYSS,
                                  T, CurDAG->getRegister(Alpha::F31, T),
                                  CurDAG->getRegister(Alpha::F31, T));
    } else if (CN->getValueAPF().isNegZero()) {
      return CurDAG->SelectNodeTo(N, isDouble ? Alpha::CPYSNT : Alpha::CPYSNS,
                                  T, CurDAG->getRegister(Alpha::F31, T),
                                  CurDAG->getRegister(Alpha::F31, T));
    } else {
      report_fatal_error("Unhandled FP constant type");
    }
    break;
  }

  case ISD::SETCC:
    if (N->getOperand(0).getNode()->getValueType(0).isFloatingPoint()) {
      ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();

      unsigned Opc = Alpha::WTF;
      bool rev = false;
      bool inv = false;
      switch(CC) {
      default: DEBUG(N->dump(CurDAG)); llvm_unreachable("Unknown FP comparison!");
      case ISD::SETEQ: case ISD::SETOEQ: case ISD::SETUEQ:
        Opc = Alpha::CMPTEQ; break;
      case ISD::SETLT: case ISD::SETOLT: case ISD::SETULT: 
        Opc = Alpha::CMPTLT; break;
      case ISD::SETLE: case ISD::SETOLE: case ISD::SETULE: 
        Opc = Alpha::CMPTLE; break;
      case ISD::SETGT: case ISD::SETOGT: case ISD::SETUGT: 
        Opc = Alpha::CMPTLT; rev = true; break;
      case ISD::SETGE: case ISD::SETOGE: case ISD::SETUGE: 
        Opc = Alpha::CMPTLE; rev = true; break;
      case ISD::SETNE: case ISD::SETONE: case ISD::SETUNE:
        Opc = Alpha::CMPTEQ; inv = true; break;
      case ISD::SETO:
        Opc = Alpha::CMPTUN; inv = true; break;
      case ISD::SETUO:
        Opc = Alpha::CMPTUN; break;
      };
      SDValue tmp1 = N->getOperand(rev?1:0);
      SDValue tmp2 = N->getOperand(rev?0:1);
      SDNode *cmp = CurDAG->getMachineNode(Opc, dl, MVT::f64, tmp1, tmp2);
      if (inv) 
        cmp = CurDAG->getMachineNode(Alpha::CMPTEQ, dl, 
                                     MVT::f64, SDValue(cmp, 0), 
                                     CurDAG->getRegister(Alpha::F31, MVT::f64));
      switch(CC) {
      case ISD::SETUEQ: case ISD::SETULT: case ISD::SETULE:
      case ISD::SETUNE: case ISD::SETUGT: case ISD::SETUGE:
       {
         SDNode* cmp2 = CurDAG->getMachineNode(Alpha::CMPTUN, dl, MVT::f64,
                                               tmp1, tmp2);
         cmp = CurDAG->getMachineNode(Alpha::ADDT, dl, MVT::f64, 
                                      SDValue(cmp2, 0), SDValue(cmp, 0));
         break;
       }
      default: break;
      }

      SDNode* LD = CurDAG->getMachineNode(Alpha::FTOIT, dl,
                                          MVT::i64, SDValue(cmp, 0));
      return CurDAG->getMachineNode(Alpha::CMPULT, dl, MVT::i64, 
                                    CurDAG->getRegister(Alpha::R31, MVT::i64),
                                    SDValue(LD,0));
    }
    break;

  case ISD::AND: {
    ConstantSDNode* SC = NULL;
    ConstantSDNode* MC = NULL;
    if (N->getOperand(0).getOpcode() == ISD::SRL &&
        (MC = dyn_cast<ConstantSDNode>(N->getOperand(1))) &&
        (SC = dyn_cast<ConstantSDNode>(N->getOperand(0).getOperand(1)))) {
      uint64_t sval = SC->getZExtValue();
      uint64_t mval = MC->getZExtValue();
      // If the result is a zap, let the autogened stuff handle it.
      if (get_zapImm(N->getOperand(0), mval))
        break;
      // given mask X, and shift S, we want to see if there is any zap in the
      // mask if we play around with the botton S bits
      uint64_t dontcare = (~0ULL) >> (64 - sval);
      uint64_t mask = mval << sval;
      
      if (get_zapImm(mask | dontcare))
        mask = mask | dontcare;
      
      if (get_zapImm(mask)) {
        SDValue Z = 
          SDValue(CurDAG->getMachineNode(Alpha::ZAPNOTi, dl, MVT::i64,
                                         N->getOperand(0).getOperand(0),
                                         getI64Imm(get_zapImm(mask))), 0);
        return CurDAG->getMachineNode(Alpha::SRLr, dl, MVT::i64, Z, 
                                      getI64Imm(sval));
      }
    }
    break;
  }

  }

  return SelectCode(N);
}

void AlphaDAGToDAGISel::SelectCALL(SDNode *N) {
  //TODO: add flag stuff to prevent nondeturministic breakage!

  SDValue Chain = N->getOperand(0);
  SDValue Addr = N->getOperand(1);
  SDValue InFlag = N->getOperand(N->getNumOperands() - 1);
  DebugLoc dl = N->getDebugLoc();

   if (Addr.getOpcode() == AlphaISD::GPRelLo) {
     SDValue GOT = SDValue(getGlobalBaseReg(), 0);
     Chain = CurDAG->getCopyToReg(Chain, dl, Alpha::R29, GOT, InFlag);
     InFlag = Chain.getValue(1);
     Chain = SDValue(CurDAG->getMachineNode(Alpha::BSR, dl, MVT::Other, 
                                            MVT::Glue, Addr.getOperand(0),
                                            Chain, InFlag), 0);
   } else {
     Chain = CurDAG->getCopyToReg(Chain, dl, Alpha::R27, Addr, InFlag);
     InFlag = Chain.getValue(1);
     Chain = SDValue(CurDAG->getMachineNode(Alpha::JSR, dl, MVT::Other,
                                            MVT::Glue, Chain, InFlag), 0);
   }
   InFlag = Chain.getValue(1);

  ReplaceUses(SDValue(N, 0), Chain);
  ReplaceUses(SDValue(N, 1), InFlag);
}


/// createAlphaISelDag - This pass converts a legalized DAG into a 
/// Alpha-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createAlphaISelDag(AlphaTargetMachine &TM) {
  return new AlphaDAGToDAGISel(TM);
}
