//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

  class MCAsmStreamer : public MCStreamer {
    raw_ostream &OS;

    MCSection *CurSection;

  public:
    MCAsmStreamer(MCContext &Context, raw_ostream &_OS)
      : MCStreamer(Context), OS(_OS), CurSection(0) {}
    ~MCAsmStreamer() {}

    /// @name MCStreamer Interface
    /// @{

    virtual void SwitchSection(MCSection *Section);

    virtual void EmitLabel(MCSymbol *Symbol);

    virtual void SubsectionsViaSymbols(void);

    virtual void EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                                bool MakeAbsolute = false);

    virtual void EmitSymbolAttribute(MCSymbol *Symbol, SymbolAttr Attribute);

    virtual void EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                  unsigned Pow2Alignment, bool IsLocal);

    virtual void EmitZerofill(MCSection *Section, MCSymbol *Symbol = NULL,
                              unsigned Size = 0, unsigned Pow2Alignment = 0);

    virtual void AbortAssembly(const char *AbortReason = NULL);

    virtual void EmitBytes(const char *Data, unsigned Length);

    virtual void EmitValue(const MCValue &Value, unsigned Size);

    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0);

    virtual void EmitValueToOffset(const MCValue &Offset, 
                                   unsigned char Value = 0);
    
    virtual void EmitInstruction(const MCInst &Inst);

    virtual void Finish();
    
    /// @}
  };

}

/// Allow printing values directly to a raw_ostream.
static inline raw_ostream &operator<<(raw_ostream &os, const MCValue &Value) {
  if (Value.getSymA()) {
    os << Value.getSymA()->getName();
    if (Value.getSymB())
      os << " - " << Value.getSymB()->getName();
    if (Value.getConstant())
      os << " + " << Value.getConstant();
  } else {
    assert(!Value.getSymB() && "Invalid machine code value!");
    os << Value.getConstant();
  }

  return os;
}

static inline int64_t truncateToSize(int64_t Value, unsigned Bytes) {
  assert(Bytes && "Invalid size!");
  return Value & ((uint64_t) (int64_t) -1 >> (64 - Bytes * 8));
}

static inline MCValue truncateToSize(const MCValue &Value, unsigned Bytes) {
  return MCValue::get(Value.getSymA(), Value.getSymB(), 
                      truncateToSize(Value.getConstant(), Bytes));
}

void MCAsmStreamer::SwitchSection(MCSection *Section) {
  if (Section != CurSection) {
    CurSection = Section;

    // FIXME: Really we would like the segment, flags, etc. to be separate
    // values instead of embedded in the name. Not all assemblers understand all
    // this stuff though.
    OS << ".section " << Section->getName() << "\n";
  }
}

void MCAsmStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->getSection() == 0 && "Cannot emit a symbol twice!");
  assert(CurSection && "Cannot emit before setting section!");
  assert(!getContext().GetSymbolValue(Symbol) && 
         "Cannot emit symbol which was directly assigned to!");

  OS << Symbol->getName() << ":\n";
  Symbol->setSection(CurSection);
  Symbol->setExternal(false);
}

void MCAsmStreamer::SubsectionsViaSymbols(void) {
  OS << ".subsections_via_symbols\n";
}

void MCAsmStreamer::AbortAssembly(const char *AbortReason) {
  OS << ".abort";
  if (AbortReason != NULL)
    OS << ' ' << AbortReason;
  OS << '\n';
  
}

void MCAsmStreamer::EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                                   bool MakeAbsolute) {
  assert(!Symbol->getSection() && "Cannot assign to a label!");

  if (MakeAbsolute) {
    OS << ".set " << Symbol->getName() << ", " << Value << '\n';
  } else {
    OS << Symbol->getName() << " = " << Value << '\n';
  }

  getContext().SetSymbolValue(Symbol, Value);
}

void MCAsmStreamer::EmitSymbolAttribute(MCSymbol *Symbol, 
                                        SymbolAttr Attribute) {
  switch (Attribute) {
  case Global: OS << ".globl"; break;
  case Hidden: OS << ".hidden"; break;
  case IndirectSymbol: OS << ".indirect_symbol"; break;
  case Internal: OS << ".internal"; break;
  case LazyReference: OS << ".lazy_reference"; break;
  case NoDeadStrip: OS << ".no_dead_strip"; break;
  case PrivateExtern: OS << ".private_extern"; break;
  case Protected: OS << ".protected"; break;
  case Reference: OS << ".reference"; break;
  case Weak: OS << ".weak"; break;
  case WeakDefinition: OS << ".weak_definition"; break;
  case WeakReference: OS << ".weak_reference"; break;
  }

  OS << ' ' << Symbol->getName() << '\n';
}

void MCAsmStreamer::EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                     unsigned Pow2Alignment, bool IsLocal) {
  if (IsLocal)
    OS << ".lcomm";
  else
    OS << ".comm";
  OS << ' ' << Symbol->getName() << ',' << Size;
  if (Pow2Alignment != 0)
    OS << ',' << Pow2Alignment;
  OS << '\n';
}

void MCAsmStreamer::EmitZerofill(MCSection *Section, MCSymbol *Symbol,
                                 unsigned Size, unsigned Pow2Alignment) {
  // Note: a .zerofill directive does not switch sections
  // FIXME: Really we would like the segment and section names as well as the
  // section type to be separate values instead of embedded in the name. Not
  // all assemblers understand all this stuff though.
  OS << ".zerofill " << Section->getName();
  if (Symbol != NULL) {
    OS << ',' << Symbol->getName() << ',' << Size;
    if (Pow2Alignment != 0)
      OS << ',' << Pow2Alignment;
  }
  OS << '\n';
}

void MCAsmStreamer::EmitBytes(const char *Data, unsigned Length) {
  assert(CurSection && "Cannot emit contents before setting section!");
  for (unsigned i = 0; i != Length; ++i)
    OS << ".byte " << (unsigned) Data[i] << '\n';
}

void MCAsmStreamer::EmitValue(const MCValue &Value, unsigned Size) {
  assert(CurSection && "Cannot emit contents before setting section!");
  // Need target hooks to know how to print this.
  switch (Size) {
  default:
    LLVM_UNREACHABLE("Invalid size for machine code value!");
  case 1: OS << ".byte"; break;
  case 2: OS << ".short"; break;
  case 4: OS << ".long"; break;
  case 8: OS << ".quad"; break;
  }

  OS << ' ' << truncateToSize(Value, Size) << '\n';
}

void MCAsmStreamer::EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                         unsigned ValueSize,
                                         unsigned MaxBytesToEmit) {
  // Some assemblers don't support .balign, so we always emit as .p2align if
  // this is a power of two. Otherwise we assume the client knows the target
  // supports .balign and use that.
  unsigned Pow2 = Log2_32(ByteAlignment);
  bool IsPow2 = (1U << Pow2) == ByteAlignment;

  switch (ValueSize) {
  default:
    LLVM_UNREACHABLE("Invalid size for machine code value!");
  case 8:
    LLVM_UNREACHABLE("Unsupported alignment size!");
  case 1: OS << (IsPow2 ? ".p2align" : ".balign"); break;
  case 2: OS << (IsPow2 ? ".p2alignw" : ".balignw"); break;
  case 4: OS << (IsPow2 ? ".p2alignl" : ".balignl"); break;
  }

  OS << ' ' << (IsPow2 ? Pow2 : ByteAlignment);

  OS << ", " << truncateToSize(Value, ValueSize);
  if (MaxBytesToEmit) 
    OS << ", " << MaxBytesToEmit;
  OS << '\n';
}

void MCAsmStreamer::EmitValueToOffset(const MCValue &Offset, 
                                      unsigned char Value) {
  // FIXME: Verify that Offset is associated with the current section.
  OS << ".org " << Offset << ", " << (unsigned) Value << '\n';
}

static raw_ostream &operator<<(raw_ostream &OS, const MCOperand &Op) {
  if (Op.isReg())
    return OS << "reg:" << Op.getReg();
  if (Op.isImm())
    return OS << "imm:" << Op.getImm();
  if (Op.isMBBLabel())
    return OS << "mbblabel:(" 
              << Op.getMBBLabelFunction() << ", " << Op.getMBBLabelBlock();
  assert(Op.isMCValue() && "Invalid operand!");
  return OS << "val:" << Op.getMCValue();
}

void MCAsmStreamer::EmitInstruction(const MCInst &Inst) {
  assert(CurSection && "Cannot emit contents before setting section!");
  // FIXME: Implement proper printing.
  OS << "MCInst("
     << "opcode=" << Inst.getOpcode() << ", "
     << "operands=[";
  for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i) {
    if (i)
      OS << ", ";
    OS << Inst.getOperand(i);
  }
  OS << "])\n";
}

void MCAsmStreamer::Finish() {
  OS.flush();
}
    
MCStreamer *llvm::createAsmStreamer(MCContext &Context, raw_ostream &OS) {
  return new MCAsmStreamer(Context, OS);
}
