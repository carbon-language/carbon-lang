//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

namespace {

class MCAsmStreamer : public MCStreamer {
  formatted_raw_ostream &OS;
  const MCAsmInfo &MAI;
  bool IsLittleEndian, IsVerboseAsm;
  MCInstPrinter *InstPrinter;
  MCCodeEmitter *Emitter;
  
  SmallString<128> CommentToEmit;
public:
  MCAsmStreamer(MCContext &Context, formatted_raw_ostream &os,
                const MCAsmInfo &mai,
                bool isLittleEndian, bool isVerboseAsm, MCInstPrinter *printer,
                MCCodeEmitter *emitter)
    : MCStreamer(Context), OS(os), MAI(mai), IsLittleEndian(isLittleEndian),
      IsVerboseAsm(isVerboseAsm), InstPrinter(printer), Emitter(emitter) {}
  ~MCAsmStreamer() {}

  bool isLittleEndian() const { return IsLittleEndian; }
  
  
  inline void EmitEOL() {
    if (CommentToEmit.empty()) {
      OS << '\n';
      return;
    }
    EmitCommentsAndEOL();
  }
  void EmitCommentsAndEOL();
  
  /// AddComment - Add a comment that can be emitted to the generated .s
  /// file if applicable as a QoI issue to make the output of the compiler
  /// more readable.  This only affects the MCAsmStreamer, and only when
  /// verbose assembly output is enabled.
  virtual void AddComment(const Twine &T);
  
  /// @name MCStreamer Interface
  /// @{

  virtual void SwitchSection(const MCSection *Section);

  virtual void EmitLabel(MCSymbol *Symbol);

  virtual void EmitAssemblerFlag(AssemblerFlag Flag);

  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);

  virtual void EmitSymbolAttribute(MCSymbol *Symbol, SymbolAttr Attribute);

  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);

  virtual void EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                unsigned ByteAlignment);

  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0);

  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);

  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitIntValue(uint64_t Value, unsigned Size, unsigned AddrSpace);

  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                        unsigned AddrSpace);

  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);

  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);
  
  virtual void EmitInstruction(const MCInst &Inst);

  virtual void Finish();
  
  /// @}
};

} // end anonymous namespace.

/// AddComment - Add a comment that can be emitted to the generated .s
/// file if applicable as a QoI issue to make the output of the compiler
/// more readable.  This only affects the MCAsmStreamer, and only when
/// verbose assembly output is enabled.
void MCAsmStreamer::AddComment(const Twine &T) {
  if (!IsVerboseAsm) return;
  // Each comment goes on its own line.
  if (!CommentToEmit.empty())
    CommentToEmit.push_back('\n');
  T.toVector(CommentToEmit);
}

void MCAsmStreamer::EmitCommentsAndEOL() {
  StringRef Comments = CommentToEmit.str();
  while (!Comments.empty()) {
    // Emit a line of comments.
    OS.PadToColumn(MAI.getCommentColumn());
    size_t Position = Comments.find('\n');
    OS << MAI.getCommentString() << ' ' << Comments.substr(0, Position) << '\n';
    
    if (Position == StringRef::npos) break;
    Comments = Comments.substr(Position+1);
  }
  
  CommentToEmit.clear();
}


static inline int64_t truncateToSize(int64_t Value, unsigned Bytes) {
  assert(Bytes && "Invalid size!");
  return Value & ((uint64_t) (int64_t) -1 >> (64 - Bytes * 8));
}

static inline const MCExpr *truncateToSize(const MCExpr *Value,
                                           unsigned Bytes) {
  // FIXME: Do we really need this routine?
  return Value;
}

void MCAsmStreamer::SwitchSection(const MCSection *Section) {
  assert(Section && "Cannot switch to a null section!");
  if (Section != CurSection) {
    CurSection = Section;
    Section->PrintSwitchToSection(MAI, OS);
  }
}

void MCAsmStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  assert(CurSection && "Cannot emit before setting section!");

  OS << *Symbol << ":";
  EmitEOL();
  Symbol->setSection(*CurSection);
}

void MCAsmStreamer::EmitAssemblerFlag(AssemblerFlag Flag) {
  switch (Flag) {
  default: assert(0 && "Invalid flag!");
  case SubsectionsViaSymbols: OS << ".subsections_via_symbols"; break;
  }
  EmitEOL();
}

void MCAsmStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // Only absolute symbols can be redefined.
  assert((Symbol->isUndefined() || Symbol->isAbsolute()) &&
         "Cannot define a symbol twice!");

  OS << *Symbol << " = " << *Value;
  EmitEOL();

  // FIXME: Lift context changes into super class.
  // FIXME: Set associated section.
  Symbol->setValue(Value);
}

void MCAsmStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                        SymbolAttr Attribute) {
  switch (Attribute) {
  case Global:         OS << MAI.getGlobalDirective(); break; // .globl
  case Hidden:         OS << ".hidden ";          break;
  case IndirectSymbol: OS << ".indirect_symbol "; break;
  case Internal:       OS << ".internal ";        break;
  case LazyReference:  OS << ".lazy_reference ";  break;
  case NoDeadStrip:    OS << ".no_dead_strip ";   break;
  case PrivateExtern:  OS << ".private_extern ";  break;
  case Protected:      OS << ".protected ";       break;
  case Reference:      OS << ".reference ";       break;
  case Weak:           OS << ".weak ";            break;
  case WeakDefinition: OS << ".weak_definition "; break;
  case WeakReference:  OS << ".weak_reference ";  break;
  }

  OS << *Symbol;
  EmitEOL();
}

void MCAsmStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  OS << ".desc" << ' ' << *Symbol << ',' << DescValue;
  EmitEOL();
}

void MCAsmStreamer::EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                     unsigned ByteAlignment) {
  OS << MAI.getCOMMDirective() << *Symbol << ',' << Size;
  if (ByteAlignment != 0 && MAI.getCOMMDirectiveTakesAlignment()) {
    if (MAI.getAlignmentIsInBytes())
      OS << ',' << ByteAlignment;
    else
      OS << ',' << Log2_32(ByteAlignment);
  }
  EmitEOL();
}

void MCAsmStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                 unsigned Size, unsigned ByteAlignment) {
  // Note: a .zerofill directive does not switch sections.
  OS << ".zerofill ";
  
  // This is a mach-o specific directive.
  const MCSectionMachO *MOSection = ((const MCSectionMachO*)Section);
  OS << MOSection->getSegmentName() << "," << MOSection->getSectionName();
  
  if (Symbol != NULL) {
    OS << ',' << *Symbol << ',' << Size;
    if (ByteAlignment != 0)
      OS << ',' << Log2_32(ByteAlignment);
  }
  EmitEOL();
}

void MCAsmStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  assert(CurSection && "Cannot emit contents before setting section!");
  const char *Directive = MAI.getData8bitsDirective(AddrSpace);
  for (unsigned i = 0, e = Data.size(); i != e; ++i) {
    OS << Directive << (unsigned)(unsigned char)Data[i];
    EmitEOL();
  }
}

/// EmitIntValue - Special case of EmitValue that avoids the client having
/// to pass in a MCExpr for constant integers.
void MCAsmStreamer::EmitIntValue(uint64_t Value, unsigned Size,
                                 unsigned AddrSpace) {
  assert(CurSection && "Cannot emit contents before setting section!");
  const char *Directive = 0;
  switch (Size) {
  default: break;
  case 1: Directive = MAI.getData8bitsDirective(AddrSpace); break;
  case 2: Directive = MAI.getData16bitsDirective(AddrSpace); break;
  case 4: Directive = MAI.getData32bitsDirective(AddrSpace); break;
  case 8:
    Directive = MAI.getData64bitsDirective(AddrSpace);
    // If the target doesn't support 64-bit data, emit as two 32-bit halves.
    if (Directive) break;
    if (isLittleEndian()) {
      EmitIntValue((uint32_t)(Value >> 0 ), 4, AddrSpace);
      EmitIntValue((uint32_t)(Value >> 32), 4, AddrSpace);
    } else {
      EmitIntValue((uint32_t)(Value >> 32), 4, AddrSpace);
      EmitIntValue((uint32_t)(Value >> 0 ), 4, AddrSpace);
    }
    return;
  }
  
  assert(Directive && "Invalid size for machine code value!");
  OS << Directive << truncateToSize(Value, Size);
  EmitEOL();
}

void MCAsmStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                              unsigned AddrSpace) {
  assert(CurSection && "Cannot emit contents before setting section!");
  const char *Directive = 0;
  switch (Size) {
  default: break;
  case 1: Directive = MAI.getData8bitsDirective(AddrSpace); break;
  case 2: Directive = MAI.getData16bitsDirective(AddrSpace); break;
  case 4: Directive = MAI.getData32bitsDirective(AddrSpace); break;
  case 8: Directive = MAI.getData64bitsDirective(AddrSpace); break;
  }
  
  assert(Directive && "Invalid size for machine code value!");
  OS << Directive << *truncateToSize(Value, Size);
  EmitEOL();
}

/// EmitFill - Emit NumBytes bytes worth of the value specified by
/// FillValue.  This implements directives such as '.space'.
void MCAsmStreamer::EmitFill(uint64_t NumBytes, uint8_t FillValue,
                             unsigned AddrSpace) {
  if (NumBytes == 0) return;
  
  if (AddrSpace == 0)
    if (const char *ZeroDirective = MAI.getZeroDirective()) {
      OS << ZeroDirective << NumBytes;
      if (FillValue != 0)
        OS << ',' << (int)FillValue;
      EmitEOL();
      return;
    }

  // Emit a byte at a time.
  MCStreamer::EmitFill(NumBytes, FillValue, AddrSpace);
}

void MCAsmStreamer::EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                         unsigned ValueSize,
                                         unsigned MaxBytesToEmit) {
  // Some assemblers don't support non-power of two alignments, so we always
  // emit alignments as a power of two if possible.
  if (isPowerOf2_32(ByteAlignment)) {
    switch (ValueSize) {
    default: llvm_unreachable("Invalid size for machine code value!");
    case 1: OS << MAI.getAlignDirective(); break;
    // FIXME: use MAI for this!
    case 2: OS << ".p2alignw "; break;
    case 4: OS << ".p2alignl "; break;
    case 8: llvm_unreachable("Unsupported alignment size!");
    }
    
    if (MAI.getAlignmentIsInBytes())
      OS << ByteAlignment;
    else
      OS << Log2_32(ByteAlignment);

    if (Value || MaxBytesToEmit) {
      OS << ", 0x";
      OS.write_hex(truncateToSize(Value, ValueSize));

      if (MaxBytesToEmit) 
        OS << ", " << MaxBytesToEmit;
    }
    EmitEOL();
    return;
  }
  
  // Non-power of two alignment.  This is not widely supported by assemblers.
  // FIXME: Parameterize this based on MAI.
  switch (ValueSize) {
  default: llvm_unreachable("Invalid size for machine code value!");
  case 1: OS << ".balign";  break;
  case 2: OS << ".balignw"; break;
  case 4: OS << ".balignl"; break;
  case 8: llvm_unreachable("Unsupported alignment size!");
  }

  OS << ' ' << ByteAlignment;
  OS << ", " << truncateToSize(Value, ValueSize);
  if (MaxBytesToEmit) 
    OS << ", " << MaxBytesToEmit;
  EmitEOL();
}

void MCAsmStreamer::EmitValueToOffset(const MCExpr *Offset,
                                      unsigned char Value) {
  // FIXME: Verify that Offset is associated with the current section.
  OS << ".org " << *Offset << ", " << (unsigned) Value;
  EmitEOL();
}

void MCAsmStreamer::EmitInstruction(const MCInst &Inst) {
  assert(CurSection && "Cannot emit contents before setting section!");

  // If we have an AsmPrinter, use that to print.
  if (InstPrinter) {
    InstPrinter->printInst(&Inst);
    EmitEOL();

    // Show the encoding if we have a code emitter.
    if (Emitter) {
      SmallString<256> Code;
      raw_svector_ostream VecOS(Code);
      Emitter->EncodeInstruction(Inst, VecOS);
      VecOS.flush();
  
      OS.indent(20);
      OS << " # encoding: [";
      for (unsigned i = 0, e = Code.size(); i != e; ++i) {
        if (i)
          OS << ',';
        OS << format("%#04x", uint8_t(Code[i]));
      }
      OS << "]\n";
    }

    return;
  }

  // Otherwise fall back to a structural printing for now. Eventually we should
  // always have access to the target specific printer.
  Inst.print(OS, &MAI);
  EmitEOL();
}

void MCAsmStreamer::Finish() {
  OS.flush();
}
    
MCStreamer *llvm::createAsmStreamer(MCContext &Context,
                                    formatted_raw_ostream &OS,
                                    const MCAsmInfo &MAI, bool isLittleEndian,
                                    bool isVerboseAsm, MCInstPrinter *IP,
                                    MCCodeEmitter *CE) {
  return new MCAsmStreamer(Context, OS, MAI, isLittleEndian, isVerboseAsm,
                           IP, CE);
}
