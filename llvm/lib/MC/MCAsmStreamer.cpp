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
  raw_svector_ostream CommentStream;
public:
  MCAsmStreamer(MCContext &Context, formatted_raw_ostream &os,
                const MCAsmInfo &mai,
                bool isLittleEndian, bool isVerboseAsm, MCInstPrinter *printer,
                MCCodeEmitter *emitter)
    : MCStreamer(Context), OS(os), MAI(mai), IsLittleEndian(isLittleEndian),
      IsVerboseAsm(isVerboseAsm), InstPrinter(printer), Emitter(emitter),
      CommentStream(CommentToEmit) {}
  ~MCAsmStreamer() {}

  bool isLittleEndian() const { return IsLittleEndian; }
  
  inline void EmitEOL() {
    // If we don't have any comments, just emit a \n.
    if (!IsVerboseAsm) {
      OS << '\n';
      return;
    }
    EmitCommentsAndEOL();
  }
  void EmitCommentsAndEOL();

  /// isVerboseAsm - Return true if this streamer supports verbose assembly at
  /// all.
  virtual bool isVerboseAsm() const { return IsVerboseAsm; }
  
  /// AddComment - Add a comment that can be emitted to the generated .s
  /// file if applicable as a QoI issue to make the output of the compiler
  /// more readable.  This only affects the MCAsmStreamer, and only when
  /// verbose assembly output is enabled.
  virtual void AddComment(const Twine &T);
  
  /// GetCommentOS - Return a raw_ostream that comments can be written to.
  /// Unlike AddComment, you are required to terminate comments with \n if you
  /// use this method.
  virtual raw_ostream &GetCommentOS() {
    if (!IsVerboseAsm)
      return nulls();  // Discard comments unless in verbose asm mode.
    return CommentStream;
  }
  
  /// AddBlankLine - Emit a blank line to a .s file to pretty it up.
  virtual void AddBlankLine() {
    EmitEOL();
  }
  
  /// @name MCStreamer Interface
  /// @{

  virtual void SwitchSection(const MCSection *Section);

  virtual void EmitLabel(MCSymbol *Symbol);

  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);

  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);

  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);

  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);

  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);

  /// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
  ///
  /// @param Symbol - The common symbol to emit.
  /// @param Size - The size of the common symbol.
  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size);
  
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0);

  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);

  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitIntValue(uint64_t Value, unsigned Size, unsigned AddrSpace);
  virtual void EmitGPRel32Value(const MCExpr *Value);
  

  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                        unsigned AddrSpace);

  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);

  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);

  virtual void EmitFileDirective(StringRef Filename);
  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename);

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
  
  // Make sure that CommentStream is flushed.
  CommentStream.flush();
  
  T.toVector(CommentToEmit);
  // Each comment goes on its own line.
  CommentToEmit.push_back('\n');
  
  // Tell the comment stream that the vector changed underneath it.
  CommentStream.resync();
}

void MCAsmStreamer::EmitCommentsAndEOL() {
  if (CommentToEmit.empty() && CommentStream.GetNumBytesInBuffer() == 0) {
    OS << '\n';
    return;
  }
  
  CommentStream.flush();
  StringRef Comments = CommentToEmit.str();
  
  assert(Comments.back() == '\n' &&
         "Comment array not newline terminated");
  do {
    // Emit a line of comments.
    OS.PadToColumn(MAI.getCommentColumn());
    size_t Position = Comments.find('\n');
    OS << MAI.getCommentString() << ' ' << Comments.substr(0, Position) << '\n';
    
    Comments = Comments.substr(Position+1);
  } while (!Comments.empty());
  
  CommentToEmit.clear();
  // Tell the comment stream that the vector changed underneath it.
  CommentStream.resync();
}


static inline int64_t truncateToSize(int64_t Value, unsigned Bytes) {
  assert(Bytes && "Invalid size!");
  return Value & ((uint64_t) (int64_t) -1 >> (64 - Bytes * 8));
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

void MCAsmStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  default: assert(0 && "Invalid flag!");
  case MCAF_SubsectionsViaSymbols: OS << ".subsections_via_symbols"; break;
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
                                        MCSymbolAttr Attribute) {
  switch (Attribute) {
  case MCSA_Invalid: assert(0 && "Invalid symbol attribute");
  case MCSA_ELF_TypeFunction:    /// .type _foo, STT_FUNC  # aka @function
  case MCSA_ELF_TypeIndFunction: /// .type _foo, STT_GNU_IFUNC
  case MCSA_ELF_TypeObject:      /// .type _foo, STT_OBJECT  # aka @object
  case MCSA_ELF_TypeTLS:         /// .type _foo, STT_TLS     # aka @tls_object
  case MCSA_ELF_TypeCommon:      /// .type _foo, STT_COMMON  # aka @common
  case MCSA_ELF_TypeNoType:      /// .type _foo, STT_NOTYPE  # aka @notype
    assert(MAI.hasDotTypeDotSizeDirective() && "Symbol Attr not supported");
    OS << "\t.type " << *Symbol << ','
       << ((MAI.getCommentString()[0] != '@') ? '@' : '%');
    switch (Attribute) {
    default: assert(0 && "Unknown ELF .type");
    case MCSA_ELF_TypeFunction:    OS << "function"; break;
    case MCSA_ELF_TypeIndFunction: OS << "gnu_indirect_function"; break;
    case MCSA_ELF_TypeObject:      OS << "object"; break;
    case MCSA_ELF_TypeTLS:         OS << "tls_object"; break;
    case MCSA_ELF_TypeCommon:      OS << "common"; break;
    case MCSA_ELF_TypeNoType:      OS << "no_type"; break;
    }
    EmitEOL();
    return;
  case MCSA_Global: // .globl/.global
    OS << MAI.getGlobalDirective();
    break;
  case MCSA_Hidden:         OS << ".hidden ";          break;
  case MCSA_IndirectSymbol: OS << ".indirect_symbol "; break;
  case MCSA_Internal:       OS << ".internal ";        break;
  case MCSA_LazyReference:  OS << ".lazy_reference ";  break;
  case MCSA_Local:          OS << ".local ";           break;
  case MCSA_NoDeadStrip:    OS << ".no_dead_strip ";   break;
  case MCSA_PrivateExtern:  OS << ".private_extern ";  break;
  case MCSA_Protected:      OS << ".protected ";       break;
  case MCSA_Reference:      OS << ".reference ";       break;
  case MCSA_Weak:           OS << ".weak ";            break;
  case MCSA_WeakDefinition: OS << ".weak_definition "; break;
      // .weak_reference
  case MCSA_WeakReference:  OS << MAI.getWeakRefDirective(); break;
  }

  OS << *Symbol;
  EmitEOL();
}

void MCAsmStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  OS << ".desc" << ' ' << *Symbol << ',' << DescValue;
  EmitEOL();
}

void MCAsmStreamer::EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  assert(MAI.hasDotTypeDotSizeDirective());
  OS << "\t.size\t" << *Symbol << ", " << *Value << '\n';
}

void MCAsmStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment) {
  OS << "\t.comm\t" << *Symbol << ',' << Size;
  if (ByteAlignment != 0) {
    if (MAI.getCOMMDirectiveAlignmentIsInBytes())
      OS << ',' << ByteAlignment;
    else
      OS << ',' << Log2_32(ByteAlignment);
  }
  EmitEOL();
}

/// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
///
/// @param Symbol - The common symbol to emit.
/// @param Size - The size of the common symbol.
void MCAsmStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
  assert(MAI.hasLCOMMDirective() && "Doesn't have .lcomm, can't emit it!");
  OS << "\t.lcomm\t" << *Symbol << ',' << Size;
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

static inline char toOctal(int X) { return (X&7)+'0'; }

static void PrintQuotedString(StringRef Data, raw_ostream &OS) {
  OS << '"';
  
  for (unsigned i = 0, e = Data.size(); i != e; ++i) {
    unsigned char C = Data[i];
    if (C == '"' || C == '\\') {
      OS << '\\' << (char)C;
      continue;
    }
    
    if (isprint((unsigned char)C)) {
      OS << (char)C;
      continue;
    }
    
    switch (C) {
      case '\b': OS << "\\b"; break;
      case '\f': OS << "\\f"; break;
      case '\n': OS << "\\n"; break;
      case '\r': OS << "\\r"; break;
      case '\t': OS << "\\t"; break;
      default:
        OS << '\\';
        OS << toOctal(C >> 6);
        OS << toOctal(C >> 3);
        OS << toOctal(C >> 0);
        break;
    }
  }
  
  OS << '"';
}


void MCAsmStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  assert(CurSection && "Cannot emit contents before setting section!");
  if (Data.empty()) return;
  
  if (Data.size() == 1) {
    OS << MAI.getData8bitsDirective(AddrSpace);
    OS << (unsigned)(unsigned char)Data[0];
    EmitEOL();
    return;
  }

  // If the data ends with 0 and the target supports .asciz, use it, otherwise
  // use .ascii
  if (MAI.getAscizDirective() && Data.back() == 0) {
    OS << MAI.getAscizDirective();
    Data = Data.substr(0, Data.size()-1);
  } else {
    OS << MAI.getAsciiDirective();
  }

  OS << ' ';
  PrintQuotedString(Data, OS);
  EmitEOL();
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
  OS << Directive << *Value;
  EmitEOL();
}

void MCAsmStreamer::EmitGPRel32Value(const MCExpr *Value) {
  assert(MAI.getGPRel32Directive() != 0);
  OS << MAI.getGPRel32Directive() << *Value;
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


void MCAsmStreamer::EmitFileDirective(StringRef Filename) {
  assert(MAI.hasSingleParameterDotFile());
  OS << "\t.file\t";
  PrintQuotedString(Filename, OS);
  EmitEOL();
}

void MCAsmStreamer::EmitDwarfFileDirective(unsigned FileNo, StringRef Filename){
  OS << "\t.file\t" << FileNo << ' ';
  PrintQuotedString(Filename, OS);
  EmitEOL();
}


void MCAsmStreamer::EmitInstruction(const MCInst &Inst) {
  assert(CurSection && "Cannot emit contents before setting section!");

  // Show the encoding in a comment if we have a code emitter.
  if (Emitter) {
    SmallString<256> Code;
    raw_svector_ostream VecOS(Code);
    Emitter->EncodeInstruction(Inst, VecOS);
    VecOS.flush();
    
    raw_ostream &OS = GetCommentOS();
    OS << "encoding: [";
    for (unsigned i = 0, e = Code.size(); i != e; ++i) {
      if (i)
        OS << ',';
      OS << format("%#04x", uint8_t(Code[i]));
    }
    OS << "]\n";
  }
  
  // If we have an AsmPrinter, use that to print.
  if (InstPrinter) {
    InstPrinter->printInst(&Inst);
    EmitEOL();
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
