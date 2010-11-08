//===- PTXMCAsmStreamer.h - PTX Text Assembly Output ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTXMCAsmStreamer class.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_MCASMSTREAMER_H
#define PTX_MCASMSTREAMER_H

#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
class MCContext;
class MCInstPrinter;
class Twine;

class PTXMCAsmStreamer : public MCStreamer {
  formatted_raw_ostream &OS;
  const MCAsmInfo &MAI;
  OwningPtr<MCInstPrinter> InstPrinter;
  OwningPtr<MCCodeEmitter> Emitter;

  SmallString<128> CommentToEmit;
  raw_svector_ostream CommentStream;

  unsigned IsLittleEndian : 1;
  unsigned IsVerboseAsm : 1;
  unsigned ShowInst : 1;

public:
  PTXMCAsmStreamer(MCContext &Context,
                   formatted_raw_ostream &os,
                   bool isLittleEndian,
                   bool isVerboseAsm,
                   MCInstPrinter *printer,
                   MCCodeEmitter *emitter,
                   bool showInst)
    : MCStreamer(Context), OS(os), MAI(Context.getAsmInfo()),
      InstPrinter(printer), Emitter(emitter), CommentStream(CommentToEmit),
      IsLittleEndian(isLittleEndian), IsVerboseAsm(isVerboseAsm),
      ShowInst(showInst) {
    if (InstPrinter && IsVerboseAsm)
      InstPrinter->setCommentStream(CommentStream);
  }

  ~PTXMCAsmStreamer() {}

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

  /// hasRawTextSupport - We support EmitRawText.
  virtual bool hasRawTextSupport() const { return true; }

  /// AddComment - Add a comment that can be emitted to the generated .s
  /// file if applicable as a QoI issue to make the output of the compiler
  /// more readable.  This only affects the MCAsmStreamer, and only when
  /// verbose assembly output is enabled.
  virtual void AddComment(const Twine &T);

  /// AddEncodingComment - Add a comment showing the encoding of an instruction.
  virtual void AddEncodingComment(const MCInst &Inst);

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

  virtual void InitSections() {
    // FIXME, this is MachO specific, but the testsuite
    // expects this.
    SwitchSection(getContext().
                  getMachOSection("__TEXT", "__text",
                                  MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                  0, SectionKind::getText()));
  }

  virtual void EmitLabel(MCSymbol *Symbol);

  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);

  virtual void EmitThumbFunc(MCSymbol *Func);

  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);

  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);

  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);

  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol);
  virtual void EmitCOFFSymbolStorageClass(int StorageClass);
  virtual void EmitCOFFSymbolType(int Type);
  virtual void EndCOFFSymbolDef();
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

  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0);

  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);

  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitIntValue(uint64_t Value, unsigned Size, unsigned AddrSpace);
  virtual void EmitULEB128Value(const MCExpr *Value, unsigned AddrSpace = 0);
  virtual void EmitSLEB128Value(const MCExpr *Value, unsigned AddrSpace = 0);
  virtual void EmitGPRel32Value(const MCExpr *Value);


  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                        unsigned AddrSpace);

  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);

  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);

  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);

  virtual void EmitFileDirective(StringRef Filename);
  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename);

  virtual void EmitInstruction(const MCInst &Inst);

  /// EmitRawText - If this file is backed by an assembly streamer, this dumps
  /// the specified string in the output .s file.  This capability is
  /// indicated by the hasRawTextSupport() predicate.
  virtual void EmitRawText(StringRef String);

  virtual void Finish();

  /// @}

}; // class PTXMCAsmStreamer

MCStreamer *createPTXAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                 bool isLittleEndian, bool isVerboseAsm,
                                 MCInstPrinter *InstPrint,
                                 MCCodeEmitter *CE,
                                 bool ShowInst);
} // namespace llvm

#endif // PTX_MCASMSTREAMER_H
