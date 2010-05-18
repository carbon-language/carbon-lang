//===- MCStreamer.h - High-level Streaming Machine Code Output --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCStreamer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSTREAMER_H
#define LLVM_MC_MCSTREAMER_H

#include "llvm/System/DataTypes.h"
#include "llvm/MC/MCDirectives.h"

namespace llvm {
  class MCAsmInfo;
  class MCCodeEmitter;
  class MCContext;
  class MCExpr;
  class MCInst;
  class MCInstPrinter;
  class MCSection;
  class MCSymbol;
  class StringRef;
  class TargetAsmBackend;
  class Twine;
  class raw_ostream;
  class formatted_raw_ostream;

  /// MCStreamer - Streaming machine code generation interface.  This interface
  /// is intended to provide a programatic interface that is very similar to the
  /// level that an assembler .s file provides.  It has callbacks to emit bytes,
  /// handle directives, etc.  The implementation of this interface retains
  /// state to know what the current section is etc.
  ///
  /// There are multiple implementations of this interface: one for writing out
  /// a .s file, and implementations that write out .o files of various formats.
  ///
  class MCStreamer {
    MCContext &Context;

    MCStreamer(const MCStreamer&); // DO NOT IMPLEMENT
    MCStreamer &operator=(const MCStreamer&); // DO NOT IMPLEMENT

  protected:
    MCStreamer(MCContext &Ctx);

    /// CurSection - This is the current section code is being emitted to, it is
    /// kept up to date by SwitchSection.
    const MCSection *CurSection;

  public:
    virtual ~MCStreamer();

    MCContext &getContext() const { return Context; }

    /// @name Assembly File Formatting.
    /// @{
    
    /// isVerboseAsm - Return true if this streamer supports verbose assembly
    /// and if it is enabled.
    virtual bool isVerboseAsm() const { return false; }
    
    /// hasRawTextSupport - Return true if this asm streamer supports emitting
    /// unformatted text to the .s file with EmitRawText.
    virtual bool hasRawTextSupport() const { return false; }

    /// AddComment - Add a comment that can be emitted to the generated .s
    /// file if applicable as a QoI issue to make the output of the compiler
    /// more readable.  This only affects the MCAsmStreamer, and only when
    /// verbose assembly output is enabled.
    ///
    /// If the comment includes embedded \n's, they will each get the comment
    /// prefix as appropriate.  The added comment should not end with a \n.
    virtual void AddComment(const Twine &T) {}
    
    /// GetCommentOS - Return a raw_ostream that comments can be written to.
    /// Unlike AddComment, you are required to terminate comments with \n if you
    /// use this method.
    virtual raw_ostream &GetCommentOS();
    
    /// AddBlankLine - Emit a blank line to a .s file to pretty it up.
    virtual void AddBlankLine() {}
    
    /// @}
    
    /// @name Symbol & Section Management
    /// @{
    
    /// getCurrentSection - Return the current section that the streamer is
    /// emitting code to.
    const MCSection *getCurrentSection() const { return CurSection; }

    /// SwitchSection - Set the current section where code is being emitted to
    /// @p Section.  This is required to update CurSection.
    ///
    /// This corresponds to assembler directives like .section, .text, etc.
    virtual void SwitchSection(const MCSection *Section) = 0;
    
    /// EmitLabel - Emit a label for @p Symbol into the current section.
    ///
    /// This corresponds to an assembler statement such as:
    ///   foo:
    ///
    /// @param Symbol - The symbol to emit. A given symbol should only be
    /// emitted as a label once, and symbols emitted as a label should never be
    /// used in an assignment.
    virtual void EmitLabel(MCSymbol *Symbol) = 0;

    /// EmitAssemblerFlag - Note in the output the specified @p Flag
    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) = 0;

    /// EmitAssignment - Emit an assignment of @p Value to @p Symbol.
    ///
    /// This corresponds to an assembler statement such as:
    ///  symbol = value
    ///
    /// The assignment generates no code, but has the side effect of binding the
    /// value in the current context. For the assembly streamer, this prints the
    /// binding into the .s file.
    ///
    /// @param Symbol - The symbol being assigned to.
    /// @param Value - The value for the symbol.
    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) = 0;

    /// EmitSymbolAttribute - Add the given @p Attribute to @p Symbol.
    virtual void EmitSymbolAttribute(MCSymbol *Symbol,
                                     MCSymbolAttr Attribute) = 0;

    /// EmitSymbolDesc - Set the @p DescValue for the @p Symbol.
    ///
    /// @param Symbol - The symbol to have its n_desc field set.
    /// @param DescValue - The value to set into the n_desc field.
    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) = 0;

    /// BeginCOFFSymbolDef - Start emitting COFF symbol definition
    ///
    /// @param Symbol - The symbol to have its External & Type fields set.
    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) = 0;

    /// EmitCOFFSymbolStorageClass - Emit the storage class of the symbol.
    ///
    /// @param StorageClass - The storage class the symbol should have.
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) = 0;

    /// EmitCOFFSymbolType - Emit the type of the symbol.
    ///
    /// @param Type - A COFF type identifier (see COFF::SymbolType in X86COFF.h)
    virtual void EmitCOFFSymbolType(int Type) = 0;

    /// EndCOFFSymbolDef - Marks the end of the symbol definition.
    virtual void EndCOFFSymbolDef() = 0;

    /// EmitELFSize - Emit an ELF .size directive.
    ///
    /// This corresponds to an assembler statement such as:
    ///  .size symbol, expression
    ///
    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) = 0;
    
    /// EmitCommonSymbol - Emit a common symbol.
    ///
    /// @param Symbol - The common symbol to emit.
    /// @param Size - The size of the common symbol.
    /// @param ByteAlignment - The alignment of the symbol if
    /// non-zero. This must be a power of 2.
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) = 0;

    /// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
    ///
    /// @param Symbol - The common symbol to emit.
    /// @param Size - The size of the common symbol.
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) = 0;
    
    /// EmitZerofill - Emit a the zerofill section and an option symbol.
    ///
    /// @param Section - The zerofill section to create and or to put the symbol
    /// @param Symbol - The zerofill symbol to emit, if non-NULL.
    /// @param Size - The size of the zerofill symbol.
    /// @param ByteAlignment - The alignment of the zerofill symbol if
    /// non-zero. This must be a power of 2 on some targets.
    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                              unsigned Size = 0,unsigned ByteAlignment = 0) = 0;

    /// EmitTBSSSymbol - Emit a thread local bss (.tbss) symbol.
    ///
    /// @param Section - The thread local common section.
    /// @param Symbol - The thread local common symbol to emit.
    /// @param Size - The size of the symbol.
    /// @param ByteAlignment - The alignment of the thread local common symbol
    /// if non-zero.  This must be a power of 2 on some targets.
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment = 0) = 0;
    /// @}
    /// @name Generating Data
    /// @{

    /// EmitBytes - Emit the bytes in \arg Data into the output.
    ///
    /// This is used to implement assembler directives such as .byte, .ascii,
    /// etc.
    virtual void EmitBytes(StringRef Data, unsigned AddrSpace) = 0;

    /// EmitValue - Emit the expression @p Value into the output as a native
    /// integer of the given @p Size bytes.
    ///
    /// This is used to implement assembler directives such as .word, .quad,
    /// etc.
    ///
    /// @param Value - The value to emit.
    /// @param Size - The size of the integer (in bytes) to emit. This must
    /// match a native machine width.
    virtual void EmitValue(const MCExpr *Value, unsigned Size,
                           unsigned AddrSpace) = 0;

    /// EmitIntValue - Special case of EmitValue that avoids the client having
    /// to pass in a MCExpr for constant integers.
    virtual void EmitIntValue(uint64_t Value, unsigned Size,unsigned AddrSpace);

    /// EmitSymbolValue - Special case of EmitValue that avoids the client
    /// having to pass in a MCExpr for MCSymbols.
    virtual void EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                                 unsigned AddrSpace);
    
    /// EmitGPRel32Value - Emit the expression @p Value into the output as a
    /// gprel32 (32-bit GP relative) value.
    ///
    /// This is used to implement assembler directives such as .gprel32 on
    /// targets that support them.
    virtual void EmitGPRel32Value(const MCExpr *Value) = 0;
    
    /// EmitFill - Emit NumBytes bytes worth of the value specified by
    /// FillValue.  This implements directives such as '.space'.
    virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                          unsigned AddrSpace);
    
    /// EmitZeros - Emit NumBytes worth of zeros.  This is a convenience
    /// function that just wraps EmitFill.
    void EmitZeros(uint64_t NumBytes, unsigned AddrSpace) {
      EmitFill(NumBytes, 0, AddrSpace);
    }

    
    /// EmitValueToAlignment - Emit some number of copies of @p Value until
    /// the byte alignment @p ByteAlignment is reached.
    ///
    /// If the number of bytes need to emit for the alignment is not a multiple
    /// of @p ValueSize, then the contents of the emitted fill bytes is
    /// undefined.
    ///
    /// This used to implement the .align assembler directive.
    ///
    /// @param ByteAlignment - The alignment to reach. This must be a power of
    /// two on some targets.
    /// @param Value - The value to use when filling bytes.
    /// @param ValueSize - The size of the integer (in bytes) to emit for
    /// @p Value. This must match a native machine width.
    /// @param MaxBytesToEmit - The maximum numbers of bytes to emit, or 0. If
    /// the alignment cannot be reached in this many bytes, no bytes are
    /// emitted.
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0) = 0;

    /// EmitCodeAlignment - Emit nops until the byte alignment @p ByteAlignment
    /// is reached.
    ///
    /// This used to align code where the alignment bytes may be executed.  This
    /// can emit different bytes for different sizes to optimize execution.
    ///
    /// @param ByteAlignment - The alignment to reach. This must be a power of
    /// two on some targets.
    /// @param MaxBytesToEmit - The maximum numbers of bytes to emit, or 0. If
    /// the alignment cannot be reached in this many bytes, no bytes are
    /// emitted.
    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit = 0) = 0;

    /// EmitValueToOffset - Emit some number of copies of @p Value until the
    /// byte offset @p Offset is reached.
    ///
    /// This is used to implement assembler directives such as .org.
    ///
    /// @param Offset - The offset to reach. This may be an expression, but the
    /// expression must be associated with the current section.
    /// @param Value - The value to use when filling bytes.
    virtual void EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value = 0) = 0;
    
    /// @}
    
    /// EmitFileDirective - Switch to a new logical file.  This is used to
    /// implement the '.file "foo.c"' assembler directive.
    virtual void EmitFileDirective(StringRef Filename) = 0;
    
    /// EmitDwarfFileDirective - Associate a filename with a specified logical
    /// file number.  This implements the DWARF2 '.file 4 "foo.c"' assembler
    /// directive.
    virtual void EmitDwarfFileDirective(unsigned FileNo,StringRef Filename) = 0;

    /// EmitInstruction - Emit the given @p Instruction into the current
    /// section.
    virtual void EmitInstruction(const MCInst &Inst) = 0;

    /// EmitRawText - If this file is backed by a assembly streamer, this dumps
    /// the specified string in the output .s file.  This capability is
    /// indicated by the hasRawTextSupport() predicate.  By default this aborts.
    virtual void EmitRawText(StringRef String);
    void EmitRawText(const Twine &String);
    
    /// Finish - Finish emission of machine code and flush any output.
    virtual void Finish() = 0;
  };

  /// createNullStreamer - Create a dummy machine code streamer, which does
  /// nothing. This is useful for timing the assembler front end.
  MCStreamer *createNullStreamer(MCContext &Ctx);

  /// createAsmStreamer - Create a machine code streamer which will print out
  /// assembly for the native target, suitable for compiling with a native
  /// assembler.
  ///
  /// \param InstPrint - If given, the instruction printer to use. If not given
  /// the MCInst representation will be printed.  This method takes ownership of
  /// InstPrint.
  ///
  /// \param CE - If given, a code emitter to use to show the instruction
  /// encoding inline with the assembly.
  ///
  /// \param ShowInst - Whether to show the MCInst representation inline with
  /// the assembly.
  MCStreamer *createAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                bool isLittleEndian, bool isVerboseAsm,
                                MCInstPrinter *InstPrint = 0,
                                MCCodeEmitter *CE = 0,
                                bool ShowInst = false);

  /// createMachOStream - Create a machine code streamer which will generative
  /// Mach-O format object files.
  MCStreamer *createMachOStreamer(MCContext &Ctx, TargetAsmBackend &TAB,
                                  raw_ostream &OS, MCCodeEmitter *CE,
                                  bool RelaxAll = false);

} // end namespace llvm

#endif
