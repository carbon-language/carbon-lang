//===- MCStreamer.h - High-level Streaming Machine Code Output --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSTREAMER_H
#define LLVM_MC_MCSTREAMER_H

namespace llvm {
  class MCAtom;
  class MCContext;
  class MCImm;
  class MCInst;
  class MCSection;
  class MCSymbol;
  class raw_ostream;

  /// MCStreamer - Streaming machine code generation interface.
  class MCStreamer {
  public:
    enum SymbolAttr {
      Global,
      Weak,
      PrivateExtern
    };

  private:
    MCContext &Context;

    MCStreamer(const MCStreamer&); // DO NOT IMPLEMENT
    MCStreamer &operator=(const MCStreamer&); // DO NOT IMPLEMENT

  public:
    MCStreamer(MCContext &Ctx);
    virtual ~MCStreamer();

    MCContext &getContext() const { return Context; }

    virtual void SwitchSection(MCSection *Sect) = 0;

    virtual void EmitSymbol(MCSymbol *Sym);
    virtual void EmitSymbolAssignment(MCSymbol *Sym, const MCImm &Value) = 0;
    virtual void EmitSymbolAttribute(MCSymbol *Sym, 
                                     SymbolAttr Attr) = 0;

    virtual void EmitBytes(const char *Data, unsigned Length) = 0;
    virtual void EmitValue(const MCImm &Value, unsigned Size) = 0;
    virtual void EmitInstruction(const MCInst &Inst) = 0;
  };

  MCStreamer *createAsmStreamer(MCContext &Ctx, raw_ostream &OS);
  MCStreamer *createMachOStreamer(MCContext &Ctx, raw_ostream &OS);
  MCStreamer *createELFStreamer(MCContext &Ctx, raw_ostream &OS);

} // end namespace llvm

#endif
