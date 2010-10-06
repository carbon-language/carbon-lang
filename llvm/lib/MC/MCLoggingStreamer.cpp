//===- lib/MC/MCLoggingStreamer.cpp - API Logging Streamer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

class MCLoggingStreamer : public MCStreamer {
  llvm::OwningPtr<MCStreamer> Child;
  
  raw_ostream &OS;

public:
  MCLoggingStreamer(MCStreamer *_Child, raw_ostream &_OS)
    : MCStreamer(_Child->getContext()), Child(_Child), OS(_OS) {}

  void LogCall(const char *Function) {
    OS << Function << "\n";
  }

  void LogCall(const char *Function, const Twine &Message) {
    OS << Function << ": " << Message << "\n";
  }

  virtual bool isVerboseAsm() const { return Child->isVerboseAsm(); }
  
  virtual bool hasRawTextSupport() const { return Child->hasRawTextSupport(); }

  virtual raw_ostream &GetCommentOS() { return Child->GetCommentOS(); }

  virtual void AddComment(const Twine &T) {
    LogCall("AddComment", T);
    return Child->AddComment(T);
  }

  virtual void AddBlankLine() {
    LogCall("AddBlankLine");
    return Child->AddBlankLine();
  }

  virtual void SwitchSection(const MCSection *Section) {
    CurSection = Section;
    LogCall("SwitchSection");
    return Child->SwitchSection(Section);
  }

  virtual void InitSections() {
    LogCall("InitSections");
    return Child->InitSections();
  }

  virtual void EmitLabel(MCSymbol *Symbol) {
    LogCall("EmitLabel");
    return Child->EmitLabel(Symbol);
  }

  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {
    LogCall("EmitAssemblerFlag");
    return Child->EmitAssemblerFlag(Flag);
  }

  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
    LogCall("EmitAssignment");
    return Child->EmitAssignment(Symbol, Value);
  }

  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) {
    LogCall("EmitSymbolAttribute");
    return Child->EmitSymbolAttribute(Symbol, Attribute);
  }

  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
    LogCall("EmitSymbolDesc");
    return Child->EmitSymbolDesc(Symbol, DescValue);
  }

  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {
    LogCall("BeginCOFFSymbolDef");
    return Child->BeginCOFFSymbolDef(Symbol);
  }

  virtual void EmitCOFFSymbolStorageClass(int StorageClass) {
    LogCall("EmitCOFFSymbolStorageClass");
    return Child->EmitCOFFSymbolStorageClass(StorageClass);
  }

  virtual void EmitCOFFSymbolType(int Type) {
    LogCall("EmitCOFFSymbolType");
    return Child->EmitCOFFSymbolType(Type);
  }

  virtual void EndCOFFSymbolDef() {
    LogCall("EndCOFFSymbolDef");
    return Child->EndCOFFSymbolDef();
  }

  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
    LogCall("EmitELFSize");
    return Child->EmitELFSize(Symbol, Value);
  }

  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment) {
    LogCall("EmitCommonSymbol");
    return Child->EmitCommonSymbol(Symbol, Size, ByteAlignment);
  }

  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
    LogCall("EmitLocalCommonSymbol");
    return Child->EmitLocalCommonSymbol(Symbol, Size);
  }
  
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0) {
    LogCall("EmitZerofill");
    return Child->EmitZerofill(Section, Symbol, Size, ByteAlignment);
  }

  virtual void EmitTBSSSymbol (const MCSection *Section, MCSymbol *Symbol,
                               uint64_t Size, unsigned ByteAlignment = 0) {
    LogCall("EmitTBSSSymbol");
    return Child->EmitTBSSSymbol(Section, Symbol, Size, ByteAlignment);
  }

  virtual void EmitBytes(StringRef Data, unsigned AddrSpace) {
    LogCall("EmitBytes");
    return Child->EmitBytes(Data, AddrSpace);
  }

  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace){
    LogCall("EmitValue");
    return Child->EmitValue(Value, Size, AddrSpace);
  }

  virtual void EmitIntValue(uint64_t Value, unsigned Size, unsigned AddrSpace) {
    LogCall("EmitIntValue");
    return Child->EmitIntValue(Value, Size, AddrSpace);
  }

  virtual void EmitGPRel32Value(const MCExpr *Value) {
    LogCall("EmitGPRel32Value");
    return Child->EmitGPRel32Value(Value);
  }

  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                        unsigned AddrSpace) {
    LogCall("EmitFill");
    return Child->EmitFill(NumBytes, FillValue, AddrSpace);
  }

  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0) {
    LogCall("EmitValueToAlignment");
    return Child->EmitValueToAlignment(ByteAlignment, Value,
                                       ValueSize, MaxBytesToEmit);
  }

  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0) {
    LogCall("EmitCodeAlignment");
    return Child->EmitCodeAlignment(ByteAlignment, MaxBytesToEmit);
  }

  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0) {
    LogCall("EmitValueToOffset");
    return Child->EmitValueToOffset(Offset, Value);
  }

  virtual void EmitFileDirective(StringRef Filename) {
    LogCall("EmitFileDirective", "FileName:" + Filename);
    return Child->EmitFileDirective(Filename);
  }

  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename) {
    LogCall("EmitDwarfFileDirective",
            "FileNo:" + Twine(FileNo) + " Filename:" + Filename);
    return Child->EmitDwarfFileDirective(FileNo, Filename);
  }

  virtual void EmitInstruction(const MCInst &Inst) {
    LogCall("EmitInstruction");
    return Child->EmitInstruction(Inst);
  }

  virtual void EmitRawText(StringRef String) {
    LogCall("EmitRawText", "\"" + String + "\"");
    return Child->EmitRawText(String);
  }

  virtual void Finish() {
    LogCall("Finish");
    return Child->Finish();
  }

};

} // end anonymous namespace.

MCStreamer *llvm::createLoggingStreamer(MCStreamer *Child, raw_ostream &OS) {
  return new MCLoggingStreamer(Child, OS);
}
