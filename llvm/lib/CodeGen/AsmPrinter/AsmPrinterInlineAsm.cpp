//===-- AsmPrinterInlineAsm.cpp - AsmPrinter Inline Asm Handling ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the inline assembler pieces of the AsmPrinter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCParser/AsmParser.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// EmitInlineAsm - Emit a blob of inline asm to the output streamer.
void AsmPrinter::EmitInlineAsm(StringRef Str, unsigned LocCookie) const {
  assert(!Str.empty() && "Can't emit empty inline asm block");
  
  // Remember if the buffer is nul terminated or not so we can avoid a copy.
  bool isNullTerminated = Str.back() == 0;
  if (isNullTerminated)
    Str = Str.substr(0, Str.size()-1);
  
  // If the output streamer is actually a .s file, just emit the blob textually.
  // This is useful in case the asm parser doesn't handle something but the
  // system assembler does.
  if (OutStreamer.hasRawTextSupport()) {
    OutStreamer.EmitRawText(Str);
    return;
  }
  
  SourceMgr SrcMgr;
  
  // If the current LLVMContext has an inline asm handler, set it in SourceMgr.
  LLVMContext &LLVMCtx = MMI->getModule()->getContext();
  bool HasDiagHandler = false;
  if (void *DiagHandler = LLVMCtx.getInlineAsmDiagnosticHandler()) {
    SrcMgr.setDiagHandler((SourceMgr::DiagHandlerTy)(intptr_t)DiagHandler,
                          LLVMCtx.getInlineAsmDiagnosticContext(), LocCookie);
    HasDiagHandler = true;
  }
  
  MemoryBuffer *Buffer;
  if (isNullTerminated)
    Buffer = MemoryBuffer::getMemBuffer(Str, "<inline asm>");
  else
    Buffer = MemoryBuffer::getMemBufferCopy(Str, "<inline asm>");

  // Tell SrcMgr about this buffer, it takes ownership of the buffer.
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());
  
  AsmParser Parser(SrcMgr, OutContext, OutStreamer, *MAI);
  OwningPtr<TargetAsmParser> TAP(TM.getTarget().createAsmParser(Parser));
  if (!TAP)
    report_fatal_error("Inline asm not supported by this streamer because"
                       " we don't have an asm parser for this target\n");
  Parser.setTargetParser(*TAP.get());

  // Don't implicitly switch to the text section before the asm.
  int Res = Parser.Run(/*NoInitialTextSection*/ true,
                       /*NoFinalize*/ true);
  if (Res && !HasDiagHandler)
    report_fatal_error("Error parsing inline asm\n");
}


/// EmitInlineAsm - This method formats and emits the specified machine
/// instruction that is an inline asm.
void AsmPrinter::EmitInlineAsm(const MachineInstr *MI) const {
  assert(MI->isInlineAsm() && "printInlineAsm only works on inline asms");
  
  unsigned NumOperands = MI->getNumOperands();
  
  // Count the number of register definitions to find the asm string.
  unsigned NumDefs = 0;
  for (; MI->getOperand(NumDefs).isReg() && MI->getOperand(NumDefs).isDef();
       ++NumDefs)
    assert(NumDefs != NumOperands-2 && "No asm string?");
  
  assert(MI->getOperand(NumDefs).isSymbol() && "No asm string?");

  // Disassemble the AsmStr, printing out the literal pieces, the operands, etc.
  const char *AsmStr = MI->getOperand(NumDefs).getSymbolName();

  // If this asmstr is empty, just print the #APP/#NOAPP markers.
  // These are useful to see where empty asm's wound up.
  if (AsmStr[0] == 0) {
    // Don't emit the comments if writing to a .o file.
    if (!OutStreamer.hasRawTextSupport()) return;

    OutStreamer.EmitRawText(Twine("\t")+MAI->getCommentString()+
                            MAI->getInlineAsmStart());
    OutStreamer.EmitRawText(Twine("\t")+MAI->getCommentString()+
                            MAI->getInlineAsmEnd());
    return;
  }

  // Emit the #APP start marker.  This has to happen even if verbose-asm isn't
  // enabled, so we use EmitRawText.
  if (OutStreamer.hasRawTextSupport())
    OutStreamer.EmitRawText(Twine("\t")+MAI->getCommentString()+
                            MAI->getInlineAsmStart());

  // Get the !srcloc metadata node if we have it, and decode the loc cookie from
  // it.
  unsigned LocCookie = 0;
  for (unsigned i = MI->getNumOperands(); i != 0; --i) {
    if (MI->getOperand(i-1).isMetadata())
      if (const MDNode *SrcLoc = MI->getOperand(i-1).getMetadata())
        if (SrcLoc->getNumOperands() != 0)
          if (const ConstantInt *CI =
              dyn_cast<ConstantInt>(SrcLoc->getOperand(0))) {
            LocCookie = CI->getZExtValue();
            break;
          }
  }
  
  // Emit the inline asm to a temporary string so we can emit it through
  // EmitInlineAsm.
  SmallString<256> StringData;
  raw_svector_ostream OS(StringData);
  
  OS << '\t';

  // The variant of the current asmprinter.
  int AsmPrinterVariant = MAI->getAssemblerDialect();

  int CurVariant = -1;            // The number of the {.|.|.} region we are in.
  const char *LastEmitted = AsmStr; // One past the last character emitted.
  
  while (*LastEmitted) {
    switch (*LastEmitted) {
    default: {
      // Not a special case, emit the string section literally.
      const char *LiteralEnd = LastEmitted+1;
      while (*LiteralEnd && *LiteralEnd != '{' && *LiteralEnd != '|' &&
             *LiteralEnd != '}' && *LiteralEnd != '$' && *LiteralEnd != '\n')
        ++LiteralEnd;
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
        OS.write(LastEmitted, LiteralEnd-LastEmitted);
      LastEmitted = LiteralEnd;
      break;
    }
    case '\n':
      ++LastEmitted;   // Consume newline character.
      OS << '\n';      // Indent code with newline.
      break;
    case '$': {
      ++LastEmitted;   // Consume '$' character.
      bool Done = true;

      // Handle escapes.
      switch (*LastEmitted) {
      default: Done = false; break;
      case '$':     // $$ -> $
        if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
          OS << '$';
        ++LastEmitted;  // Consume second '$' character.
        break;
      case '(':             // $( -> same as GCC's { character.
        ++LastEmitted;      // Consume '(' character.
        if (CurVariant != -1)
          report_fatal_error("Nested variants found in inline asm string: '" +
                             Twine(AsmStr) + "'");
        CurVariant = 0;     // We're in the first variant now.
        break;
      case '|':
        ++LastEmitted;  // consume '|' character.
        if (CurVariant == -1)
          OS << '|';       // this is gcc's behavior for | outside a variant
        else
          ++CurVariant;   // We're in the next variant.
        break;
      case ')':         // $) -> same as GCC's } char.
        ++LastEmitted;  // consume ')' character.
        if (CurVariant == -1)
          OS << '}';     // this is gcc's behavior for } outside a variant
        else 
          CurVariant = -1;
        break;
      }
      if (Done) break;
      
      bool HasCurlyBraces = false;
      if (*LastEmitted == '{') {     // ${variable}
        ++LastEmitted;               // Consume '{' character.
        HasCurlyBraces = true;
      }
      
      // If we have ${:foo}, then this is not a real operand reference, it is a
      // "magic" string reference, just like in .td files.  Arrange to call
      // PrintSpecial.
      if (HasCurlyBraces && *LastEmitted == ':') {
        ++LastEmitted;
        const char *StrStart = LastEmitted;
        const char *StrEnd = strchr(StrStart, '}');
        if (StrEnd == 0)
          report_fatal_error("Unterminated ${:foo} operand in inline asm"
                             " string: '" + Twine(AsmStr) + "'");
        
        std::string Val(StrStart, StrEnd);
        PrintSpecial(MI, OS, Val.c_str());
        LastEmitted = StrEnd+1;
        break;
      }
            
      const char *IDStart = LastEmitted;
      const char *IDEnd = IDStart;
      while (*IDEnd >= '0' && *IDEnd <= '9') ++IDEnd;      
      
      unsigned Val;
      if (StringRef(IDStart, IDEnd-IDStart).getAsInteger(10, Val))
        report_fatal_error("Bad $ operand number in inline asm string: '" +
                           Twine(AsmStr) + "'");
      LastEmitted = IDEnd;
      
      char Modifier[2] = { 0, 0 };
      
      if (HasCurlyBraces) {
        // If we have curly braces, check for a modifier character.  This
        // supports syntax like ${0:u}, which correspond to "%u0" in GCC asm.
        if (*LastEmitted == ':') {
          ++LastEmitted;    // Consume ':' character.
          if (*LastEmitted == 0)
            report_fatal_error("Bad ${:} expression in inline asm string: '" +
                               Twine(AsmStr) + "'");
          
          Modifier[0] = *LastEmitted;
          ++LastEmitted;    // Consume modifier character.
        }
        
        if (*LastEmitted != '}')
          report_fatal_error("Bad ${} expression in inline asm string: '" +
                             Twine(AsmStr) + "'");
        ++LastEmitted;    // Consume '}' character.
      }
      
      if (Val >= NumOperands-1)
        report_fatal_error("Invalid $ operand number in inline asm string: '" +
                           Twine(AsmStr) + "'");
      
      // Okay, we finally have a value number.  Ask the target to print this
      // operand!
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant) {
        unsigned OpNo = 1;

        bool Error = false;

        // Scan to find the machine operand number for the operand.
        for (; Val; --Val) {
          if (OpNo >= MI->getNumOperands()) break;
          unsigned OpFlags = MI->getOperand(OpNo).getImm();
          OpNo += InlineAsm::getNumOperandRegisters(OpFlags) + 1;
        }

        if (OpNo >= MI->getNumOperands()) {
          Error = true;
        } else {
          unsigned OpFlags = MI->getOperand(OpNo).getImm();
          ++OpNo;  // Skip over the ID number.

          if (Modifier[0] == 'l')  // labels are target independent
            // FIXME: What if the operand isn't an MBB, report error?
            OS << *MI->getOperand(OpNo).getMBB()->getSymbol();
          else {
            AsmPrinter *AP = const_cast<AsmPrinter*>(this);
            if (InlineAsm::isMemKind(OpFlags)) {
              Error = AP->PrintAsmMemoryOperand(MI, OpNo, AsmPrinterVariant,
                                                Modifier[0] ? Modifier : 0,
                                                OS);
            } else {
              Error = AP->PrintAsmOperand(MI, OpNo, AsmPrinterVariant,
                                          Modifier[0] ? Modifier : 0, OS);
            }
          }
        }
        if (Error) {
          std::string msg;
          raw_string_ostream Msg(msg);
          Msg << "invalid operand in inline asm: '" << AsmStr << "'";
          MMI->getModule()->getContext().emitError(LocCookie, Msg.str());
        }
      }
      break;
    }
    }
  }
  OS << '\n' << (char)0;  // null terminate string.
  EmitInlineAsm(OS.str(), LocCookie);
  
  // Emit the #NOAPP end marker.  This has to happen even if verbose-asm isn't
  // enabled, so we use EmitRawText.
  if (OutStreamer.hasRawTextSupport())
    OutStreamer.EmitRawText(Twine("\t")+MAI->getCommentString()+
                            MAI->getInlineAsmEnd());
}


/// PrintSpecial - Print information related to the specified machine instr
/// that is independent of the operand, and may be independent of the instr
/// itself.  This can be useful for portably encoding the comment character
/// or other bits of target-specific knowledge into the asmstrings.  The
/// syntax used is ${:comment}.  Targets can override this to add support
/// for their own strange codes.
void AsmPrinter::PrintSpecial(const MachineInstr *MI, raw_ostream &OS,
                              const char *Code) const {
  if (!strcmp(Code, "private")) {
    OS << MAI->getPrivateGlobalPrefix();
  } else if (!strcmp(Code, "comment")) {
    OS << MAI->getCommentString();
  } else if (!strcmp(Code, "uid")) {
    // Comparing the address of MI isn't sufficient, because machineinstrs may
    // be allocated to the same address across functions.
    
    // If this is a new LastFn instruction, bump the counter.
    if (LastMI != MI || LastFn != getFunctionNumber()) {
      ++Counter;
      LastMI = MI;
      LastFn = getFunctionNumber();
    }
    OS << Counter;
  } else {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Unknown special formatter '" << Code
         << "' for machine instr: " << *MI;
    report_fatal_error(Msg.str());
  }    
}

/// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
/// instruction, using the specified assembler variant.  Targets should
/// override this to format as appropriate.
bool AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode,
                                 raw_ostream &O) {
  // Target doesn't support this yet!
  return true;
}

bool AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode, raw_ostream &O) {
  // Target doesn't support this yet!
  return true;
}

