//===-- llvm/CodeGen/DwarfWriter.cpp - Dwarf Framework ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//


#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Support/CommandLine.h"


namespace llvm {

static cl::opt<bool>
DwarfVerbose("dwarf-verbose", cl::Hidden,
                                cl::desc("Add comments to dwarf directives."));

/// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
/// unsigned leb128 value.
///
void DwarfWriter::EmitULEB128Bytes(unsigned Value, std::string Comment) {
  if (hasLEB128) {
    O << "\t.uleb128\t"
      << Value;
  } else {
    O << Asm->getData8bitsDirective();
    EmitULEB128(Value);
  }
  if (DwarfVerbose) {
    O << "\t"
      << Asm->getCommentString()
      << " "
      << Comment
      << " "
      << Value;
  }
  O << "\n";
}

/// EmitSLEB128Bytes - Emit an assembler byte data directive to compose a
/// signed leb128 value.
///
void DwarfWriter::EmitSLEB128Bytes(int Value, std::string Comment) {
  if (hasLEB128) {
    O << "\t.sleb128\t"
      << Value;
  } else {
    O << Asm->getData8bitsDirective();
    EmitSLEB128(Value);
  }
  if (DwarfVerbose) {
    O << "\t"
      << Asm->getCommentString()
      << " "
      << Comment
      << " "
      << Value;
  }
  O << "\n";
}

/// BeginModule - Emit all dwarf sections that should come prior to the content.
///
void DwarfWriter::BeginModule() {
  EmitComment("Dwarf Begin Module");
  
  // define base addresses for dwarf sections
  Asm->SwitchSection(DwarfAbbrevSection, 0);
  EmitLabel("abbrev", 0);
  Asm->SwitchSection(DwarfInfoSection, 0);
  EmitLabel("info", 0);
  Asm->SwitchSection(DwarfLineSection, 0);
  EmitLabel("line", 0);
}

/// EndModule - Emit all dwarf sections that should come after the content.
///
void DwarfWriter::EndModule() {
  EmitComment("Dwarf End Module");
  // Print out dwarf file info
  std::vector<std::string> Sources = DebugInfo.getSourceFiles();
  for (unsigned i = 0, N = Sources.size(); i < N; i++) {
    O << "\t; .file\t" << (i + 1) << "," << "\"" << Sources[i]  << "\"" << "\n";
  }
}


/// BeginFunction - Emit pre-function debug information.
///
void DwarfWriter::BeginFunction() {
  EmitComment("Dwarf Begin Function");
}

/// EndFunction - Emit post-function debug information.
///
void DwarfWriter::EndFunction() {
  EmitComment("Dwarf End Function");
}


} // End llvm namespace

