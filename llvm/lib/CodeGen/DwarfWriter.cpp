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

#include "llvm/CodeGen/DwarfWriter.h"

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>

using namespace llvm;

static cl::opt<bool>
DwarfVerbose("dwarf-verbose", cl::Hidden,
                                cl::desc("Add comments to dwarf directives."));

/// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
/// unsigned leb128 value.  Comment is added to the end of the directive if
/// DwarfVerbose is true (should not contain any newlines.)
///
void DwarfWriter::EmitULEB128Bytes(unsigned Value, const char *Comment) const {
  if (hasLEB128) {
    O << "\t.uleb128\t"
      << Value;
  } else {
    O << Asm->Data8bitsDirective;
    EmitULEB128(Value);
  }
  if (DwarfVerbose) {
    O << "\t"
      << Asm->CommentString
      << " "
      << Comment
      << " "
      << Value;
  }
  O << "\n";
}

/// EmitSLEB128Bytes - Emit an assembler byte data directive to compose a
/// signed leb128 value.  Comment is added to the end of the directive if
/// DwarfVerbose is true (should not contain any newlines.)
///
void DwarfWriter::EmitSLEB128Bytes(int Value, const char *Comment) const {
  if (hasLEB128) {
    O << "\t.sleb128\t"
      << Value;
  } else {
    O << Asm->Data8bitsDirective;
    EmitSLEB128(Value);
  }
  if (DwarfVerbose) {
    O << "\t"
      << Asm->CommentString
      << " "
      << Comment
      << " "
      << Value;
  }
  O << "\n";
}

/// EmitHex - Emit a hexidecimal string to the output stream.
///
void DwarfWriter::EmitHex(unsigned Value) const {
  O << "0x"
    << std::hex
    << Value
    << std::dec;
}

/// EmitComment - Emit a simple string comment.
///
void DwarfWriter::EmitComment(const char *Comment) const {
  O << "\t"
    << Asm->CommentString
    << " "
    << Comment
    << "\n";
}

/// EmitULEB128 - Emit a series of hexidecimal values (separated by commas)
/// representing an unsigned leb128 value.
///
void DwarfWriter::EmitULEB128(unsigned Value) const {
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    if (Value) Byte |= 0x80;
    EmitHex(Byte);
    if (Value) O << ", ";
  } while (Value);
}

/// EmitSLEB128 - Emit a series of hexidecimal values (separated by commas)
/// representing a signed leb128 value.
///
void DwarfWriter::EmitSLEB128(int Value) const {
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    EmitHex(Byte);
    if (IsMore) O << ", ";
  } while (IsMore);
}

/// EmitLabelName - Emit label name for internal use by dwarf.
///
void DwarfWriter::EmitLabelName(const char *Tag, int Num) const {
  O << Asm->PrivateGlobalPrefix
    << "debug_"
    << Tag
    << Num;
}

/// EmitLabel - Emit location label for internal use by dwarf.
///
void DwarfWriter::EmitLabel(const char *Tag, int Num) const {
  EmitLabelName(Tag, Num);
  O << ":\n";
}

/// EmitInitial -Emit initial dwarf declarations.
///
void DwarfWriter::EmitInitial() const {
  // Dwarf section's base addresses.
  Asm->SwitchSection(DwarfAbbrevSection, 0);
  EmitLabel("abbrev", 0);
  Asm->SwitchSection(DwarfInfoSection, 0);
  EmitLabel("info", 0);
  Asm->SwitchSection(DwarfLineSection, 0);
  EmitLabel("line", 0);
}

/// ShouldEmitDwarf - Determine if dwarf declarations should be made.
///
bool DwarfWriter::ShouldEmitDwarf() {
  // Check if debug info is present.
  if (!DebugInfo || !DebugInfo->hasInfo()) return false;
  
  // Make sure initial declarations are made.
  if (!didInitial) {
    EmitInitial();
    didInitial = true;
  }
  
  // Okay to emit.
  return true;
}

/// BeginModule - Emit all dwarf sections that should come prior to the content.
///
void DwarfWriter::BeginModule() {
  if (!ShouldEmitDwarf()) return;
  EmitComment("Dwarf Begin Module");
}

/// EndModule - Emit all dwarf sections that should come after the content.
///
void DwarfWriter::EndModule() {
  if (!ShouldEmitDwarf()) return;
  EmitComment("Dwarf End Module");
  // Print out dwarf file info
  std::vector<std::string> Sources = DebugInfo->getSourceFiles();
  for (unsigned i = 0, N = Sources.size(); i < N; i++) {
    O << "\t; .file\t" << (i + 1) << "," << "\"" << Sources[i]  << "\"" << "\n";
  }
}

/// BeginFunction - Emit pre-function debug information.
///
void DwarfWriter::BeginFunction() {
  if (!ShouldEmitDwarf()) return;
  EmitComment("Dwarf Begin Function");
}

/// EndFunction - Emit post-function debug information.
///
void DwarfWriter::EndFunction() {
  if (!ShouldEmitDwarf()) return;
  EmitComment("Dwarf End Function");
}
