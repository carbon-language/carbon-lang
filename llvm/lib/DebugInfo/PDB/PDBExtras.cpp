//===- PDBExtras.cpp - helper functions and classes for PDBs -----*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

#define CASE_OUTPUT_ENUM_CLASS_STR(Class, Value, Str, Stream)                  \
  case Class::Value:                                                           \
    Stream << Str;                                                             \
    break;

#define CASE_OUTPUT_ENUM_CLASS_NAME(Class, Value, Stream)                      \
  CASE_OUTPUT_ENUM_CLASS_STR(Class, Value, #Value, Stream)

raw_ostream &llvm::operator<<(raw_ostream &OS, const stream_indent &Indent) {
  OS.indent(Indent.Width);
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_RegisterId &Reg) {
  switch (Reg) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, AL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, AH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, AX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, SP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, SI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EAX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ECX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EDX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EBX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ESP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EBP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ESI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EDI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ES, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, SS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, FS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, GS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, IP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RAX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RBX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RCX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RDX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RSI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RDI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RBP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RSP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R8, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R9, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R10, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R11, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R12, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R13, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R14, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R15, OS)
  default:
    OS << "Unknown";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_LocType &Loc) {
  switch (Loc) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, Static, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, TLS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, RegRel, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, ThisRel, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, Enregistered, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, BitField, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, Slot, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, IlRel, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, MetaData, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_LocType, Constant, OS)
  default:
    OS << "Unknown";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_ThunkOrdinal &Thunk) {
  switch (Thunk) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, BranchIsland, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, Pcode, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, Standard, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, ThisAdjustor, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, TrampIncremental, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, UnknownLoad, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, Vcall, OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_Checksum &Checksum) {
  switch (Checksum) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Checksum, None, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Checksum, MD5, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Checksum, SHA1, OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_Lang &Lang) {
  switch (Lang) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, C, OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_Lang, Cpp, "C++", OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Fortran, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Masm, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Pascal, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Basic, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Cobol, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Link, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Cvtres, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Cvtpgd, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, CSharp, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, VB, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, ILAsm, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Java, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, JScript, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, MSIL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, HLSL, OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_SymType &Tag) {
  switch (Tag) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Exe, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Compiland, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, CompilandDetails, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, CompilandEnv, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Function, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Block, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Data, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Annotation, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Label, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, PublicSymbol, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, UDT, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Enum, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FunctionSig, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, PointerType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, ArrayType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, BuiltinType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Typedef, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, BaseClass, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Friend, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FunctionArg, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FuncDebugStart, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FuncDebugEnd, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, UsingNamespace, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, VTableShape, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, VTable, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Custom, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Thunk, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, CustomType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, ManagedType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Dimension, OS)
  default:
    OS << "Unknown";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_UniqueId &Id) {
  static const char *Lookup = "0123456789ABCDEF";

  static_assert(sizeof(PDB_UniqueId) == 16, "Expected 16-byte GUID");
  ArrayRef<uint8_t> GuidBytes(reinterpret_cast<const uint8_t*>(&Id), 16);
  OS << "{";
  for (int i=0; i < 16;) {
    uint8_t Byte = GuidBytes[i];
    uint8_t HighNibble = (Byte >> 4) & 0xF;
    uint8_t LowNibble = Byte & 0xF;
    OS << Lookup[HighNibble] << Lookup[LowNibble];
    ++i;
    if (i>=4 && i<=10 && i%2==0)
      OS << "-";
  }
  OS << "}";
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const VersionInfo &Version) {
  OS << Version.Major << "." << Version.Minor << "." << Version.Build;
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const TagStats &Stats) {
  for (auto Tag : Stats) {
    OS << Tag.first << ":" << Tag.second << " ";
  }
  return OS;
}
