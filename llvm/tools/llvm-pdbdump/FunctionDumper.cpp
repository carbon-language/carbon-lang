//===- FunctionDumper.cpp ------------------------------------ *- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FunctionDumper.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace llvm;

namespace {
template <class T>
void dumpClassParentWithScopeOperator(const T &Symbol, llvm::raw_ostream &OS,
                                      llvm::FunctionDumper &Dumper) {
  uint32_t ClassParentId = Symbol.getClassParentId();
  auto ClassParent =
      Symbol.getSession().getConcreteSymbolById<PDBSymbolTypeUDT>(
          ClassParentId);
  if (!ClassParent)
    return;

  OS << ClassParent->getName() << "::";
}
}

FunctionDumper::FunctionDumper() : PDBSymDumper(true) {}

void FunctionDumper::start(const PDBSymbolTypeFunctionSig &Symbol,
                           PointerType Pointer, raw_ostream &OS) {
  auto ReturnType = Symbol.getReturnType();
  ReturnType->dump(OS, 0, *this);
  OS << " ";
  uint32_t ClassParentId = Symbol.getClassParentId();
  auto ClassParent =
      Symbol.getSession().getConcreteSymbolById<PDBSymbolTypeUDT>(
          ClassParentId);

  if (Pointer == PointerType::None) {
    OS << Symbol.getCallingConvention() << " ";
    if (ClassParent)
      OS << "(" << ClassParent->getName() << "::)";
  } else {
    OS << "(" << Symbol.getCallingConvention() << " ";
    if (ClassParent)
      OS << ClassParent->getName() << "::";
    if (Pointer == PointerType::Reference)
      OS << "&";
    else
      OS << "*";
    OS << ")";
  }

  OS << "(";
  if (auto ChildEnum = Symbol.getArguments()) {
    uint32_t Index = 0;
    while (auto Arg = ChildEnum->getNext()) {
      Arg->dump(OS, 0, *this);
      if (++Index < ChildEnum->getChildCount())
        OS << ", ";
    }
  }
  OS << ")";

  if (Symbol.isConstType())
    OS << " const";
  if (Symbol.isVolatileType())
    OS << " volatile";
}

void FunctionDumper::start(const PDBSymbolFunc &Symbol, raw_ostream &OS) {
  if (Symbol.isVirtual() || Symbol.isPureVirtual())
    OS << "virtual ";

  auto Signature = Symbol.getSignature();
  if (!Signature) {
    OS << Symbol.getName();
    return;
  }

  auto ReturnType = Signature->getReturnType();
  ReturnType->dump(OS, 0, *this);

  OS << " " << Signature->getCallingConvention() << " ";
  OS << Symbol.getName();

  OS << "(";
  if (auto ChildEnum = Signature->getArguments()) {
    uint32_t Index = 0;
    while (auto Arg = ChildEnum->getNext()) {
      Arg->dump(OS, 0, *this);
      if (++Index < ChildEnum->getChildCount())
        OS << ", ";
    }
  }
  OS << ")";

  if (Symbol.isConstType())
    OS << " const";
  if (Symbol.isVolatileType())
    OS << " volatile";
  if (Symbol.isPureVirtual())
    OS << " = 0";
}

void FunctionDumper::dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
                          int Indent) {
  uint32_t ElementTypeId = Symbol.getTypeId();
  auto ElementType = Symbol.getSession().getSymbolById(ElementTypeId);
  if (!ElementType)
    return;

  ElementType->dump(OS, 0, *this);
  OS << "[" << Symbol.getLength() << "]";
}

void FunctionDumper::dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                          int Indent) {
  PDB_BuiltinType Type = Symbol.getBuiltinType();
  OS << Type;
  if (Type == PDB_BuiltinType::UInt || Type == PDB_BuiltinType::Int)
    OS << (8 * Symbol.getLength()) << "_t";
}

void FunctionDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                          int Indent) {
  dumpClassParentWithScopeOperator(Symbol, OS, *this);
  OS << Symbol.getName();
}

void FunctionDumper::dump(const PDBSymbolTypeFunctionArg &Symbol,
                          raw_ostream &OS, int Indent) {
  // PDBSymbolTypeFunctionArg is just a shim over the real argument.  Just drill
  // through to the
  // real thing and dump it.
  uint32_t TypeId = Symbol.getTypeId();
  auto Type = Symbol.getSession().getSymbolById(TypeId);
  if (!Type)
    return;
  Type->dump(OS, 0, *this);
}

void FunctionDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                          int Indent) {
  dumpClassParentWithScopeOperator(Symbol, OS, *this);
  OS << Symbol.getName();
}

void FunctionDumper::dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                          int Indent) {
  uint32_t PointeeId = Symbol.getTypeId();
  auto PointeeType = Symbol.getSession().getSymbolById(PointeeId);
  if (!PointeeType)
    return;

  if (auto FuncSig = dyn_cast<PDBSymbolTypeFunctionSig>(PointeeType.get())) {
    FunctionDumper NestedDumper;
    PointerType Pointer =
        Symbol.isReference() ? PointerType::Reference : PointerType::Pointer;
    NestedDumper.start(*FuncSig, Pointer, OS);
  } else {
    if (Symbol.isConstType())
      OS << "const ";
    if (Symbol.isVolatileType())
      OS << "volatile ";
    PointeeType->dump(OS, Indent, *this);
    OS << (Symbol.isReference() ? "&" : "*");
  }
}

void FunctionDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                          int Indent) {
  OS << Symbol.getName();
}
