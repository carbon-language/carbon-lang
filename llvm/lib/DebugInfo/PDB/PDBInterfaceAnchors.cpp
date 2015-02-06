//===- PDBInterfaceAnchors.h - defines class anchor funcions ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Class anchors are necessary per the LLVM Coding style guide, to ensure that
// the vtable is only generated in this object file, and not in every object
// file that incldues the corresponding header.
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/IPDBDataStream.h"
#include "llvm/DebugInfo/PDB/IPDBLineNumber.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"

#include "llvm/DebugInfo/PDB/PDBSymbolAnnotation.h"
#include "llvm/DebugInfo/PDB/PDBSymbolBlock.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCustom.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolLabel.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeCustom.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeDimension.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFriend.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeManaged.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTableShape.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUnknown.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUsingNamespace.h"

using namespace llvm;

IPDBSession::~IPDBSession() {}

IPDBDataStream::~IPDBDataStream() {}

IPDBRawSymbol::~IPDBRawSymbol() {}

IPDBSourceFile::~IPDBSourceFile() {}

IPDBLineNumber::~IPDBLineNumber() {}

// All of the concrete symbol types have their methods declared inline through
// the use of a forwarding macro, so the constructor should be declared out of
// line to get the vtable in this file.
#define FORWARD_SYMBOL_CONSTRUCTOR(ClassName)                                  \
  ClassName::ClassName(std::unique_ptr<IPDBRawSymbol> Symbol)                  \
      : PDBSymbol(std::move(Symbol)) {}

FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolAnnotation)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolBlock)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolCompiland)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolCompilandDetails)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolCompilandEnv)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolCustom)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolData)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolExe)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolFunc)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolFuncDebugEnd)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolFuncDebugStart)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolLabel)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolPublicSymbol)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolThunk)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeArray)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeBaseClass)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeBuiltin)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeCustom)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeDimension)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeEnum)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeFriend)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeFunctionArg)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeFunctionSig)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeManaged)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypePointer)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeTypedef)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeUDT)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeVTable)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolTypeVTableShape)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolUnknown)
FORWARD_SYMBOL_CONSTRUCTOR(PDBSymbolUsingNamespace)
