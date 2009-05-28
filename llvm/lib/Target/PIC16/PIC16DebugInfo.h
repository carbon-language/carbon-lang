//===-- PIC16DebugInfo.h - Interfaces for PIC16 Debug Information ============//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the helper functions for representing debug information.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16DBG_H
#define PIC16DBG_H

#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetAsmInfo.h" 
#include <map>

namespace llvm {
  namespace PIC16Dbg {
    enum VarType {
      T_NULL,
      T_VOID,
      T_CHAR,
      T_SHORT,
      T_INT,
      T_LONG,
      T_FLOAT,
      T_DOUBLE,
      T_STRUCT,
      T_UNION,
      T_ENUM,
      T_MOE,
      T_UCHAR,
      T_USHORT,
      T_UINT,
      T_ULONG
    };
    enum DerivedType {
      DT_NONE,
      DT_PTR,
      DT_FCN,
      DT_ARY
    };
    enum TypeSize {
      S_BASIC = 5,
      S_DERIVED = 3
    };
    enum DbgClass {
      C_NULL,
      C_AUTO,
      C_EXT,
      C_STAT,
      C_REG,
      C_EXTDEF,
      C_LABEL,
      C_ULABEL,
      C_MOS,
      C_ARG,
      C_STRTAG,
      C_MOU,
      C_UNTAG,
      C_TPDEF,
      C_USTATIC,
      C_ENTAG,
      C_MOE,
      C_REGPARM,
      C_FIELD,
      C_AUTOARG,
      C_LASTENT,
      C_BLOCK = 100,
      C_FCN,
      C_EOS,
      C_FILE,
      C_LINE,
      C_ALIAS,
      C_HIDDEN,
      C_EOF,
      C_LIST,
      C_SECTION,
      C_EFCN = 255
    };
    enum SymbolSize {
      AuxSize =20
    };
  }

  class raw_ostream;

  class PIC16DbgInfo {
    std::map <std::string, DISubprogram *> FunctNameMap;
    raw_ostream &O;
    const TargetAsmInfo *TAI;
  public:
     PIC16DbgInfo(raw_ostream &o, const TargetAsmInfo *T) : O(o), TAI(T) {}
    ~PIC16DbgInfo();
    void PopulateDebugInfo(DIType Ty, unsigned short &TypeNo, bool &HasAux,
                           int Aux[], std::string &TypeName);
    unsigned GetTypeDebugNumber(std::string &type);
    short getClass(DIGlobalVariable DIGV);
    void PopulateFunctsDI(Module &M);
    DISubprogram *getFunctDI(std::string FunctName);
    void EmitFunctBeginDI(const Function *F);
    void EmitFunctEndDI(const Function *F, unsigned Line);
    void EmitAuxEntry(const std::string VarName, int Aux[], int num);
    inline void EmitSymbol(std::string Name, int Class);
    void EmitVarDebugInfo(Module &M);
    void EmitFileDirective(Module &M);
  };
} // end namespace llvm;
#endif
