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
  class MachineFunction;
  class DebugLoc;
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

  class formatted_raw_ostream;

  class PIC16DbgInfo {
    formatted_raw_ostream &O;
    const TargetAsmInfo *TAI;
    std::string CurFile;
    unsigned CurLine;

    // EmitDebugDirectives is set if debug information is available. Default
    // value for it is false.
    bool EmitDebugDirectives;

  public:
    PIC16DbgInfo(formatted_raw_ostream &o, const TargetAsmInfo *T)
      : O(o), TAI(T) {
      CurFile = "";
      CurLine = 0;
      EmitDebugDirectives = false; 
    }

    void BeginModule (Module &M);
    void BeginFunction (const MachineFunction &MF);
    void ChangeDebugLoc (const MachineFunction &MF, const DebugLoc &DL,
                         bool IsInBeginFunction = false);
    void EndFunction (const MachineFunction &MF);
    void EndModule (Module &M);


    private:
    void SwitchToCU (GlobalVariable *CU);
    void SwitchToLine (unsigned Line, bool IsInBeginFunction = false);

    void PopulateDebugInfo (DIType Ty, unsigned short &TypeNo, bool &HasAux,
                           int Aux[], std::string &TypeName);
    void PopulateBasicTypeInfo (DIType Ty, unsigned short &TypeNo);
    void PopulateDerivedTypeInfo (DIType Ty, unsigned short &TypeNo, 
                                  bool &HasAux, int Aux[],
                                  std::string &TypeName);

    void PopulateCompositeTypeInfo (DIType Ty, unsigned short &TypeNo,
                                    bool &HasAux, int Aux[],
                                    std::string &TypeName);
    void PopulateArrayTypeInfo (DIType Ty, unsigned short &TypeNo,
                                bool &HasAux, int Aux[],
                                std::string &TypeName);

    void PopulateStructOrUnionTypeInfo (DIType Ty, unsigned short &TypeNo,
                                        bool &HasAux, int Aux[],
                                        std::string &TypeName);
    void PopulateEnumTypeInfo (DIType Ty, unsigned short &TypeNo);

    unsigned GetTypeDebugNumber(std::string &Type);
    short getStorageClass(DIGlobalVariable DIGV);
    void EmitFunctBeginDI(const Function *F);
    void EmitCompositeTypeDecls(Module &M);
    void EmitCompositeTypeElements (DICompositeType CTy, unsigned Suffix);
    void EmitFunctEndDI(const Function *F, unsigned Line);
    void EmitAuxEntry(const std::string VarName, int Aux[], 
                      int num = PIC16Dbg::AuxSize, std::string TagName = "");
    inline void EmitSymbol(std::string Name, short Class, 
                           unsigned short Type = PIC16Dbg::T_NULL, 
                           unsigned long Value = 0);
    void EmitVarDebugInfo(Module &M);
    void EmitEOF();
  };
} // end namespace llvm;
#endif
