//===-- COFFDumper.cpp - COFF-specific dumper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the COFF-specific dumper for llvm-readobj.
///
//===----------------------------------------------------------------------===//

#include "llvm-readobj.h"
#include "Error.h"
#include "ObjDumper.h"
#include "StreamWriter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Win64EH.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>
#include <time.h>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::Win64EH;

namespace {

class COFFDumper : public ObjDumper {
public:
  COFFDumper(const llvm::object::COFFObjectFile *Obj, StreamWriter& Writer)
    : ObjDumper(Writer)
    , Obj(Obj) {
    cacheRelocations();
  }

  virtual void printFileHeaders() override;
  virtual void printSections() override;
  virtual void printRelocations() override;
  virtual void printSymbols() override;
  virtual void printDynamicSymbols() override;
  virtual void printUnwindInfo() override;

private:
  void printSymbol(const SymbolRef &Sym);
  void printRelocation(const SectionRef &Section, const RelocationRef &Reloc);
  void printDataDirectory(uint32_t Index, const std::string &FieldName);
  void printX64UnwindInfo();

  template <class PEHeader> void printPEHeader(const PEHeader *Hdr);
  void printBaseOfDataField(const pe32_header *Hdr);
  void printBaseOfDataField(const pe32plus_header *Hdr);

  void printRuntimeFunction(const RuntimeFunction& RTF,
                            const coff_section *Section,
                            uint64_t SectionOffset);

  void printUnwindInfo(const Win64EH::UnwindInfo& UI,
                       const coff_section *Section, uint64_t SectionOffset);

  void printUnwindCode(const Win64EH::UnwindInfo &UI, ArrayRef<UnwindCode> UCs);

  void printCodeViewLineTables(const SectionRef &Section);

  void cacheRelocations();

  error_code resolveRelocation(const coff_section *Section, uint64_t Offset,
                               const coff_section *&ReesolvedSection,
                               uint64_t &ResolvedAddress);

  error_code resolveSymbol(const coff_section *Section, uint64_t Offset,
                           SymbolRef &Sym);
  error_code resolveSymbolName(const coff_section *Section, uint64_t Offset,
                               StringRef &Name);
  std::string formatSymbol(const coff_section *Section, uint64_t Offset,
                           uint32_t Disp);

  typedef DenseMap<const coff_section*, std::vector<RelocationRef> > RelocMapTy;

  const llvm::object::COFFObjectFile *Obj;
  RelocMapTy RelocMap;
};

} // namespace


namespace llvm {

error_code createCOFFDumper(const object::ObjectFile *Obj, StreamWriter &Writer,
                            std::unique_ptr<ObjDumper> &Result) {
  const COFFObjectFile *COFFObj = dyn_cast<COFFObjectFile>(Obj);
  if (!COFFObj)
    return readobj_error::unsupported_obj_file_format;

  Result.reset(new COFFDumper(COFFObj, Writer));
  return readobj_error::success;
}

} // namespace llvm


// Returns the name of the unwind code.
static StringRef getUnwindCodeTypeName(uint8_t Code) {
  switch(Code) {
  default: llvm_unreachable("Invalid unwind code");
  case UOP_PushNonVol: return "PUSH_NONVOL";
  case UOP_AllocLarge: return "ALLOC_LARGE";
  case UOP_AllocSmall: return "ALLOC_SMALL";
  case UOP_SetFPReg: return "SET_FPREG";
  case UOP_SaveNonVol: return "SAVE_NONVOL";
  case UOP_SaveNonVolBig: return "SAVE_NONVOL_FAR";
  case UOP_SaveXMM128: return "SAVE_XMM128";
  case UOP_SaveXMM128Big: return "SAVE_XMM128_FAR";
  case UOP_PushMachFrame: return "PUSH_MACHFRAME";
  }
}

// Returns the name of a referenced register.
static StringRef getUnwindRegisterName(uint8_t Reg) {
  switch(Reg) {
  default: llvm_unreachable("Invalid register");
  case 0: return "RAX";
  case 1: return "RCX";
  case 2: return "RDX";
  case 3: return "RBX";
  case 4: return "RSP";
  case 5: return "RBP";
  case 6: return "RSI";
  case 7: return "RDI";
  case 8: return "R8";
  case 9: return "R9";
  case 10: return "R10";
  case 11: return "R11";
  case 12: return "R12";
  case 13: return "R13";
  case 14: return "R14";
  case 15: return "R15";
  }
}

// Calculates the number of array slots required for the unwind code.
static unsigned getNumUsedSlots(const UnwindCode &UnwindCode) {
  switch (UnwindCode.getUnwindOp()) {
  default: llvm_unreachable("Invalid unwind code");
  case UOP_PushNonVol:
  case UOP_AllocSmall:
  case UOP_SetFPReg:
  case UOP_PushMachFrame:
    return 1;
  case UOP_SaveNonVol:
  case UOP_SaveXMM128:
    return 2;
  case UOP_SaveNonVolBig:
  case UOP_SaveXMM128Big:
    return 3;
  case UOP_AllocLarge:
    return (UnwindCode.getOpInfo() == 0) ? 2 : 3;
  }
}

// Given a symbol sym this functions returns the address and section of it.
static error_code resolveSectionAndAddress(const COFFObjectFile *Obj,
                                           const SymbolRef &Sym,
                                           const coff_section *&ResolvedSection,
                                           uint64_t &ResolvedAddr) {
  if (error_code EC = Sym.getAddress(ResolvedAddr))
    return EC;

  section_iterator iter(Obj->section_begin());
  if (error_code EC = Sym.getSection(iter))
    return EC;

  ResolvedSection = Obj->getCOFFSection(*iter);
  return object_error::success;
}

// Given a a section and an offset into this section the function returns the
// symbol used for the relocation at the offset.
error_code COFFDumper::resolveSymbol(const coff_section *Section,
                                     uint64_t Offset, SymbolRef &Sym) {
  const auto &Relocations = RelocMap[Section];
  for (const auto &Relocation : Relocations) {
    uint64_t RelocationOffset;
    if (error_code EC = Relocation.getOffset(RelocationOffset))
      return EC;

    if (RelocationOffset == Offset) {
      Sym = *Relocation.getSymbol();
      return readobj_error::success;
    }
  }
  return readobj_error::unknown_symbol;
}

// Given a section and an offset into this section the function returns the name
// of the symbol used for the relocation at the offset.
error_code COFFDumper::resolveSymbolName(const coff_section *Section,
                                         uint64_t Offset, StringRef &Name) {
  SymbolRef Symbol;
  if (error_code EC = resolveSymbol(Section, Offset, Symbol))
    return EC;
  if (error_code EC = Symbol.getName(Name))
    return EC;
  return object_error::success;
}

static const EnumEntry<COFF::MachineTypes> ImageFileMachineType[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_UNKNOWN  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_AM33     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_AMD64    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_ARM      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_ARMNT    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_EBC      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_I386     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_IA64     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_M32R     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_MIPS16   ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_MIPSFPU  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_MIPSFPU16),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_POWERPC  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_POWERPCFP),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_R4000    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH3      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH3DSP   ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH4      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH5      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_THUMB    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_WCEMIPSV2)
};

static const EnumEntry<COFF::Characteristics> ImageFileCharacteristics[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_RELOCS_STRIPPED        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_EXECUTABLE_IMAGE       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_LINE_NUMS_STRIPPED     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_LOCAL_SYMS_STRIPPED    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_AGGRESSIVE_WS_TRIM     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_LARGE_ADDRESS_AWARE    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_BYTES_REVERSED_LO      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_32BIT_MACHINE          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_DEBUG_STRIPPED         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_NET_RUN_FROM_SWAP      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_SYSTEM                 ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_DLL                    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_UP_SYSTEM_ONLY         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_BYTES_REVERSED_HI      )
};

static const EnumEntry<COFF::WindowsSubsystem> PEWindowsSubsystem[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_UNKNOWN                ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_NATIVE                 ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_WINDOWS_GUI            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_WINDOWS_CUI            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_POSIX_CUI              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_WINDOWS_CE_GUI         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_APPLICATION        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_ROM                ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_XBOX                   ),
};

static const EnumEntry<COFF::DLLCharacteristics> PEDLLCharacteristics[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_HIGH_ENTROPY_VA      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_FORCE_INTEGRITY      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NX_COMPAT            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NO_ISOLATION         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NO_SEH               ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NO_BIND              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_WDM_DRIVER           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_TERMINAL_SERVER_AWARE),
};

static const EnumEntry<COFF::SectionCharacteristics>
ImageSectionCharacteristics[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_TYPE_NO_PAD           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_CNT_CODE              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_CNT_INITIALIZED_DATA  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_CNT_UNINITIALIZED_DATA),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_OTHER             ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_INFO              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_REMOVE            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_COMDAT            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_GPREL                 ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_PURGEABLE         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_16BIT             ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_LOCKED            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_PRELOAD           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_1BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_2BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_4BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_8BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_16BYTES         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_32BYTES         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_64BYTES         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_128BYTES        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_256BYTES        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_512BYTES        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_1024BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_2048BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_4096BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_8192BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_NRELOC_OVFL       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_DISCARDABLE       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_NOT_CACHED        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_NOT_PAGED         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_SHARED            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_EXECUTE           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_READ              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_WRITE             )
};

static const EnumEntry<COFF::SymbolBaseType> ImageSymType[] = {
  { "Null"  , COFF::IMAGE_SYM_TYPE_NULL   },
  { "Void"  , COFF::IMAGE_SYM_TYPE_VOID   },
  { "Char"  , COFF::IMAGE_SYM_TYPE_CHAR   },
  { "Short" , COFF::IMAGE_SYM_TYPE_SHORT  },
  { "Int"   , COFF::IMAGE_SYM_TYPE_INT    },
  { "Long"  , COFF::IMAGE_SYM_TYPE_LONG   },
  { "Float" , COFF::IMAGE_SYM_TYPE_FLOAT  },
  { "Double", COFF::IMAGE_SYM_TYPE_DOUBLE },
  { "Struct", COFF::IMAGE_SYM_TYPE_STRUCT },
  { "Union" , COFF::IMAGE_SYM_TYPE_UNION  },
  { "Enum"  , COFF::IMAGE_SYM_TYPE_ENUM   },
  { "MOE"   , COFF::IMAGE_SYM_TYPE_MOE    },
  { "Byte"  , COFF::IMAGE_SYM_TYPE_BYTE   },
  { "Word"  , COFF::IMAGE_SYM_TYPE_WORD   },
  { "UInt"  , COFF::IMAGE_SYM_TYPE_UINT   },
  { "DWord" , COFF::IMAGE_SYM_TYPE_DWORD  }
};

static const EnumEntry<COFF::SymbolComplexType> ImageSymDType[] = {
  { "Null"    , COFF::IMAGE_SYM_DTYPE_NULL     },
  { "Pointer" , COFF::IMAGE_SYM_DTYPE_POINTER  },
  { "Function", COFF::IMAGE_SYM_DTYPE_FUNCTION },
  { "Array"   , COFF::IMAGE_SYM_DTYPE_ARRAY    }
};

static const EnumEntry<COFF::SymbolStorageClass> ImageSymClass[] = {
  { "EndOfFunction"  , COFF::IMAGE_SYM_CLASS_END_OF_FUNCTION  },
  { "Null"           , COFF::IMAGE_SYM_CLASS_NULL             },
  { "Automatic"      , COFF::IMAGE_SYM_CLASS_AUTOMATIC        },
  { "External"       , COFF::IMAGE_SYM_CLASS_EXTERNAL         },
  { "Static"         , COFF::IMAGE_SYM_CLASS_STATIC           },
  { "Register"       , COFF::IMAGE_SYM_CLASS_REGISTER         },
  { "ExternalDef"    , COFF::IMAGE_SYM_CLASS_EXTERNAL_DEF     },
  { "Label"          , COFF::IMAGE_SYM_CLASS_LABEL            },
  { "UndefinedLabel" , COFF::IMAGE_SYM_CLASS_UNDEFINED_LABEL  },
  { "MemberOfStruct" , COFF::IMAGE_SYM_CLASS_MEMBER_OF_STRUCT },
  { "Argument"       , COFF::IMAGE_SYM_CLASS_ARGUMENT         },
  { "StructTag"      , COFF::IMAGE_SYM_CLASS_STRUCT_TAG       },
  { "MemberOfUnion"  , COFF::IMAGE_SYM_CLASS_MEMBER_OF_UNION  },
  { "UnionTag"       , COFF::IMAGE_SYM_CLASS_UNION_TAG        },
  { "TypeDefinition" , COFF::IMAGE_SYM_CLASS_TYPE_DEFINITION  },
  { "UndefinedStatic", COFF::IMAGE_SYM_CLASS_UNDEFINED_STATIC },
  { "EnumTag"        , COFF::IMAGE_SYM_CLASS_ENUM_TAG         },
  { "MemberOfEnum"   , COFF::IMAGE_SYM_CLASS_MEMBER_OF_ENUM   },
  { "RegisterParam"  , COFF::IMAGE_SYM_CLASS_REGISTER_PARAM   },
  { "BitField"       , COFF::IMAGE_SYM_CLASS_BIT_FIELD        },
  { "Block"          , COFF::IMAGE_SYM_CLASS_BLOCK            },
  { "Function"       , COFF::IMAGE_SYM_CLASS_FUNCTION         },
  { "EndOfStruct"    , COFF::IMAGE_SYM_CLASS_END_OF_STRUCT    },
  { "File"           , COFF::IMAGE_SYM_CLASS_FILE             },
  { "Section"        , COFF::IMAGE_SYM_CLASS_SECTION          },
  { "WeakExternal"   , COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL    },
  { "CLRToken"       , COFF::IMAGE_SYM_CLASS_CLR_TOKEN        }
};

static const EnumEntry<COFF::COMDATType> ImageCOMDATSelect[] = {
  { "NoDuplicates", COFF::IMAGE_COMDAT_SELECT_NODUPLICATES },
  { "Any"         , COFF::IMAGE_COMDAT_SELECT_ANY          },
  { "SameSize"    , COFF::IMAGE_COMDAT_SELECT_SAME_SIZE    },
  { "ExactMatch"  , COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH  },
  { "Associative" , COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE  },
  { "Largest"     , COFF::IMAGE_COMDAT_SELECT_LARGEST      },
  { "Newest"      , COFF::IMAGE_COMDAT_SELECT_NEWEST       }
};

static const EnumEntry<COFF::WeakExternalCharacteristics>
WeakExternalCharacteristics[] = {
  { "NoLibrary", COFF::IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY },
  { "Library"  , COFF::IMAGE_WEAK_EXTERN_SEARCH_LIBRARY   },
  { "Alias"    , COFF::IMAGE_WEAK_EXTERN_SEARCH_ALIAS     }
};

static const EnumEntry<unsigned> UnwindFlags[] = {
  { "ExceptionHandler", Win64EH::UNW_ExceptionHandler },
  { "TerminateHandler", Win64EH::UNW_TerminateHandler },
  { "ChainInfo"       , Win64EH::UNW_ChainInfo        }
};

static const EnumEntry<unsigned> UnwindOpInfo[] = {
  { "RAX",  0 },
  { "RCX",  1 },
  { "RDX",  2 },
  { "RBX",  3 },
  { "RSP",  4 },
  { "RBP",  5 },
  { "RSI",  6 },
  { "RDI",  7 },
  { "R8",   8 },
  { "R9",   9 },
  { "R10", 10 },
  { "R11", 11 },
  { "R12", 12 },
  { "R13", 13 },
  { "R14", 14 },
  { "R15", 15 }
};

static uint64_t getOffsetOfLSDA(const Win64EH::UnwindInfo& UI) {
  return static_cast<const char*>(UI.getLanguageSpecificData())
         - reinterpret_cast<const char*>(&UI);
}

static uint32_t getLargeSlotValue(ArrayRef<UnwindCode> UCs) {
  if (UCs.size() < 3)
    return 0;

  return UCs[1].FrameOffset + (static_cast<uint32_t>(UCs[2].FrameOffset) << 16);
}

template<typename T>
static error_code getSymbolAuxData(const COFFObjectFile *Obj,
                                   const coff_symbol *Symbol, const T* &Aux) {
  ArrayRef<uint8_t> AuxData = Obj->getSymbolAuxData(Symbol);
  Aux = reinterpret_cast<const T*>(AuxData.data());
  return readobj_error::success;
}

std::string COFFDumper::formatSymbol(const coff_section *Section,
                                     uint64_t Offset, uint32_t Disp) {
  std::string Buffer;
  raw_string_ostream Str(Buffer);

  StringRef Sym;
  if (resolveSymbolName(Section, Offset, Sym)) {
    Str << format(" (0x%" PRIX64 ")", Offset);
    return Str.str();
  }

  Str << Sym;
  if (Disp > 0) {
    Str << format(" +0x%X (0x%" PRIX64 ")", Disp, Offset);
  } else {
    Str << format(" (0x%" PRIX64 ")", Offset);
  }

  return Str.str();
}

error_code COFFDumper::resolveRelocation(const coff_section *Section,
                                         uint64_t Offset,
                                         const coff_section *&ResolvedSection,
                                         uint64_t &ResolvedAddress) {
  SymbolRef Sym;
  if (error_code EC = resolveSymbol(Section, Offset, Sym))
    return EC;

  if (error_code EC = resolveSectionAndAddress(Obj, Sym, ResolvedSection,
                                               ResolvedAddress))
    return EC;

  return object_error::success;
}

void COFFDumper::cacheRelocations() {
  for (const SectionRef &S : Obj->sections()) {
    const coff_section *Section = Obj->getCOFFSection(S);

    for (const RelocationRef &Reloc : S.relocations())
      RelocMap[Section].push_back(Reloc);

    // Sort relocations by address.
    std::sort(RelocMap[Section].begin(), RelocMap[Section].end(),
              relocAddressLess);
  }
}

void COFFDumper::printDataDirectory(uint32_t Index, const std::string &FieldName) {
  const data_directory *Data;
  if (Obj->getDataDirectory(Index, Data))
    return;
  W.printHex(FieldName + "RVA", Data->RelativeVirtualAddress);
  W.printHex(FieldName + "Size", Data->Size);
}

void COFFDumper::printFileHeaders() {
  // Print COFF header
  const coff_file_header *COFFHeader = nullptr;
  if (error(Obj->getCOFFHeader(COFFHeader)))
    return;

  time_t TDS = COFFHeader->TimeDateStamp;
  char FormattedTime[20] = { };
  strftime(FormattedTime, 20, "%Y-%m-%d %H:%M:%S", gmtime(&TDS));

  {
    DictScope D(W, "ImageFileHeader");
    W.printEnum  ("Machine", COFFHeader->Machine,
                    makeArrayRef(ImageFileMachineType));
    W.printNumber("SectionCount", COFFHeader->NumberOfSections);
    W.printHex   ("TimeDateStamp", FormattedTime, COFFHeader->TimeDateStamp);
    W.printHex   ("PointerToSymbolTable", COFFHeader->PointerToSymbolTable);
    W.printNumber("SymbolCount", COFFHeader->NumberOfSymbols);
    W.printNumber("OptionalHeaderSize", COFFHeader->SizeOfOptionalHeader);
    W.printFlags ("Characteristics", COFFHeader->Characteristics,
                    makeArrayRef(ImageFileCharacteristics));
  }

  // Print PE header. This header does not exist if this is an object file and
  // not an executable.
  const pe32_header *PEHeader = nullptr;
  if (error(Obj->getPE32Header(PEHeader)))
    return;
  if (PEHeader)
    printPEHeader<pe32_header>(PEHeader);

  const pe32plus_header *PEPlusHeader = nullptr;
  if (error(Obj->getPE32PlusHeader(PEPlusHeader)))
    return;
  if (PEPlusHeader)
    printPEHeader<pe32plus_header>(PEPlusHeader);
}

template <class PEHeader>
void COFFDumper::printPEHeader(const PEHeader *Hdr) {
  DictScope D(W, "ImageOptionalHeader");
  W.printNumber("MajorLinkerVersion", Hdr->MajorLinkerVersion);
  W.printNumber("MinorLinkerVersion", Hdr->MinorLinkerVersion);
  W.printNumber("SizeOfCode", Hdr->SizeOfCode);
  W.printNumber("SizeOfInitializedData", Hdr->SizeOfInitializedData);
  W.printNumber("SizeOfUninitializedData", Hdr->SizeOfUninitializedData);
  W.printHex   ("AddressOfEntryPoint", Hdr->AddressOfEntryPoint);
  W.printHex   ("BaseOfCode", Hdr->BaseOfCode);
  printBaseOfDataField(Hdr);
  W.printHex   ("ImageBase", Hdr->ImageBase);
  W.printNumber("SectionAlignment", Hdr->SectionAlignment);
  W.printNumber("FileAlignment", Hdr->FileAlignment);
  W.printNumber("MajorOperatingSystemVersion",
                Hdr->MajorOperatingSystemVersion);
  W.printNumber("MinorOperatingSystemVersion",
                Hdr->MinorOperatingSystemVersion);
  W.printNumber("MajorImageVersion", Hdr->MajorImageVersion);
  W.printNumber("MinorImageVersion", Hdr->MinorImageVersion);
  W.printNumber("MajorSubsystemVersion", Hdr->MajorSubsystemVersion);
  W.printNumber("MinorSubsystemVersion", Hdr->MinorSubsystemVersion);
  W.printNumber("SizeOfImage", Hdr->SizeOfImage);
  W.printNumber("SizeOfHeaders", Hdr->SizeOfHeaders);
  W.printEnum  ("Subsystem", Hdr->Subsystem, makeArrayRef(PEWindowsSubsystem));
  W.printFlags ("Subsystem", Hdr->DLLCharacteristics,
                makeArrayRef(PEDLLCharacteristics));
  W.printNumber("SizeOfStackReserve", Hdr->SizeOfStackReserve);
  W.printNumber("SizeOfStackCommit", Hdr->SizeOfStackCommit);
  W.printNumber("SizeOfHeapReserve", Hdr->SizeOfHeapReserve);
  W.printNumber("SizeOfHeapCommit", Hdr->SizeOfHeapCommit);
  W.printNumber("NumberOfRvaAndSize", Hdr->NumberOfRvaAndSize);

  if (Hdr->NumberOfRvaAndSize > 0) {
    DictScope D(W, "DataDirectory");
    static const char * const directory[] = {
      "ExportTable", "ImportTable", "ResourceTable", "ExceptionTable",
      "CertificateTable", "BaseRelocationTable", "Debug", "Architecture",
      "GlobalPtr", "TLSTable", "LoadConfigTable", "BoundImport", "IAT",
      "DelayImportDescriptor", "CLRRuntimeHeader", "Reserved"
    };

    for (uint32_t i = 0; i < Hdr->NumberOfRvaAndSize; ++i) {
      printDataDirectory(i, directory[i]);
    }
  }
}

void COFFDumper::printBaseOfDataField(const pe32_header *Hdr) {
  W.printHex("BaseOfData", Hdr->BaseOfData);
}

void COFFDumper::printBaseOfDataField(const pe32plus_header *) {}

void COFFDumper::printCodeViewLineTables(const SectionRef &Section) {
  StringRef Data;
  if (error(Section.getContents(Data)))
    return;

  SmallVector<StringRef, 10> FunctionNames;
  StringMap<StringRef> FunctionLineTables;
  StringRef FileIndexToStringOffsetTable;
  StringRef StringTable;

  ListScope D(W, "CodeViewLineTables");
  {
    DataExtractor DE(Data, true, 4);
    uint32_t Offset = 0,
             Magic = DE.getU32(&Offset);
    W.printHex("Magic", Magic);
    if (Magic != COFF::DEBUG_SECTION_MAGIC) {
      error(object_error::parse_failed);
      return;
    }

    bool Finished = false;
    while (DE.isValidOffset(Offset) && !Finished) {
      // The section consists of a number of subsection in the following format:
      // |Type|PayloadSize|Payload...|
      uint32_t SubSectionType = DE.getU32(&Offset),
               PayloadSize = DE.getU32(&Offset);
      ListScope S(W, "Subsection");
      W.printHex("Type", SubSectionType);
      W.printHex("PayloadSize", PayloadSize);
      if (PayloadSize > Data.size() - Offset) {
        error(object_error::parse_failed);
        return;
      }

      // Print the raw contents to simplify debugging if anything goes wrong
      // afterwards.
      StringRef Contents = Data.substr(Offset, PayloadSize);
      W.printBinaryBlock("Contents", Contents);

      switch (SubSectionType) {
      case COFF::DEBUG_LINE_TABLE_SUBSECTION: {
        // Holds a PC to file:line table.  Some data to parse this subsection is
        // stored in the other subsections, so just check sanity and store the
        // pointers for deferred processing.

        if (PayloadSize < 12) {
          // There should be at least three words to store two function
          // relocations and size of the code.
          error(object_error::parse_failed);
          return;
        }

        StringRef FunctionName;
        if (error(resolveSymbolName(Obj->getCOFFSection(Section), Offset,
                                    FunctionName)))
          return;
        W.printString("FunctionName", FunctionName);
        if (FunctionLineTables.count(FunctionName) != 0) {
          // Saw debug info for this function already?
          error(object_error::parse_failed);
          return;
        }

        FunctionLineTables[FunctionName] = Contents;
        FunctionNames.push_back(FunctionName);
        break;
      }
      case COFF::DEBUG_STRING_TABLE_SUBSECTION:
        if (PayloadSize == 0 || StringTable.data() != nullptr ||
            Contents.back() != '\0') {
          // Empty or duplicate or non-null-terminated subsection.
          error(object_error::parse_failed);
          return;
        }
        StringTable = Contents;
        break;
      case COFF::DEBUG_INDEX_SUBSECTION:
        // Holds the translation table from file indices
        // to offsets in the string table.

        if (PayloadSize == 0 ||
            FileIndexToStringOffsetTable.data() != nullptr) {
          // Empty or duplicate subsection.
          error(object_error::parse_failed);
          return;
        }
        FileIndexToStringOffsetTable = Contents;
        break;
      }
      Offset += PayloadSize;

      // Align the reading pointer by 4.
      Offset += (-Offset) % 4;
    }
  }

  // Dump the line tables now that we've read all the subsections and know all
  // the required information.
  for (unsigned I = 0, E = FunctionNames.size(); I != E; ++I) {
    StringRef Name = FunctionNames[I];
    ListScope S(W, "FunctionLineTable");
    W.printString("FunctionName", Name);

    DataExtractor DE(FunctionLineTables[Name], true, 4);
    uint32_t Offset = 8;  // Skip relocations.
    uint32_t FunctionSize = DE.getU32(&Offset);
    W.printHex("CodeSize", FunctionSize);
    while (DE.isValidOffset(Offset)) {
      // For each range of lines with the same filename, we have a segment
      // in the line table.  The filename string is accessed using double
      // indirection to the string table subsection using the index subsection.
      uint32_t OffsetInIndex = DE.getU32(&Offset),
               SegmentLength   = DE.getU32(&Offset),
               FullSegmentSize = DE.getU32(&Offset);
      if (FullSegmentSize != 12 + 8 * SegmentLength) {
        error(object_error::parse_failed);
        return;
      }

      uint32_t FilenameOffset;
      {
        DataExtractor SDE(FileIndexToStringOffsetTable, true, 4);
        uint32_t OffsetInSDE = OffsetInIndex;
        if (!SDE.isValidOffset(OffsetInSDE)) {
          error(object_error::parse_failed);
          return;
        }
        FilenameOffset = SDE.getU32(&OffsetInSDE);
      }

      if (FilenameOffset == 0 || FilenameOffset + 1 >= StringTable.size() ||
          StringTable.data()[FilenameOffset - 1] != '\0') {
        // Each string in an F3 subsection should be preceded by a null
        // character.
        error(object_error::parse_failed);
        return;
      }

      StringRef Filename(StringTable.data() + FilenameOffset);
      ListScope S(W, "FilenameSegment");
      W.printString("Filename", Filename);
      for (unsigned J = 0; J != SegmentLength && DE.isValidOffset(Offset);
           ++J) {
        // Then go the (PC, LineNumber) pairs.  The line number is stored in the
        // least significant 31 bits of the respective word in the table.
        uint32_t PC = DE.getU32(&Offset),
                 LineNumber = DE.getU32(&Offset) & 0x7fffffff;
        if (PC >= FunctionSize) {
          error(object_error::parse_failed);
          return;
        }
        char Buffer[32];
        format("+0x%X", PC).snprint(Buffer, 32);
        W.printNumber(Buffer, LineNumber);
      }
    }
  }
}

void COFFDumper::printSections() {
  ListScope SectionsD(W, "Sections");
  int SectionNumber = 0;
  for (const SectionRef &Sec : Obj->sections()) {
    ++SectionNumber;
    const coff_section *Section = Obj->getCOFFSection(Sec);

    StringRef Name;
    if (error(Sec.getName(Name)))
      Name = "";

    DictScope D(W, "Section");
    W.printNumber("Number", SectionNumber);
    W.printBinary("Name", Name, Section->Name);
    W.printHex   ("VirtualSize", Section->VirtualSize);
    W.printHex   ("VirtualAddress", Section->VirtualAddress);
    W.printNumber("RawDataSize", Section->SizeOfRawData);
    W.printHex   ("PointerToRawData", Section->PointerToRawData);
    W.printHex   ("PointerToRelocations", Section->PointerToRelocations);
    W.printHex   ("PointerToLineNumbers", Section->PointerToLinenumbers);
    W.printNumber("RelocationCount", Section->NumberOfRelocations);
    W.printNumber("LineNumberCount", Section->NumberOfLinenumbers);
    W.printFlags ("Characteristics", Section->Characteristics,
                    makeArrayRef(ImageSectionCharacteristics),
                    COFF::SectionCharacteristics(0x00F00000));

    if (opts::SectionRelocations) {
      ListScope D(W, "Relocations");
      for (const RelocationRef &Reloc : Sec.relocations())
        printRelocation(Sec, Reloc);
    }

    if (opts::SectionSymbols) {
      ListScope D(W, "Symbols");
      for (const SymbolRef &Symbol : Obj->symbols()) {
        bool Contained = false;
        if (Sec.containsSymbol(Symbol, Contained) || !Contained)
          continue;

        printSymbol(Symbol);
      }
    }

    if (Name == ".debug$S" && opts::CodeViewLineTables)
      printCodeViewLineTables(Sec);

    if (opts::SectionData) {
      StringRef Data;
      if (error(Sec.getContents(Data)))
        break;

      W.printBinaryBlock("SectionData", Data);
    }
  }
}

void COFFDumper::printRelocations() {
  ListScope D(W, "Relocations");

  int SectionNumber = 0;
  for (const SectionRef &Section : Obj->sections()) {
    ++SectionNumber;
    StringRef Name;
    if (error(Section.getName(Name)))
      continue;

    bool PrintedGroup = false;
    for (const RelocationRef &Reloc : Section.relocations()) {
      if (!PrintedGroup) {
        W.startLine() << "Section (" << SectionNumber << ") " << Name << " {\n";
        W.indent();
        PrintedGroup = true;
      }

      printRelocation(Section, Reloc);
    }

    if (PrintedGroup) {
      W.unindent();
      W.startLine() << "}\n";
    }
  }
}

void COFFDumper::printRelocation(const SectionRef &Section,
                                 const RelocationRef &Reloc) {
  uint64_t Offset;
  uint64_t RelocType;
  SmallString<32> RelocName;
  StringRef SymbolName;
  StringRef Contents;
  if (error(Reloc.getOffset(Offset)))
    return;
  if (error(Reloc.getType(RelocType)))
    return;
  if (error(Reloc.getTypeName(RelocName)))
    return;
  symbol_iterator Symbol = Reloc.getSymbol();
  if (error(Symbol->getName(SymbolName)))
    return;
  if (error(Section.getContents(Contents)))
    return;

  if (opts::ExpandRelocs) {
    DictScope Group(W, "Relocation");
    W.printHex("Offset", Offset);
    W.printNumber("Type", RelocName, RelocType);
    W.printString("Symbol", SymbolName.size() > 0 ? SymbolName : "-");
  } else {
    raw_ostream& OS = W.startLine();
    OS << W.hex(Offset)
       << " " << RelocName
       << " " << (SymbolName.size() > 0 ? SymbolName : "-")
       << "\n";
  }
}

void COFFDumper::printSymbols() {
  ListScope Group(W, "Symbols");

  for (const SymbolRef &Symbol : Obj->symbols())
    printSymbol(Symbol);
}

void COFFDumper::printDynamicSymbols() { ListScope Group(W, "DynamicSymbols"); }

void COFFDumper::printSymbol(const SymbolRef &Sym) {
  DictScope D(W, "Symbol");

  const coff_symbol *Symbol = Obj->getCOFFSymbol(Sym);
  const coff_section *Section;
  if (error_code EC = Obj->getSection(Symbol->SectionNumber, Section)) {
    W.startLine() << "Invalid section number: " << EC.message() << "\n";
    W.flush();
    return;
  }

  StringRef SymbolName;
  if (Obj->getSymbolName(Symbol, SymbolName))
    SymbolName = "";

  StringRef SectionName = "";
  if (Section)
    Obj->getSectionName(Section, SectionName);

  W.printString("Name", SymbolName);
  W.printNumber("Value", Symbol->Value);
  W.printNumber("Section", SectionName, Symbol->SectionNumber);
  W.printEnum  ("BaseType", Symbol->getBaseType(), makeArrayRef(ImageSymType));
  W.printEnum  ("ComplexType", Symbol->getComplexType(),
                                                   makeArrayRef(ImageSymDType));
  W.printEnum  ("StorageClass", Symbol->StorageClass,
                                                   makeArrayRef(ImageSymClass));
  W.printNumber("AuxSymbolCount", Symbol->NumberOfAuxSymbols);

  for (unsigned I = 0; I < Symbol->NumberOfAuxSymbols; ++I) {
    if (Symbol->isFunctionDefinition()) {
      const coff_aux_function_definition *Aux;
      if (error(getSymbolAuxData(Obj, Symbol + I, Aux)))
        break;

      DictScope AS(W, "AuxFunctionDef");
      W.printNumber("TagIndex", Aux->TagIndex);
      W.printNumber("TotalSize", Aux->TotalSize);
      W.printHex("PointerToLineNumber", Aux->PointerToLinenumber);
      W.printHex("PointerToNextFunction", Aux->PointerToNextFunction);
      W.printBinary("Unused", makeArrayRef(Aux->Unused));

    } else if (Symbol->isWeakExternal()) {
      const coff_aux_weak_external *Aux;
      if (error(getSymbolAuxData(Obj, Symbol + I, Aux)))
        break;

      const coff_symbol *Linked;
      StringRef LinkedName;
      error_code EC;
      if ((EC = Obj->getSymbol(Aux->TagIndex, Linked)) ||
          (EC = Obj->getSymbolName(Linked, LinkedName))) {
        LinkedName = "";
        error(EC);
      }

      DictScope AS(W, "AuxWeakExternal");
      W.printNumber("Linked", LinkedName, Aux->TagIndex);
      W.printEnum  ("Search", Aux->Characteristics,
                    makeArrayRef(WeakExternalCharacteristics));
      W.printBinary("Unused", makeArrayRef(Aux->Unused));

    } else if (Symbol->isFileRecord()) {
      const coff_aux_file *Aux;
      if (error(getSymbolAuxData(Obj, Symbol + I, Aux)))
        break;

      DictScope AS(W, "AuxFileRecord");

      StringRef Name(Aux->FileName,
                     Symbol->NumberOfAuxSymbols * COFF::SymbolSize);
      W.printString("FileName", Name.rtrim(StringRef("\0", 1)));
      break;
    } else if (Symbol->isSectionDefinition()) {
      const coff_aux_section_definition *Aux;
      if (error(getSymbolAuxData(Obj, Symbol + I, Aux)))
        break;

      DictScope AS(W, "AuxSectionDef");
      W.printNumber("Length", Aux->Length);
      W.printNumber("RelocationCount", Aux->NumberOfRelocations);
      W.printNumber("LineNumberCount", Aux->NumberOfLinenumbers);
      W.printHex("Checksum", Aux->CheckSum);
      W.printNumber("Number", Aux->Number);
      W.printEnum("Selection", Aux->Selection, makeArrayRef(ImageCOMDATSelect));
      W.printBinary("Unused", makeArrayRef(Aux->Unused));

      if (Section && Section->Characteristics & COFF::IMAGE_SCN_LNK_COMDAT
          && Aux->Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) {
        const coff_section *Assoc;
        StringRef AssocName;
        error_code EC;
        if ((EC = Obj->getSection(Aux->Number, Assoc)) ||
            (EC = Obj->getSectionName(Assoc, AssocName))) {
          AssocName = "";
          error(EC);
        }

        W.printNumber("AssocSection", AssocName, Aux->Number);
      }
    } else if (Symbol->isCLRToken()) {
      const coff_aux_clr_token *Aux;
      if (error(getSymbolAuxData(Obj, Symbol + I, Aux)))
        break;

      const coff_symbol *ReferredSym;
      StringRef ReferredName;
      error_code EC;
      if ((EC = Obj->getSymbol(Aux->SymbolTableIndex, ReferredSym)) ||
          (EC = Obj->getSymbolName(ReferredSym, ReferredName))) {
        ReferredName = "";
        error(EC);
      }

      DictScope AS(W, "AuxCLRToken");
      W.printNumber("AuxType", Aux->AuxType);
      W.printNumber("Reserved", Aux->Reserved);
      W.printNumber("SymbolTableIndex", ReferredName, Aux->SymbolTableIndex);
      W.printBinary("Unused", makeArrayRef(Aux->Unused));

    } else {
      W.startLine() << "<unhandled auxiliary record>\n";
    }
  }
}

void COFFDumper::printUnwindInfo() {
  const coff_file_header *Header;
  if (error(Obj->getCOFFHeader(Header)))
    return;

  ListScope D(W, "UnwindInformation");
  if (Header->Machine != COFF::IMAGE_FILE_MACHINE_AMD64) {
    W.startLine() << "Unsupported image machine type "
              "(currently only AMD64 is supported).\n";
    return;
  }

  printX64UnwindInfo();
}

void COFFDumper::printX64UnwindInfo() {
  for (const SectionRef &Section : Obj->sections()) {
    StringRef Name;
    if (error(Section.getName(Name)))
      continue;
    if (Name != ".pdata" && !Name.startswith(".pdata$"))
      continue;

    const coff_section *PData = Obj->getCOFFSection(Section);

    ArrayRef<uint8_t> Contents;
    if (error(Obj->getSectionContents(PData, Contents)) || Contents.empty())
      continue;

    ArrayRef<RuntimeFunction> RFs(
      reinterpret_cast<const RuntimeFunction *>(Contents.data()),
      Contents.size() / sizeof(RuntimeFunction));

    for (const RuntimeFunction *I = RFs.begin(), *E = RFs.end(); I < E; ++I) {
      const uint64_t OffsetInSection = std::distance(RFs.begin(), I)
                                     * sizeof(RuntimeFunction);

      printRuntimeFunction(*I, PData, OffsetInSection);
    }
  }
}

void COFFDumper::printRuntimeFunction(const RuntimeFunction& RTF,
                                      const coff_section *Section,
                                      uint64_t SectionOffset) {

  DictScope D(W, "RuntimeFunction");
  W.printString("StartAddress",
                formatSymbol(Section, SectionOffset + 0, RTF.StartAddress));
  W.printString("EndAddress",
                formatSymbol(Section, SectionOffset + 4, RTF.EndAddress));
  W.printString("UnwindInfoAddress",
                formatSymbol(Section, SectionOffset + 8, RTF.UnwindInfoOffset));

  const coff_section* XData = nullptr;
  uint64_t UnwindInfoOffset = 0;
  if (error(getSectionFromRelocation(Section, SectionOffset + 8,
                                     XData, UnwindInfoOffset)))
    return;

  ArrayRef<uint8_t> XContents;
  if (error(Obj->getSectionContents(XData, XContents)) || XContents.empty())
    return;

  UnwindInfoOffset += RTF.UnwindInfoOffset;
  if (UnwindInfoOffset > XContents.size())
    return;

  const Win64EH::UnwindInfo *UI =
    reinterpret_cast<const Win64EH::UnwindInfo *>(
      XContents.data() + UnwindInfoOffset);

  printUnwindInfo(*UI, XData, UnwindInfoOffset);
}

void COFFDumper::printUnwindInfo(const Win64EH::UnwindInfo& UI,
                                 const coff_section *Section,
                                 uint64_t SectionOffset) {
  DictScope D(W, "UnwindInfo");
  W.printNumber("Version", UI.getVersion());
  W.printFlags("Flags", UI.getFlags(), makeArrayRef(UnwindFlags));
  W.printNumber("PrologSize", UI.PrologSize);
  if (UI.getFrameRegister() != 0) {
    W.printEnum("FrameRegister", UI.getFrameRegister(),
                makeArrayRef(UnwindOpInfo));
    W.printHex("FrameOffset", UI.getFrameOffset());
  } else {
    W.printString("FrameRegister", StringRef("-"));
    W.printString("FrameOffset", StringRef("-"));
  }

  W.printNumber("UnwindCodeCount", UI.NumCodes);
  {
    ListScope CodesD(W, "UnwindCodes");
    ArrayRef<UnwindCode> UCs(&UI.UnwindCodes[0], UI.NumCodes);
    for (const UnwindCode *I = UCs.begin(), *E = UCs.end(); I < E; ++I) {
      unsigned UsedSlots = getNumUsedSlots(*I);
      if (UsedSlots > UCs.size()) {
        errs() << "Corrupt unwind data";
        return;
      }
      printUnwindCode(UI, ArrayRef<UnwindCode>(I, E));
      I += UsedSlots - 1;
    }
  }

  uint64_t LSDAOffset = SectionOffset + getOffsetOfLSDA(UI);
  if (UI.getFlags() & (UNW_ExceptionHandler | UNW_TerminateHandler)) {
    W.printString("Handler",
                  formatSymbol(Section, LSDAOffset,
                               UI.getLanguageSpecificHandlerOffset()));
  } else if (UI.getFlags() & UNW_ChainInfo) {
    const RuntimeFunction *Chained = UI.getChainedFunctionEntry();
    if (Chained) {
      DictScope D(W, "Chained");
      W.printString("StartAddress", formatSymbol(Section, LSDAOffset + 0,
                                                 Chained->StartAddress));
      W.printString("EndAddress", formatSymbol(Section, LSDAOffset + 4,
                                               Chained->EndAddress));
      W.printString("UnwindInfoAddress",
                    formatSymbol(Section, LSDAOffset + 8,
                                 Chained->UnwindInfoOffset));
    }
  }
}

// Prints one unwind code. Because an unwind code can occupy up to 3 slots in
// the unwind codes array, this function requires that the correct number of
// slots is provided.
void COFFDumper::printUnwindCode(const Win64EH::UnwindInfo& UI,
                                 ArrayRef<UnwindCode> UCs) {
  assert(UCs.size() >= getNumUsedSlots(UCs[0]));

  W.startLine() << format("0x%02X: ", unsigned(UCs[0].u.CodeOffset))
                << getUnwindCodeTypeName(UCs[0].getUnwindOp());

  uint32_t AllocSize = 0;

  switch (UCs[0].getUnwindOp()) {
  case UOP_PushNonVol:
    outs() << " reg=" << getUnwindRegisterName(UCs[0].getOpInfo());
    break;

  case UOP_AllocLarge:
    if (UCs[0].getOpInfo() == 0) {
      AllocSize = UCs[1].FrameOffset * 8;
    } else {
      AllocSize = getLargeSlotValue(UCs);
    }
    outs() << " size=" << AllocSize;
    break;
  case UOP_AllocSmall:
    outs() << " size=" << ((UCs[0].getOpInfo() + 1) * 8);
    break;
  case UOP_SetFPReg:
    if (UI.getFrameRegister() == 0) {
      outs() << " reg=<invalid>";
    } else {
      outs() << " reg=" << getUnwindRegisterName(UI.getFrameRegister())
             << format(", offset=0x%X", UI.getFrameOffset() * 16);
    }
    break;
  case UOP_SaveNonVol:
    outs() << " reg=" << getUnwindRegisterName(UCs[0].getOpInfo())
           << format(", offset=0x%X", UCs[1].FrameOffset * 8);
    break;
  case UOP_SaveNonVolBig:
    outs() << " reg=" << getUnwindRegisterName(UCs[0].getOpInfo())
           << format(", offset=0x%X", getLargeSlotValue(UCs));
    break;
  case UOP_SaveXMM128:
    outs() << " reg=XMM" << static_cast<uint32_t>(UCs[0].getOpInfo())
           << format(", offset=0x%X", UCs[1].FrameOffset * 16);
    break;
  case UOP_SaveXMM128Big:
    outs() << " reg=XMM" << static_cast<uint32_t>(UCs[0].getOpInfo())
           << format(", offset=0x%X", getLargeSlotValue(UCs));
    break;
  case UOP_PushMachFrame:
    outs() << " errcode=" << (UCs[0].getOpInfo() == 0 ? "no" : "yes");
    break;
  }

  outs() << "\n";
}
