//===-- MachODump.cpp - Object file dumping utility for llvm --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachO-specific dumper for llvm-objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm-c/Disassembler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <system_error>

#if HAVE_CXXABI_H
#include <cxxabi.h>
#endif

using namespace llvm;
using namespace object;

static cl::opt<bool>
    UseDbg("g",
           cl::desc("Print line information from debug info if available"));

static cl::opt<std::string> DSYMFile("dsym",
                                     cl::desc("Use .dSYM file for debug info"));

static cl::opt<bool> FullLeadingAddr("full-leading-addr",
                                     cl::desc("Print full leading address"));

static cl::opt<bool>
    PrintImmHex("print-imm-hex",
                cl::desc("Use hex format for immediate values"));

static std::string ThumbTripleName;

static const Target *GetTarget(const MachOObjectFile *MachOObj,
                               const char **McpuDefault,
                               const Target **ThumbTarget) {
  // Figure out the target triple.
  if (TripleName.empty()) {
    llvm::Triple TT("unknown-unknown-unknown");
    llvm::Triple ThumbTriple = Triple();
    TT = MachOObj->getArch(McpuDefault, &ThumbTriple);
    TripleName = TT.str();
    ThumbTripleName = ThumbTriple.str();
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget && ThumbTripleName.empty())
    return TheTarget;

  *ThumbTarget = TargetRegistry::lookupTarget(ThumbTripleName, Error);
  if (*ThumbTarget)
    return TheTarget;

  errs() << "llvm-objdump: error: unable to get target for '";
  if (!TheTarget)
    errs() << TripleName;
  else
    errs() << ThumbTripleName;
  errs() << "', see --version and --triple.\n";
  return nullptr;
}

struct SymbolSorter {
  bool operator()(const SymbolRef &A, const SymbolRef &B) {
    SymbolRef::Type AType, BType;
    A.getType(AType);
    B.getType(BType);

    uint64_t AAddr, BAddr;
    if (AType != SymbolRef::ST_Function)
      AAddr = 0;
    else
      A.getAddress(AAddr);
    if (BType != SymbolRef::ST_Function)
      BAddr = 0;
    else
      B.getAddress(BAddr);
    return AAddr < BAddr;
  }
};

// Types for the storted data in code table that is built before disassembly
// and the predicate function to sort them.
typedef std::pair<uint64_t, DiceRef> DiceTableEntry;
typedef std::vector<DiceTableEntry> DiceTable;
typedef DiceTable::iterator dice_table_iterator;

static bool compareDiceTableEntries(const DiceTableEntry i,
                                    const DiceTableEntry j) {
  return i.first == j.first;
}

static void DumpDataInCode(const char *bytes, uint64_t Size,
                           unsigned short Kind) {
  uint64_t Value;

  switch (Kind) {
  case MachO::DICE_KIND_DATA:
    switch (Size) {
    case 4:
      Value = bytes[3] << 24 | bytes[2] << 16 | bytes[1] << 8 | bytes[0];
      outs() << "\t.long " << Value;
      break;
    case 2:
      Value = bytes[1] << 8 | bytes[0];
      outs() << "\t.short " << Value;
      break;
    case 1:
      Value = bytes[0];
      outs() << "\t.byte " << Value;
      break;
    }
    outs() << "\t@ KIND_DATA\n";
    break;
  case MachO::DICE_KIND_JUMP_TABLE8:
    Value = bytes[0];
    outs() << "\t.byte " << Value << "\t@ KIND_JUMP_TABLE8";
    break;
  case MachO::DICE_KIND_JUMP_TABLE16:
    Value = bytes[1] << 8 | bytes[0];
    outs() << "\t.short " << Value << "\t@ KIND_JUMP_TABLE16";
    break;
  case MachO::DICE_KIND_JUMP_TABLE32:
    Value = bytes[3] << 24 | bytes[2] << 16 | bytes[1] << 8 | bytes[0];
    outs() << "\t.long " << Value << "\t@ KIND_JUMP_TABLE32";
    break;
  default:
    outs() << "\t@ data in code kind = " << Kind << "\n";
    break;
  }
}

static void getSectionsAndSymbols(const MachO::mach_header Header,
                                  MachOObjectFile *MachOObj,
                                  std::vector<SectionRef> &Sections,
                                  std::vector<SymbolRef> &Symbols,
                                  SmallVectorImpl<uint64_t> &FoundFns,
                                  uint64_t &BaseSegmentAddress) {
  for (const SymbolRef &Symbol : MachOObj->symbols())
    Symbols.push_back(Symbol);

  for (const SectionRef &Section : MachOObj->sections()) {
    StringRef SectName;
    Section.getName(SectName);
    Sections.push_back(Section);
  }

  MachOObjectFile::LoadCommandInfo Command =
      MachOObj->getFirstLoadCommandInfo();
  bool BaseSegmentAddressSet = false;
  for (unsigned i = 0;; ++i) {
    if (Command.C.cmd == MachO::LC_FUNCTION_STARTS) {
      // We found a function starts segment, parse the addresses for later
      // consumption.
      MachO::linkedit_data_command LLC =
          MachOObj->getLinkeditDataLoadCommand(Command);

      MachOObj->ReadULEB128s(LLC.dataoff, FoundFns);
    } else if (Command.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command SLC = MachOObj->getSegmentLoadCommand(Command);
      StringRef SegName = SLC.segname;
      if (!BaseSegmentAddressSet && SegName != "__PAGEZERO") {
        BaseSegmentAddressSet = true;
        BaseSegmentAddress = SLC.vmaddr;
      }
    }

    if (i == Header.ncmds - 1)
      break;
    else
      Command = MachOObj->getNextLoadCommandInfo(Command);
  }
}

static void DisassembleInputMachO2(StringRef Filename,
                                   MachOObjectFile *MachOOF);

void llvm::DisassembleInputMachO(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BuffOrErr.getError()) {
    errs() << "llvm-objdump: " << Filename << ": " << EC.message() << "\n";
    return;
  }
  std::unique_ptr<MemoryBuffer> Buff = std::move(BuffOrErr.get());

  std::unique_ptr<MachOObjectFile> MachOOF = std::move(
      ObjectFile::createMachOObjectFile(Buff.get()->getMemBufferRef()).get());

  DisassembleInputMachO2(Filename, MachOOF.get());
}

typedef DenseMap<uint64_t, StringRef> SymbolAddressMap;
typedef std::pair<uint64_t, const char *> BindInfoEntry;
typedef std::vector<BindInfoEntry> BindTable;
typedef BindTable::iterator bind_table_iterator;

// The block of info used by the Symbolizer call backs.
struct DisassembleInfo {
  bool verbose;
  MachOObjectFile *O;
  SectionRef S;
  SymbolAddressMap *AddrMap;
  std::vector<SectionRef> *Sections;
  const char *class_name;
  const char *selector_name;
  char *method;
  char *demangled_name;
  BindTable *bindtable;
};

// SymbolizerGetOpInfo() is the operand information call back function.
// This is called to get the symbolic information for operand(s) of an
// instruction when it is being done.  This routine does this from
// the relocation information, symbol table, etc. That block of information
// is a pointer to the struct DisassembleInfo that was passed when the
// disassembler context was created and passed to back to here when
// called back by the disassembler for instruction operands that could have
// relocation information. The address of the instruction containing operand is
// at the Pc parameter.  The immediate value the operand has is passed in
// op_info->Value and is at Offset past the start of the instruction and has a
// byte Size of 1, 2 or 4. The symbolc information is returned in TagBuf is the
// LLVMOpInfo1 struct defined in the header "llvm-c/Disassembler.h" as symbol
// names and addends of the symbolic expression to add for the operand.  The
// value of TagType is currently 1 (for the LLVMOpInfo1 struct). If symbolic
// information is returned then this function returns 1 else it returns 0.
int SymbolizerGetOpInfo(void *DisInfo, uint64_t Pc, uint64_t Offset,
                        uint64_t Size, int TagType, void *TagBuf) {
  struct DisassembleInfo *info = (struct DisassembleInfo *)DisInfo;
  struct LLVMOpInfo1 *op_info = (struct LLVMOpInfo1 *)TagBuf;
  unsigned int value = op_info->Value;

  // Make sure all fields returned are zero if we don't set them.
  memset((void *)op_info, '\0', sizeof(struct LLVMOpInfo1));
  op_info->Value = value;

  // If the TagType is not the value 1 which it code knows about or if no
  // verbose symbolic information is wanted then just return 0, indicating no
  // information is being returned.
  if (TagType != 1 || info->verbose == false)
    return 0;

  unsigned int Arch = info->O->getArch();
  if (Arch == Triple::x86) {
    return 0;
  } else if (Arch == Triple::x86_64) {
    if (Size != 1 && Size != 2 && Size != 4 && Size != 0)
      return 0;
    // First search the section's relocation entries (if any) for an entry
    // for this section offset.
    uint64_t sect_addr = info->S.getAddress();
    uint64_t sect_offset = (Pc + Offset) - sect_addr;
    bool reloc_found = false;
    DataRefImpl Rel;
    MachO::any_relocation_info RE;
    bool isExtern = false;
    SymbolRef Symbol;
    for (const RelocationRef &Reloc : info->S.relocations()) {
      uint64_t RelocOffset;
      Reloc.getOffset(RelocOffset);
      if (RelocOffset == sect_offset) {
        Rel = Reloc.getRawDataRefImpl();
        RE = info->O->getRelocation(Rel);
        // NOTE: Scattered relocations don't exist on x86_64.
        isExtern = info->O->getPlainRelocationExternal(RE);
        if (isExtern) {
          symbol_iterator RelocSym = Reloc.getSymbol();
          Symbol = *RelocSym;
        }
        reloc_found = true;
        break;
      }
    }
    if (reloc_found && isExtern) {
      // The Value passed in will be adjusted by the Pc if the instruction
      // adds the Pc.  But for x86_64 external relocation entries the Value
      // is the offset from the external symbol.
      if (info->O->getAnyRelocationPCRel(RE))
        op_info->Value -= Pc + Offset + Size;
      StringRef SymName;
      Symbol.getName(SymName);
      const char *name = SymName.data();
      unsigned Type = info->O->getAnyRelocationType(RE);
      if (Type == MachO::X86_64_RELOC_SUBTRACTOR) {
        DataRefImpl RelNext = Rel;
        info->O->moveRelocationNext(RelNext);
        MachO::any_relocation_info RENext = info->O->getRelocation(RelNext);
        unsigned TypeNext = info->O->getAnyRelocationType(RENext);
        bool isExternNext = info->O->getPlainRelocationExternal(RENext);
        unsigned SymbolNum = info->O->getPlainRelocationSymbolNum(RENext);
        if (TypeNext == MachO::X86_64_RELOC_UNSIGNED && isExternNext) {
          op_info->SubtractSymbol.Present = 1;
          op_info->SubtractSymbol.Name = name;
          symbol_iterator RelocSymNext = info->O->getSymbolByIndex(SymbolNum);
          Symbol = *RelocSymNext;
          StringRef SymNameNext;
          Symbol.getName(SymNameNext);
          name = SymNameNext.data();
        }
      }
      // TODO: add the VariantKinds to op_info->VariantKind for relocation types
      // like: X86_64_RELOC_TLV, X86_64_RELOC_GOT_LOAD and X86_64_RELOC_GOT.
      op_info->AddSymbol.Present = 1;
      op_info->AddSymbol.Name = name;
      return 1;
    }
    // TODO:
    // Second search the external relocation entries of a fully linked image
    // (if any) for an entry that matches this segment offset.
    // uint64_t seg_offset = (Pc + Offset);
    return 0;
  } else if (Arch == Triple::arm) {
    return 0;
  } else if (Arch == Triple::aarch64) {
    return 0;
  } else {
    return 0;
  }
}

// GuessCstringPointer is passed the address of what might be a pointer to a
// literal string in a cstring section.  If that address is in a cstring section
// it returns a pointer to that string.  Else it returns nullptr.
const char *GuessCstringPointer(uint64_t ReferenceValue,
                                struct DisassembleInfo *info) {
  uint32_t LoadCommandCount = info->O->getHeader().ncmds;
  MachOObjectFile::LoadCommandInfo Load = info->O->getFirstLoadCommandInfo();
  for (unsigned I = 0;; ++I) {
    if (Load.C.cmd == MachO::LC_SEGMENT_64) {
      MachO::segment_command_64 Seg = info->O->getSegment64LoadCommand(Load);
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section_64 Sec = info->O->getSection64(Load, J);
        uint32_t section_type = Sec.flags & MachO::SECTION_TYPE;
        if (section_type == MachO::S_CSTRING_LITERALS &&
            ReferenceValue >= Sec.addr &&
            ReferenceValue < Sec.addr + Sec.size) {
          uint64_t sect_offset = ReferenceValue - Sec.addr;
          uint64_t object_offset = Sec.offset + sect_offset;
          StringRef MachOContents = info->O->getData();
          uint64_t object_size = MachOContents.size();
          const char *object_addr = (const char *)MachOContents.data();
          if (object_offset < object_size) {
            const char *name = object_addr + object_offset;
            return name;
          } else {
            return nullptr;
          }
        }
      }
    } else if (Load.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command Seg = info->O->getSegmentLoadCommand(Load);
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section Sec = info->O->getSection(Load, J);
        uint32_t section_type = Sec.flags & MachO::SECTION_TYPE;
        if (section_type == MachO::S_CSTRING_LITERALS &&
            ReferenceValue >= Sec.addr &&
            ReferenceValue < Sec.addr + Sec.size) {
          uint64_t sect_offset = ReferenceValue - Sec.addr;
          uint64_t object_offset = Sec.offset + sect_offset;
          StringRef MachOContents = info->O->getData();
          uint64_t object_size = MachOContents.size();
          const char *object_addr = (const char *)MachOContents.data();
          if (object_offset < object_size) {
            const char *name = object_addr + object_offset;
            return name;
          } else {
            return nullptr;
          }
        }
      }
    }
    if (I == LoadCommandCount - 1)
      break;
    else
      Load = info->O->getNextLoadCommandInfo(Load);
  }
  return nullptr;
}

// GuessIndirectSymbol returns the name of the indirect symbol for the
// ReferenceValue passed in or nullptr.  This is used when ReferenceValue maybe
// an address of a symbol stub or a lazy or non-lazy pointer to associate the
// symbol name being referenced by the stub or pointer.
static const char *GuessIndirectSymbol(uint64_t ReferenceValue,
                                       struct DisassembleInfo *info) {
  uint32_t LoadCommandCount = info->O->getHeader().ncmds;
  MachOObjectFile::LoadCommandInfo Load = info->O->getFirstLoadCommandInfo();
  MachO::dysymtab_command Dysymtab = info->O->getDysymtabLoadCommand();
  MachO::symtab_command Symtab = info->O->getSymtabLoadCommand();
  for (unsigned I = 0;; ++I) {
    if (Load.C.cmd == MachO::LC_SEGMENT_64) {
      MachO::segment_command_64 Seg = info->O->getSegment64LoadCommand(Load);
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section_64 Sec = info->O->getSection64(Load, J);
        uint32_t section_type = Sec.flags & MachO::SECTION_TYPE;
        if ((section_type == MachO::S_NON_LAZY_SYMBOL_POINTERS ||
             section_type == MachO::S_LAZY_SYMBOL_POINTERS ||
             section_type == MachO::S_LAZY_DYLIB_SYMBOL_POINTERS ||
             section_type == MachO::S_THREAD_LOCAL_VARIABLE_POINTERS ||
             section_type == MachO::S_SYMBOL_STUBS) &&
            ReferenceValue >= Sec.addr &&
            ReferenceValue < Sec.addr + Sec.size) {
          uint32_t stride;
          if (section_type == MachO::S_SYMBOL_STUBS)
            stride = Sec.reserved2;
          else
            stride = 8;
          if (stride == 0)
            return nullptr;
          uint32_t index = Sec.reserved1 + (ReferenceValue - Sec.addr) / stride;
          if (index < Dysymtab.nindirectsyms) {
            uint32_t indirect_symbol =
                info->O->getIndirectSymbolTableEntry(Dysymtab, index);
            if (indirect_symbol < Symtab.nsyms) {
              symbol_iterator Sym = info->O->getSymbolByIndex(indirect_symbol);
              SymbolRef Symbol = *Sym;
              StringRef SymName;
              Symbol.getName(SymName);
              const char *name = SymName.data();
              return name;
            }
          }
        }
      }
    } else if (Load.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command Seg = info->O->getSegmentLoadCommand(Load);
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section Sec = info->O->getSection(Load, J);
        uint32_t section_type = Sec.flags & MachO::SECTION_TYPE;
        if ((section_type == MachO::S_NON_LAZY_SYMBOL_POINTERS ||
             section_type == MachO::S_LAZY_SYMBOL_POINTERS ||
             section_type == MachO::S_LAZY_DYLIB_SYMBOL_POINTERS ||
             section_type == MachO::S_THREAD_LOCAL_VARIABLE_POINTERS ||
             section_type == MachO::S_SYMBOL_STUBS) &&
            ReferenceValue >= Sec.addr &&
            ReferenceValue < Sec.addr + Sec.size) {
          uint32_t stride;
          if (section_type == MachO::S_SYMBOL_STUBS)
            stride = Sec.reserved2;
          else
            stride = 4;
          if (stride == 0)
            return nullptr;
          uint32_t index = Sec.reserved1 + (ReferenceValue - Sec.addr) / stride;
          if (index < Dysymtab.nindirectsyms) {
            uint32_t indirect_symbol =
                info->O->getIndirectSymbolTableEntry(Dysymtab, index);
            if (indirect_symbol < Symtab.nsyms) {
              symbol_iterator Sym = info->O->getSymbolByIndex(indirect_symbol);
              SymbolRef Symbol = *Sym;
              StringRef SymName;
              Symbol.getName(SymName);
              const char *name = SymName.data();
              return name;
            }
          }
        }
      }
    }
    if (I == LoadCommandCount - 1)
      break;
    else
      Load = info->O->getNextLoadCommandInfo(Load);
  }
  return nullptr;
}

// method_reference() is called passing it the ReferenceName that might be
// a reference it to an Objective-C method call.  If so then it allocates and
// assembles a method call string with the values last seen and saved in
// the DisassembleInfo's class_name and selector_name fields.  This is saved
// into the method field of the info and any previous string is free'ed.
// Then the class_name field in the info is set to nullptr.  The method call
// string is set into ReferenceName and ReferenceType is set to
// LLVMDisassembler_ReferenceType_Out_Objc_Message.  If this not a method call
// then both ReferenceType and ReferenceName are left unchanged.
static void method_reference(struct DisassembleInfo *info,
                             uint64_t *ReferenceType,
                             const char **ReferenceName) {
  if (*ReferenceName != nullptr) {
    if (strcmp(*ReferenceName, "_objc_msgSend") == 0) {
      if (info->selector_name != NULL) {
        if (info->method != nullptr)
          free(info->method);
        if (info->class_name != nullptr) {
          info->method = (char *)malloc(5 + strlen(info->class_name) +
                                        strlen(info->selector_name));
          if (info->method != nullptr) {
            strcpy(info->method, "+[");
            strcat(info->method, info->class_name);
            strcat(info->method, " ");
            strcat(info->method, info->selector_name);
            strcat(info->method, "]");
            *ReferenceName = info->method;
            *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Message;
          }
        } else {
          info->method = (char *)malloc(9 + strlen(info->selector_name));
          if (info->method != nullptr) {
            strcpy(info->method, "-[%rdi ");
            strcat(info->method, info->selector_name);
            strcat(info->method, "]");
            *ReferenceName = info->method;
            *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Message;
          }
        }
        info->class_name = nullptr;
      }
    } else if (strcmp(*ReferenceName, "_objc_msgSendSuper2") == 0) {
      if (info->selector_name != NULL) {
        if (info->method != nullptr)
          free(info->method);
        info->method = (char *)malloc(17 + strlen(info->selector_name));
        if (info->method != nullptr) {
          strcpy(info->method, "-[[%rdi super] ");
          strcat(info->method, info->selector_name);
          strcat(info->method, "]");
          *ReferenceName = info->method;
          *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Message;
        }
        info->class_name = nullptr;
      }
    }
  }
}

// GuessPointerPointer() is passed the address of what might be a pointer to
// a reference to an Objective-C class, selector, message ref or cfstring.
// If so the value of the pointer is returned and one of the booleans are set
// to true.  If not zero is returned and all the booleans are set to false.
static uint64_t GuessPointerPointer(uint64_t ReferenceValue,
                                    struct DisassembleInfo *info,
                                    bool &classref, bool &selref, bool &msgref,
                                    bool &cfstring) {
  classref = false;
  selref = false;
  msgref = false;
  cfstring = false;
  uint32_t LoadCommandCount = info->O->getHeader().ncmds;
  MachOObjectFile::LoadCommandInfo Load = info->O->getFirstLoadCommandInfo();
  for (unsigned I = 0;; ++I) {
    if (Load.C.cmd == MachO::LC_SEGMENT_64) {
      MachO::segment_command_64 Seg = info->O->getSegment64LoadCommand(Load);
      for (unsigned J = 0; J < Seg.nsects; ++J) {
        MachO::section_64 Sec = info->O->getSection64(Load, J);
        if ((strncmp(Sec.sectname, "__objc_selrefs", 16) == 0 ||
             strncmp(Sec.sectname, "__objc_classrefs", 16) == 0 ||
             strncmp(Sec.sectname, "__objc_superrefs", 16) == 0 ||
             strncmp(Sec.sectname, "__objc_msgrefs", 16) == 0 ||
             strncmp(Sec.sectname, "__cfstring", 16) == 0) &&
            ReferenceValue >= Sec.addr &&
            ReferenceValue < Sec.addr + Sec.size) {
          uint64_t sect_offset = ReferenceValue - Sec.addr;
          uint64_t object_offset = Sec.offset + sect_offset;
          StringRef MachOContents = info->O->getData();
          uint64_t object_size = MachOContents.size();
          const char *object_addr = (const char *)MachOContents.data();
          if (object_offset < object_size) {
            uint64_t pointer_value;
            memcpy(&pointer_value, object_addr + object_offset,
                   sizeof(uint64_t));
            if (info->O->isLittleEndian() != sys::IsLittleEndianHost)
              sys::swapByteOrder(pointer_value);
            if (strncmp(Sec.sectname, "__objc_selrefs", 16) == 0)
              selref = true;
            else if (strncmp(Sec.sectname, "__objc_classrefs", 16) == 0 ||
                     strncmp(Sec.sectname, "__objc_superrefs", 16) == 0)
              classref = true;
            else if (strncmp(Sec.sectname, "__objc_msgrefs", 16) == 0 &&
                     ReferenceValue + 8 < Sec.addr + Sec.size) {
              msgref = true;
              memcpy(&pointer_value, object_addr + object_offset + 8,
                     sizeof(uint64_t));
              if (info->O->isLittleEndian() != sys::IsLittleEndianHost)
                sys::swapByteOrder(pointer_value);
            } else if (strncmp(Sec.sectname, "__cfstring", 16) == 0)
              cfstring = true;
            return pointer_value;
          } else {
            return 0;
          }
        }
      }
    }
    // TODO: Look for LC_SEGMENT for 32-bit Mach-O files.
    if (I == LoadCommandCount - 1)
      break;
    else
      Load = info->O->getNextLoadCommandInfo(Load);
  }
  return 0;
}

// get_pointer_64 returns a pointer to the bytes in the object file at the
// Address from a section in the Mach-O file.  And indirectly returns the
// offset into the section, number of bytes left in the section past the offset
// and which section is was being referenced.  If the Address is not in a
// section nullptr is returned.
const char *get_pointer_64(uint64_t Address, uint32_t &offset, uint32_t &left,
                           SectionRef &S, DisassembleInfo *info) {
  offset = 0;
  left = 0;
  S = SectionRef();
  for (unsigned SectIdx = 0; SectIdx != info->Sections->size(); SectIdx++) {
    uint64_t SectAddress = ((*(info->Sections))[SectIdx]).getAddress();
    uint64_t SectSize = ((*(info->Sections))[SectIdx]).getSize();
    if (Address >= SectAddress && Address < SectAddress + SectSize) {
      S = (*(info->Sections))[SectIdx];
      offset = Address - SectAddress;
      left = SectSize - offset;
      StringRef SectContents;
      ((*(info->Sections))[SectIdx]).getContents(SectContents);
      return SectContents.data() + offset;
    }
  }
  return nullptr;
}

// get_symbol_64() returns the name of a symbol (or nullptr) and the address of
// the symbol indirectly through n_value. Based on the relocation information
// for the specified section offset in the specified section reference.
const char *get_symbol_64(uint32_t sect_offset, SectionRef S,
                          DisassembleInfo *info, uint64_t &n_value) {
  n_value = 0;
  if (info->verbose == false)
    return nullptr;

  // See if there is an external relocation entry at the sect_offset.
  bool reloc_found = false;
  DataRefImpl Rel;
  MachO::any_relocation_info RE;
  bool isExtern = false;
  SymbolRef Symbol;
  for (const RelocationRef &Reloc : S.relocations()) {
    uint64_t RelocOffset;
    Reloc.getOffset(RelocOffset);
    if (RelocOffset == sect_offset) {
      Rel = Reloc.getRawDataRefImpl();
      RE = info->O->getRelocation(Rel);
      if (info->O->isRelocationScattered(RE))
        continue;
      isExtern = info->O->getPlainRelocationExternal(RE);
      if (isExtern) {
        symbol_iterator RelocSym = Reloc.getSymbol();
        Symbol = *RelocSym;
      }
      reloc_found = true;
      break;
    }
  }
  // If there is an external relocation entry for a symbol in this section
  // at this section_offset then use that symbol's value for the n_value
  // and return its name.
  const char *SymbolName = nullptr;
  if (reloc_found && isExtern) {
    Symbol.getAddress(n_value);
    StringRef name;
    Symbol.getName(name);
    if (!name.empty()) {
      SymbolName = name.data();
      return SymbolName;
    }
  }

  // TODO: For fully linked images, look through the external relocation
  // entries off the dynamic symtab command. For these the r_offset is from the
  // start of the first writeable segment in the Mach-O file.  So the offset
  // to this section from that segment is passed to this routine by the caller,
  // as the database_offset. Which is the difference of the section's starting
  // address and the first writable segment.
  //
  // NOTE: need add passing the database_offset to this routine.

  // TODO: We did not find an external relocation entry so look up the
  // ReferenceValue as an address of a symbol and if found return that symbol's
  // name.
  //
  // NOTE: need add passing the ReferenceValue to this routine.  Then that code
  // would simply be this:
  //
  // if (ReferenceValue != 0xffffffffffffffffLLU &&
  //     ReferenceValue != 0xfffffffffffffffeLLU) {
  //   StringRef name = info->AddrMap->lookup(ReferenceValue);
  //   if (!name.empty())
  //     SymbolName = name.data();
  // }

  return SymbolName;
}

// These are structs in the Objective-C meta data and read to produce the
// comments for disassembly.  While these are part of the ABI they are no
// public defintions.  So the are here not in include/llvm/Support/MachO.h .

// The cfstring object in a 64-bit Mach-O file.
struct cfstring64_t {
  uint64_t isa;        // class64_t * (64-bit pointer)
  uint64_t flags;      // flag bits
  uint64_t characters; // char * (64-bit pointer)
  uint64_t length;     // number of non-NULL characters in above
};

// The class object in a 64-bit Mach-O file.
struct class64_t {
  uint64_t isa;        // class64_t * (64-bit pointer)
  uint64_t superclass; // class64_t * (64-bit pointer)
  uint64_t cache;      // Cache (64-bit pointer)
  uint64_t vtable;     // IMP * (64-bit pointer)
  uint64_t data;       // class_ro64_t * (64-bit pointer)
};

struct class_ro64_t {
  uint32_t flags;
  uint32_t instanceStart;
  uint32_t instanceSize;
  uint32_t reserved;
  uint64_t ivarLayout;     // const uint8_t * (64-bit pointer)
  uint64_t name;           // const char * (64-bit pointer)
  uint64_t baseMethods;    // const method_list_t * (64-bit pointer)
  uint64_t baseProtocols;  // const protocol_list_t * (64-bit pointer)
  uint64_t ivars;          // const ivar_list_t * (64-bit pointer)
  uint64_t weakIvarLayout; // const uint8_t * (64-bit pointer)
  uint64_t baseProperties; // const struct objc_property_list (64-bit pointer)
};

inline void swapStruct(struct cfstring64_t &cfs) {
  sys::swapByteOrder(cfs.isa);
  sys::swapByteOrder(cfs.flags);
  sys::swapByteOrder(cfs.characters);
  sys::swapByteOrder(cfs.length);
}

inline void swapStruct(struct class64_t &c) {
  sys::swapByteOrder(c.isa);
  sys::swapByteOrder(c.superclass);
  sys::swapByteOrder(c.cache);
  sys::swapByteOrder(c.vtable);
  sys::swapByteOrder(c.data);
}

inline void swapStruct(struct class_ro64_t &cro) {
  sys::swapByteOrder(cro.flags);
  sys::swapByteOrder(cro.instanceStart);
  sys::swapByteOrder(cro.instanceSize);
  sys::swapByteOrder(cro.reserved);
  sys::swapByteOrder(cro.ivarLayout);
  sys::swapByteOrder(cro.name);
  sys::swapByteOrder(cro.baseMethods);
  sys::swapByteOrder(cro.baseProtocols);
  sys::swapByteOrder(cro.ivars);
  sys::swapByteOrder(cro.weakIvarLayout);
  sys::swapByteOrder(cro.baseProperties);
}

static const char *get_dyld_bind_info_symbolname(uint64_t ReferenceValue,
                                                 struct DisassembleInfo *info);

// get_objc2_64bit_class_name() is used for disassembly and is passed a pointer
// to an Objective-C class and returns the class name.  It is also passed the
// address of the pointer, so when the pointer is zero as it can be in an .o
// file, that is used to look for an external relocation entry with a symbol
// name.
const char *get_objc2_64bit_class_name(uint64_t pointer_value,
                                       uint64_t ReferenceValue,
                                       struct DisassembleInfo *info) {
  const char *r;
  uint32_t offset, left;
  SectionRef S;

  // The pointer_value can be 0 in an object file and have a relocation
  // entry for the class symbol at the ReferenceValue (the address of the
  // pointer).
  if (pointer_value == 0) {
    r = get_pointer_64(ReferenceValue, offset, left, S, info);
    if (r == nullptr || left < sizeof(uint64_t))
      return nullptr;
    uint64_t n_value;
    const char *symbol_name = get_symbol_64(offset, S, info, n_value);
    if (symbol_name == nullptr)
      return nullptr;
    const char *class_name = strrchr(symbol_name, '$');
    if (class_name != nullptr && class_name[1] == '_' && class_name[2] != '\0')
      return class_name + 2;
    else
      return nullptr;
  }

  // The case were the pointer_value is non-zero and points to a class defined
  // in this Mach-O file.
  r = get_pointer_64(pointer_value, offset, left, S, info);
  if (r == nullptr || left < sizeof(struct class64_t))
    return nullptr;
  struct class64_t c;
  memcpy(&c, r, sizeof(struct class64_t));
  if (info->O->isLittleEndian() != sys::IsLittleEndianHost)
    swapStruct(c);
  if (c.data == 0)
    return nullptr;
  r = get_pointer_64(c.data, offset, left, S, info);
  if (r == nullptr || left < sizeof(struct class_ro64_t))
    return nullptr;
  struct class_ro64_t cro;
  memcpy(&cro, r, sizeof(struct class_ro64_t));
  if (info->O->isLittleEndian() != sys::IsLittleEndianHost)
    swapStruct(cro);
  if (cro.name == 0)
    return nullptr;
  const char *name = get_pointer_64(cro.name, offset, left, S, info);
  return name;
}

// get_objc2_64bit_cfstring_name is used for disassembly and is passed a
// pointer to a cfstring and returns its name or nullptr.
const char *get_objc2_64bit_cfstring_name(uint64_t ReferenceValue,
                                          struct DisassembleInfo *info) {
  const char *r, *name;
  uint32_t offset, left;
  SectionRef S;
  struct cfstring64_t cfs;
  uint64_t cfs_characters;

  r = get_pointer_64(ReferenceValue, offset, left, S, info);
  if (r == nullptr || left < sizeof(struct cfstring64_t))
    return nullptr;
  memcpy(&cfs, r, sizeof(struct cfstring64_t));
  if (info->O->isLittleEndian() != sys::IsLittleEndianHost)
    swapStruct(cfs);
  if (cfs.characters == 0) {
    uint64_t n_value;
    const char *symbol_name = get_symbol_64(
        offset + offsetof(struct cfstring64_t, characters), S, info, n_value);
    if (symbol_name == nullptr)
      return nullptr;
    cfs_characters = n_value;
  } else
    cfs_characters = cfs.characters;
  name = get_pointer_64(cfs_characters, offset, left, S, info);

  return name;
}

// get_objc2_64bit_selref() is used for disassembly and is passed a the address
// of a pointer to an Objective-C selector reference when the pointer value is
// zero as in a .o file and is likely to have a external relocation entry with
// who's symbol's n_value is the real pointer to the selector name.  If that is
// the case the real pointer to the selector name is returned else 0 is
// returned
uint64_t get_objc2_64bit_selref(uint64_t ReferenceValue,
                                struct DisassembleInfo *info) {
  uint32_t offset, left;
  SectionRef S;

  const char *r = get_pointer_64(ReferenceValue, offset, left, S, info);
  if (r == nullptr || left < sizeof(uint64_t))
    return 0;
  uint64_t n_value;
  const char *symbol_name = get_symbol_64(offset, S, info, n_value);
  if (symbol_name == nullptr)
    return 0;
  return n_value;
}

// GuessLiteralPointer returns a string which for the item in the Mach-O file
// for the address passed in as ReferenceValue for printing as a comment with
// the instruction and also returns the corresponding type of that item
// indirectly through ReferenceType.
//
// If ReferenceValue is an address of literal cstring then a pointer to the
// cstring is returned and ReferenceType is set to
// LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr .
//
// If ReferenceValue is an address of an Objective-C CFString, Selector ref or
// Class ref that name is returned and the ReferenceType is set accordingly.
//
// Lastly, literals which are Symbol address in a literal pool are looked for
// and if found the symbol name is returned and ReferenceType is set to
// LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr .
//
// If there is no item in the Mach-O file for the address passed in as
// ReferenceValue nullptr is returned and ReferenceType is unchanged.
const char *GuessLiteralPointer(uint64_t ReferenceValue, uint64_t ReferencePC,
                                uint64_t *ReferenceType,
                                struct DisassembleInfo *info) {
  // TODO: This rouine's code and the routines it calls are only work with
  // x86_64 Mach-O files for now.
  unsigned int Arch = info->O->getArch();
  if (Arch != Triple::x86_64)
    return nullptr;

  // First see if there is an external relocation entry at the ReferencePC.
  uint64_t sect_addr = info->S.getAddress();
  uint64_t sect_offset = ReferencePC - sect_addr;
  bool reloc_found = false;
  DataRefImpl Rel;
  MachO::any_relocation_info RE;
  bool isExtern = false;
  SymbolRef Symbol;
  for (const RelocationRef &Reloc : info->S.relocations()) {
    uint64_t RelocOffset;
    Reloc.getOffset(RelocOffset);
    if (RelocOffset == sect_offset) {
      Rel = Reloc.getRawDataRefImpl();
      RE = info->O->getRelocation(Rel);
      if (info->O->isRelocationScattered(RE))
        continue;
      isExtern = info->O->getPlainRelocationExternal(RE);
      if (isExtern) {
        symbol_iterator RelocSym = Reloc.getSymbol();
        Symbol = *RelocSym;
      }
      reloc_found = true;
      break;
    }
  }
  // If there is an external relocation entry for a symbol in a section
  // then used that symbol's value for the value of the reference.
  if (reloc_found && isExtern) {
    if (info->O->getAnyRelocationPCRel(RE)) {
      unsigned Type = info->O->getAnyRelocationType(RE);
      if (Type == MachO::X86_64_RELOC_SIGNED) {
        Symbol.getAddress(ReferenceValue);
      }
    }
  }

  // Look for literals such as Objective-C CFStrings refs, Selector refs,
  // Message refs and Class refs.
  bool classref, selref, msgref, cfstring;
  uint64_t pointer_value = GuessPointerPointer(ReferenceValue, info, classref,
                                               selref, msgref, cfstring);
  if (classref == true && pointer_value == 0) {
    // Note the ReferenceValue is a pointer into the __objc_classrefs section.
    // And the pointer_value in that section is typically zero as it will be
    // set by dyld as part of the "bind information".
    const char *name = get_dyld_bind_info_symbolname(ReferenceValue, info);
    if (name != nullptr) {
      *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref;
      const char *class_name = strrchr(name, '$');
      if (class_name != nullptr && class_name[1] == '_' &&
          class_name[2] != '\0') {
        info->class_name = class_name + 2;
        return name;
      }
    }
  }

  if (classref == true) {
    *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref;
    const char *name =
        get_objc2_64bit_class_name(pointer_value, ReferenceValue, info);
    if (name != nullptr)
      info->class_name = name;
    else
      name = "bad class ref";
    return name;
  }

  if (cfstring == true) {
    *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref;
    const char *name = get_objc2_64bit_cfstring_name(ReferenceValue, info);
    return name;
  }

  if (selref == true && pointer_value == 0)
    pointer_value = get_objc2_64bit_selref(ReferenceValue, info);

  if (pointer_value != 0)
    ReferenceValue = pointer_value;

  const char *name = GuessCstringPointer(ReferenceValue, info);
  if (name) {
    if (pointer_value != 0 && selref == true) {
      *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref;
      info->selector_name = name;
    } else if (pointer_value != 0 && msgref == true) {
      info->class_name = nullptr;
      *ReferenceType = LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref;
      info->selector_name = name;
    } else
      *ReferenceType = LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr;
    return name;
  }

  // Lastly look for an indirect symbol with this ReferenceValue which is in
  // a literal pool.  If found return that symbol name.
  name = GuessIndirectSymbol(ReferenceValue, info);
  if (name) {
    *ReferenceType = LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr;
    return name;
  }

  return nullptr;
}

// SymbolizerSymbolLookUp is the symbol lookup function passed when creating
// the Symbolizer.  It looks up the ReferenceValue using the info passed via the
// pointer to the struct DisassembleInfo that was passed when MCSymbolizer
// is created and returns the symbol name that matches the ReferenceValue or
// nullptr if none.  The ReferenceType is passed in for the IN type of
// reference the instruction is making from the values in defined in the header
// "llvm-c/Disassembler.h".  On return the ReferenceType can set to a specific
// Out type and the ReferenceName will also be set which is added as a comment
// to the disassembled instruction.
//
#if HAVE_CXXABI_H
// If the symbol name is a C++ mangled name then the demangled name is
// returned through ReferenceName and ReferenceType is set to
// LLVMDisassembler_ReferenceType_DeMangled_Name .
#endif
//
// When this is called to get a symbol name for a branch target then the
// ReferenceType will be LLVMDisassembler_ReferenceType_In_Branch and then
// SymbolValue will be looked for in the indirect symbol table to determine if
// it is an address for a symbol stub.  If so then the symbol name for that
// stub is returned indirectly through ReferenceName and then ReferenceType is
// set to LLVMDisassembler_ReferenceType_Out_SymbolStub.
//
// When this is called with an value loaded via a PC relative load then
// ReferenceType will be LLVMDisassembler_ReferenceType_In_PCrel_Load then the
// SymbolValue is checked to be an address of literal pointer, symbol pointer,
// or an Objective-C meta data reference.  If so the output ReferenceType is
// set to correspond to that as well as setting the ReferenceName.
const char *SymbolizerSymbolLookUp(void *DisInfo, uint64_t ReferenceValue,
                                   uint64_t *ReferenceType,
                                   uint64_t ReferencePC,
                                   const char **ReferenceName) {
  struct DisassembleInfo *info = (struct DisassembleInfo *)DisInfo;
  // If no verbose symbolic information is wanted then just return nullptr.
  if (info->verbose == false) {
    *ReferenceName = nullptr;
    *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
    return nullptr;
  }

  const char *SymbolName = nullptr;
  if (ReferenceValue != 0xffffffffffffffffULL &&
      ReferenceValue != 0xfffffffffffffffeULL) {
    StringRef name = info->AddrMap->lookup(ReferenceValue);
    if (!name.empty())
      SymbolName = name.data();
  }

  if (*ReferenceType == LLVMDisassembler_ReferenceType_In_Branch) {
    *ReferenceName = GuessIndirectSymbol(ReferenceValue, info);
    if (*ReferenceName != nullptr) {
      method_reference(info, ReferenceType, ReferenceName);
      if (*ReferenceType != LLVMDisassembler_ReferenceType_Out_Objc_Message)
        *ReferenceType = LLVMDisassembler_ReferenceType_Out_SymbolStub;
    } else
#if HAVE_CXXABI_H
        if (SymbolName != nullptr && strncmp(SymbolName, "__Z", 3) == 0) {
      if (info->demangled_name != nullptr)
        free(info->demangled_name);
      int status;
      info->demangled_name =
          abi::__cxa_demangle(SymbolName + 1, nullptr, nullptr, &status);
      if (info->demangled_name != nullptr) {
        *ReferenceName = info->demangled_name;
        *ReferenceType = LLVMDisassembler_ReferenceType_DeMangled_Name;
      } else
        *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
    } else
#endif
      *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  } else if (*ReferenceType == LLVMDisassembler_ReferenceType_In_PCrel_Load) {
    *ReferenceName =
        GuessLiteralPointer(ReferenceValue, ReferencePC, ReferenceType, info);
    if (*ReferenceName)
      method_reference(info, ReferenceType, ReferenceName);
    else
      *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  }
#if HAVE_CXXABI_H
  else if (SymbolName != nullptr && strncmp(SymbolName, "__Z", 3) == 0) {
    if (info->demangled_name != nullptr)
      free(info->demangled_name);
    int status;
    info->demangled_name =
        abi::__cxa_demangle(SymbolName + 1, nullptr, nullptr, &status);
    if (info->demangled_name != nullptr) {
      *ReferenceName = info->demangled_name;
      *ReferenceType = LLVMDisassembler_ReferenceType_DeMangled_Name;
    }
  }
#endif
  else {
    *ReferenceName = nullptr;
    *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  }

  return SymbolName;
}

//
// This is the memory object used by DisAsm->getInstruction() which has its
// BasePC.  This then allows the 'address' parameter to getInstruction() to
// be the actual PC of the instruction.  Then when a branch dispacement is
// added to the PC of an instruction, the 'ReferenceValue' passed to the
// SymbolizerSymbolLookUp() routine is the correct target addresses.  As in
// the case of a fully linked Mach-O file where a section being disassembled
// generally not linked at address zero.
//
class DisasmMemoryObject : public MemoryObject {
  const uint8_t *Bytes;
  uint64_t Size;
  uint64_t BasePC;

public:
  DisasmMemoryObject(const uint8_t *bytes, uint64_t size, uint64_t basePC)
      : Bytes(bytes), Size(size), BasePC(basePC) {}

  uint64_t getBase() const override { return BasePC; }
  uint64_t getExtent() const override { return Size; }

  int readByte(uint64_t Addr, uint8_t *Byte) const override {
    if (Addr - BasePC >= Size)
      return -1;
    *Byte = Bytes[Addr - BasePC];
    return 0;
  }
};

/// \brief Emits the comments that are stored in the CommentStream.
/// Each comment in the CommentStream must end with a newline.
static void emitComments(raw_svector_ostream &CommentStream,
                         SmallString<128> &CommentsToEmit,
                         formatted_raw_ostream &FormattedOS,
                         const MCAsmInfo &MAI) {
  // Flush the stream before taking its content.
  CommentStream.flush();
  StringRef Comments = CommentsToEmit.str();
  // Get the default information for printing a comment.
  const char *CommentBegin = MAI.getCommentString();
  unsigned CommentColumn = MAI.getCommentColumn();
  bool IsFirst = true;
  while (!Comments.empty()) {
    if (!IsFirst)
      FormattedOS << '\n';
    // Emit a line of comments.
    FormattedOS.PadToColumn(CommentColumn);
    size_t Position = Comments.find('\n');
    FormattedOS << CommentBegin << ' ' << Comments.substr(0, Position);
    // Move after the newline character.
    Comments = Comments.substr(Position + 1);
    IsFirst = false;
  }
  FormattedOS.flush();

  // Tell the comment stream that the vector changed underneath it.
  CommentsToEmit.clear();
  CommentStream.resync();
}

static void DisassembleInputMachO2(StringRef Filename,
                                   MachOObjectFile *MachOOF) {
  const char *McpuDefault = nullptr;
  const Target *ThumbTarget = nullptr;
  const Target *TheTarget = GetTarget(MachOOF, &McpuDefault, &ThumbTarget);
  if (!TheTarget) {
    // GetTarget prints out stuff.
    return;
  }
  if (MCPU.empty() && McpuDefault)
    MCPU = McpuDefault;

  std::unique_ptr<const MCInstrInfo> InstrInfo(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCInstrAnalysis> InstrAnalysis(
      TheTarget->createMCInstrAnalysis(InstrInfo.get()));
  std::unique_ptr<const MCInstrInfo> ThumbInstrInfo;
  std::unique_ptr<MCInstrAnalysis> ThumbInstrAnalysis;
  if (ThumbTarget) {
    ThumbInstrInfo.reset(ThumbTarget->createMCInstrInfo());
    ThumbInstrAnalysis.reset(
        ThumbTarget->createMCInstrAnalysis(ThumbInstrInfo.get()));
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Set up disassembler.
  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, FeaturesStr));
  MCContext Ctx(AsmInfo.get(), MRI.get(), nullptr);
  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, Ctx));
  std::unique_ptr<MCSymbolizer> Symbolizer;
  struct DisassembleInfo SymbolizerInfo;
  std::unique_ptr<MCRelocationInfo> RelInfo(
      TheTarget->createMCRelocationInfo(TripleName, Ctx));
  if (RelInfo) {
    Symbolizer.reset(TheTarget->createMCSymbolizer(
        TripleName, SymbolizerGetOpInfo, SymbolizerSymbolLookUp,
        &SymbolizerInfo, &Ctx, RelInfo.release()));
    DisAsm->setSymbolizer(std::move(Symbolizer));
  }
  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      AsmPrinterVariant, *AsmInfo, *InstrInfo, *MRI, *STI));
  // Set the display preference for hex vs. decimal immediates.
  IP->setPrintImmHex(PrintImmHex);
  // Comment stream and backing vector.
  SmallString<128> CommentsToEmit;
  raw_svector_ostream CommentStream(CommentsToEmit);
  IP->setCommentStream(CommentStream);

  if (!InstrAnalysis || !AsmInfo || !STI || !DisAsm || !IP) {
    errs() << "error: couldn't initialize disassembler for target "
           << TripleName << '\n';
    return;
  }

  // Set up thumb disassembler.
  std::unique_ptr<const MCRegisterInfo> ThumbMRI;
  std::unique_ptr<const MCAsmInfo> ThumbAsmInfo;
  std::unique_ptr<const MCSubtargetInfo> ThumbSTI;
  std::unique_ptr<const MCDisassembler> ThumbDisAsm;
  std::unique_ptr<MCInstPrinter> ThumbIP;
  std::unique_ptr<MCContext> ThumbCtx;
  if (ThumbTarget) {
    ThumbMRI.reset(ThumbTarget->createMCRegInfo(ThumbTripleName));
    ThumbAsmInfo.reset(
        ThumbTarget->createMCAsmInfo(*ThumbMRI, ThumbTripleName));
    ThumbSTI.reset(
        ThumbTarget->createMCSubtargetInfo(ThumbTripleName, MCPU, FeaturesStr));
    ThumbCtx.reset(new MCContext(ThumbAsmInfo.get(), ThumbMRI.get(), nullptr));
    ThumbDisAsm.reset(ThumbTarget->createMCDisassembler(*ThumbSTI, *ThumbCtx));
    // TODO: add MCSymbolizer here for the ThumbTarget like above for TheTarget.
    int ThumbAsmPrinterVariant = ThumbAsmInfo->getAssemblerDialect();
    ThumbIP.reset(ThumbTarget->createMCInstPrinter(
        ThumbAsmPrinterVariant, *ThumbAsmInfo, *ThumbInstrInfo, *ThumbMRI,
        *ThumbSTI));
    // Set the display preference for hex vs. decimal immediates.
    ThumbIP->setPrintImmHex(PrintImmHex);
  }

  if (ThumbTarget && (!ThumbInstrAnalysis || !ThumbAsmInfo || !ThumbSTI ||
                      !ThumbDisAsm || !ThumbIP)) {
    errs() << "error: couldn't initialize disassembler for target "
           << ThumbTripleName << '\n';
    return;
  }

  outs() << '\n' << Filename << ":\n\n";

  MachO::mach_header Header = MachOOF->getHeader();

  // FIXME: Using the -cfg command line option, this code used to be able to
  // annotate relocations with the referenced symbol's name, and if this was
  // inside a __[cf]string section, the data it points to. This is now replaced
  // by the upcoming MCSymbolizer, which needs the appropriate setup done above.
  std::vector<SectionRef> Sections;
  std::vector<SymbolRef> Symbols;
  SmallVector<uint64_t, 8> FoundFns;
  uint64_t BaseSegmentAddress;

  getSectionsAndSymbols(Header, MachOOF, Sections, Symbols, FoundFns,
                        BaseSegmentAddress);

  // Sort the symbols by address, just in case they didn't come in that way.
  std::sort(Symbols.begin(), Symbols.end(), SymbolSorter());

  // Build a data in code table that is sorted on by the address of each entry.
  uint64_t BaseAddress = 0;
  if (Header.filetype == MachO::MH_OBJECT)
    BaseAddress = Sections[0].getAddress();
  else
    BaseAddress = BaseSegmentAddress;
  DiceTable Dices;
  for (dice_iterator DI = MachOOF->begin_dices(), DE = MachOOF->end_dices();
       DI != DE; ++DI) {
    uint32_t Offset;
    DI->getOffset(Offset);
    Dices.push_back(std::make_pair(BaseAddress + Offset, *DI));
  }
  array_pod_sort(Dices.begin(), Dices.end());

#ifndef NDEBUG
  raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
  raw_ostream &DebugOut = nulls();
#endif

  std::unique_ptr<DIContext> diContext;
  ObjectFile *DbgObj = MachOOF;
  // Try to find debug info and set up the DIContext for it.
  if (UseDbg) {
    // A separate DSym file path was specified, parse it as a macho file,
    // get the sections and supply it to the section name parsing machinery.
    if (!DSYMFile.empty()) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
          MemoryBuffer::getFileOrSTDIN(DSYMFile);
      if (std::error_code EC = BufOrErr.getError()) {
        errs() << "llvm-objdump: " << Filename << ": " << EC.message() << '\n';
        return;
      }
      DbgObj =
          ObjectFile::createMachOObjectFile(BufOrErr.get()->getMemBufferRef())
              .get()
              .release();
    }

    // Setup the DIContext
    diContext.reset(DIContext::getDWARFContext(*DbgObj));
  }

  for (unsigned SectIdx = 0; SectIdx != Sections.size(); SectIdx++) {

    bool SectIsText = Sections[SectIdx].isText();
    if (SectIsText == false)
      continue;

    StringRef SectName;
    if (Sections[SectIdx].getName(SectName) || SectName != "__text")
      continue; // Skip non-text sections

    DataRefImpl DR = Sections[SectIdx].getRawDataRefImpl();

    StringRef SegmentName = MachOOF->getSectionFinalSegmentName(DR);
    if (SegmentName != "__TEXT")
      continue;

    StringRef Bytes;
    Sections[SectIdx].getContents(Bytes);
    uint64_t SectAddress = Sections[SectIdx].getAddress();
    DisasmMemoryObject MemoryObject((const uint8_t *)Bytes.data(), Bytes.size(),
                                    SectAddress);
    bool symbolTableWorked = false;

    // Parse relocations.
    std::vector<std::pair<uint64_t, SymbolRef>> Relocs;
    for (const RelocationRef &Reloc : Sections[SectIdx].relocations()) {
      uint64_t RelocOffset;
      Reloc.getOffset(RelocOffset);
      uint64_t SectionAddress = Sections[SectIdx].getAddress();
      RelocOffset -= SectionAddress;

      symbol_iterator RelocSym = Reloc.getSymbol();

      Relocs.push_back(std::make_pair(RelocOffset, *RelocSym));
    }
    array_pod_sort(Relocs.begin(), Relocs.end());

    // Create a map of symbol addresses to symbol names for use by
    // the SymbolizerSymbolLookUp() routine.
    SymbolAddressMap AddrMap;
    for (const SymbolRef &Symbol : MachOOF->symbols()) {
      SymbolRef::Type ST;
      Symbol.getType(ST);
      if (ST == SymbolRef::ST_Function || ST == SymbolRef::ST_Data ||
          ST == SymbolRef::ST_Other) {
        uint64_t Address;
        Symbol.getAddress(Address);
        StringRef SymName;
        Symbol.getName(SymName);
        AddrMap[Address] = SymName;
      }
    }
    // Set up the block of info used by the Symbolizer call backs.
    SymbolizerInfo.verbose = true;
    SymbolizerInfo.O = MachOOF;
    SymbolizerInfo.S = Sections[SectIdx];
    SymbolizerInfo.AddrMap = &AddrMap;
    SymbolizerInfo.Sections = &Sections;
    SymbolizerInfo.class_name = nullptr;
    SymbolizerInfo.selector_name = nullptr;
    SymbolizerInfo.method = nullptr;
    SymbolizerInfo.demangled_name = nullptr;
    SymbolizerInfo.bindtable = nullptr;

    // Disassemble symbol by symbol.
    for (unsigned SymIdx = 0; SymIdx != Symbols.size(); SymIdx++) {
      StringRef SymName;
      Symbols[SymIdx].getName(SymName);

      SymbolRef::Type ST;
      Symbols[SymIdx].getType(ST);
      if (ST != SymbolRef::ST_Function)
        continue;

      // Make sure the symbol is defined in this section.
      bool containsSym = Sections[SectIdx].containsSymbol(Symbols[SymIdx]);
      if (!containsSym)
        continue;

      // Start at the address of the symbol relative to the section's address.
      uint64_t Start = 0;
      uint64_t SectionAddress = Sections[SectIdx].getAddress();
      Symbols[SymIdx].getAddress(Start);
      Start -= SectionAddress;

      // Stop disassembling either at the beginning of the next symbol or at
      // the end of the section.
      bool containsNextSym = false;
      uint64_t NextSym = 0;
      uint64_t NextSymIdx = SymIdx + 1;
      while (Symbols.size() > NextSymIdx) {
        SymbolRef::Type NextSymType;
        Symbols[NextSymIdx].getType(NextSymType);
        if (NextSymType == SymbolRef::ST_Function) {
          containsNextSym =
              Sections[SectIdx].containsSymbol(Symbols[NextSymIdx]);
          Symbols[NextSymIdx].getAddress(NextSym);
          NextSym -= SectionAddress;
          break;
        }
        ++NextSymIdx;
      }

      uint64_t SectSize = Sections[SectIdx].getSize();
      uint64_t End = containsNextSym ? NextSym : SectSize;
      uint64_t Size;

      symbolTableWorked = true;
      DisasmMemoryObject SectionMemoryObject((const uint8_t *)Bytes.data() +
                                                 Start,
                                             End - Start, SectAddress + Start);

      DataRefImpl Symb = Symbols[SymIdx].getRawDataRefImpl();
      bool isThumb =
          (MachOOF->getSymbolFlags(Symb) & SymbolRef::SF_Thumb) && ThumbTarget;

      outs() << SymName << ":\n";
      DILineInfo lastLine;
      for (uint64_t Index = Start; Index < End; Index += Size) {
        MCInst Inst;

        uint64_t PC = SectAddress + Index;
        if (FullLeadingAddr) {
          if (MachOOF->is64Bit())
            outs() << format("%016" PRIx64, PC);
          else
            outs() << format("%08" PRIx64, PC);
        } else {
          outs() << format("%8" PRIx64 ":", PC);
        }
        if (!NoShowRawInsn)
          outs() << "\t";

        // Check the data in code table here to see if this is data not an
        // instruction to be disassembled.
        DiceTable Dice;
        Dice.push_back(std::make_pair(PC, DiceRef()));
        dice_table_iterator DTI =
            std::search(Dices.begin(), Dices.end(), Dice.begin(), Dice.end(),
                        compareDiceTableEntries);
        if (DTI != Dices.end()) {
          uint16_t Length;
          DTI->second.getLength(Length);
          DumpBytes(StringRef(Bytes.data() + Index, Length));
          uint16_t Kind;
          DTI->second.getKind(Kind);
          DumpDataInCode(Bytes.data() + Index, Length, Kind);
          continue;
        }

        SmallVector<char, 64> AnnotationsBytes;
        raw_svector_ostream Annotations(AnnotationsBytes);

        bool gotInst;
        if (isThumb)
          gotInst = ThumbDisAsm->getInstruction(Inst, Size, SectionMemoryObject,
                                                PC, DebugOut, Annotations);
        else
          gotInst = DisAsm->getInstruction(Inst, Size, SectionMemoryObject, PC,
                                           DebugOut, Annotations);
        if (gotInst) {
          if (!NoShowRawInsn) {
            DumpBytes(StringRef(Bytes.data() + Index, Size));
          }
          formatted_raw_ostream FormattedOS(outs());
          Annotations.flush();
          StringRef AnnotationsStr = Annotations.str();
          if (isThumb)
            ThumbIP->printInst(&Inst, FormattedOS, AnnotationsStr);
          else
            IP->printInst(&Inst, FormattedOS, AnnotationsStr);
          emitComments(CommentStream, CommentsToEmit, FormattedOS, *AsmInfo);

          // Print debug info.
          if (diContext) {
            DILineInfo dli = diContext->getLineInfoForAddress(PC);
            // Print valid line info if it changed.
            if (dli != lastLine && dli.Line != 0)
              outs() << "\t## " << dli.FileName << ':' << dli.Line << ':'
                     << dli.Column;
            lastLine = dli;
          }
          outs() << "\n";
        } else {
          unsigned int Arch = MachOOF->getArch();
          if (Arch == Triple::x86_64 || Arch == Triple::x86) {
            outs() << format("\t.byte 0x%02x #bad opcode\n",
                             *(Bytes.data() + Index) & 0xff);
            Size = 1; // skip exactly one illegible byte and move on.
          } else {
            errs() << "llvm-objdump: warning: invalid instruction encoding\n";
            if (Size == 0)
              Size = 1; // skip illegible bytes
          }
        }
      }
    }
    if (!symbolTableWorked) {
      // Reading the symbol table didn't work, disassemble the whole section.
      uint64_t SectAddress = Sections[SectIdx].getAddress();
      uint64_t SectSize = Sections[SectIdx].getSize();
      uint64_t InstSize;
      for (uint64_t Index = 0; Index < SectSize; Index += InstSize) {
        MCInst Inst;

        uint64_t PC = SectAddress + Index;
        if (DisAsm->getInstruction(Inst, InstSize, MemoryObject, PC, DebugOut,
                                   nulls())) {
          if (FullLeadingAddr) {
            if (MachOOF->is64Bit())
              outs() << format("%016" PRIx64, PC);
            else
              outs() << format("%08" PRIx64, PC);
          } else {
            outs() << format("%8" PRIx64 ":", PC);
          }
          if (!NoShowRawInsn) {
            outs() << "\t";
            DumpBytes(StringRef(Bytes.data() + Index, InstSize));
          }
          IP->printInst(&Inst, outs(), "");
          outs() << "\n";
        } else {
          unsigned int Arch = MachOOF->getArch();
          if (Arch == Triple::x86_64 || Arch == Triple::x86) {
            outs() << format("\t.byte 0x%02x #bad opcode\n",
                             *(Bytes.data() + Index) & 0xff);
            InstSize = 1; // skip exactly one illegible byte and move on.
          } else {
            errs() << "llvm-objdump: warning: invalid instruction encoding\n";
            if (InstSize == 0)
              InstSize = 1; // skip illegible bytes
          }
        }
      }
    }
    if (SymbolizerInfo.method != nullptr)
      free(SymbolizerInfo.method);
    if (SymbolizerInfo.demangled_name != nullptr)
      free(SymbolizerInfo.demangled_name);
    if (SymbolizerInfo.bindtable != nullptr)
      delete SymbolizerInfo.bindtable;
  }
}

//===----------------------------------------------------------------------===//
// __compact_unwind section dumping
//===----------------------------------------------------------------------===//

namespace {

template <typename T> static uint64_t readNext(const char *&Buf) {
  using llvm::support::little;
  using llvm::support::unaligned;

  uint64_t Val = support::endian::read<T, little, unaligned>(Buf);
  Buf += sizeof(T);
  return Val;
}

struct CompactUnwindEntry {
  uint32_t OffsetInSection;

  uint64_t FunctionAddr;
  uint32_t Length;
  uint32_t CompactEncoding;
  uint64_t PersonalityAddr;
  uint64_t LSDAAddr;

  RelocationRef FunctionReloc;
  RelocationRef PersonalityReloc;
  RelocationRef LSDAReloc;

  CompactUnwindEntry(StringRef Contents, unsigned Offset, bool Is64)
      : OffsetInSection(Offset) {
    if (Is64)
      read<uint64_t>(Contents.data() + Offset);
    else
      read<uint32_t>(Contents.data() + Offset);
  }

private:
  template <typename UIntPtr> void read(const char *Buf) {
    FunctionAddr = readNext<UIntPtr>(Buf);
    Length = readNext<uint32_t>(Buf);
    CompactEncoding = readNext<uint32_t>(Buf);
    PersonalityAddr = readNext<UIntPtr>(Buf);
    LSDAAddr = readNext<UIntPtr>(Buf);
  }
};
}

/// Given a relocation from __compact_unwind, consisting of the RelocationRef
/// and data being relocated, determine the best base Name and Addend to use for
/// display purposes.
///
/// 1. An Extern relocation will directly reference a symbol (and the data is
///    then already an addend), so use that.
/// 2. Otherwise the data is an offset in the object file's layout; try to find
//     a symbol before it in the same section, and use the offset from there.
/// 3. Finally, if all that fails, fall back to an offset from the start of the
///    referenced section.
static void findUnwindRelocNameAddend(const MachOObjectFile *Obj,
                                      std::map<uint64_t, SymbolRef> &Symbols,
                                      const RelocationRef &Reloc, uint64_t Addr,
                                      StringRef &Name, uint64_t &Addend) {
  if (Reloc.getSymbol() != Obj->symbol_end()) {
    Reloc.getSymbol()->getName(Name);
    Addend = Addr;
    return;
  }

  auto RE = Obj->getRelocation(Reloc.getRawDataRefImpl());
  SectionRef RelocSection = Obj->getRelocationSection(RE);

  uint64_t SectionAddr = RelocSection.getAddress();

  auto Sym = Symbols.upper_bound(Addr);
  if (Sym == Symbols.begin()) {
    // The first symbol in the object is after this reference, the best we can
    // do is section-relative notation.
    RelocSection.getName(Name);
    Addend = Addr - SectionAddr;
    return;
  }

  // Go back one so that SymbolAddress <= Addr.
  --Sym;

  section_iterator SymSection = Obj->section_end();
  Sym->second.getSection(SymSection);
  if (RelocSection == *SymSection) {
    // There's a valid symbol in the same section before this reference.
    Sym->second.getName(Name);
    Addend = Addr - Sym->first;
    return;
  }

  // There is a symbol before this reference, but it's in a different
  // section. Probably not helpful to mention it, so use the section name.
  RelocSection.getName(Name);
  Addend = Addr - SectionAddr;
}

static void printUnwindRelocDest(const MachOObjectFile *Obj,
                                 std::map<uint64_t, SymbolRef> &Symbols,
                                 const RelocationRef &Reloc, uint64_t Addr) {
  StringRef Name;
  uint64_t Addend;

  if (!Reloc.getObjectFile())
    return;

  findUnwindRelocNameAddend(Obj, Symbols, Reloc, Addr, Name, Addend);

  outs() << Name;
  if (Addend)
    outs() << " + " << format("0x%" PRIx64, Addend);
}

static void
printMachOCompactUnwindSection(const MachOObjectFile *Obj,
                               std::map<uint64_t, SymbolRef> &Symbols,
                               const SectionRef &CompactUnwind) {

  assert(Obj->isLittleEndian() &&
         "There should not be a big-endian .o with __compact_unwind");

  bool Is64 = Obj->is64Bit();
  uint32_t PointerSize = Is64 ? sizeof(uint64_t) : sizeof(uint32_t);
  uint32_t EntrySize = 3 * PointerSize + 2 * sizeof(uint32_t);

  StringRef Contents;
  CompactUnwind.getContents(Contents);

  SmallVector<CompactUnwindEntry, 4> CompactUnwinds;

  // First populate the initial raw offsets, encodings and so on from the entry.
  for (unsigned Offset = 0; Offset < Contents.size(); Offset += EntrySize) {
    CompactUnwindEntry Entry(Contents.data(), Offset, Is64);
    CompactUnwinds.push_back(Entry);
  }

  // Next we need to look at the relocations to find out what objects are
  // actually being referred to.
  for (const RelocationRef &Reloc : CompactUnwind.relocations()) {
    uint64_t RelocAddress;
    Reloc.getOffset(RelocAddress);

    uint32_t EntryIdx = RelocAddress / EntrySize;
    uint32_t OffsetInEntry = RelocAddress - EntryIdx * EntrySize;
    CompactUnwindEntry &Entry = CompactUnwinds[EntryIdx];

    if (OffsetInEntry == 0)
      Entry.FunctionReloc = Reloc;
    else if (OffsetInEntry == PointerSize + 2 * sizeof(uint32_t))
      Entry.PersonalityReloc = Reloc;
    else if (OffsetInEntry == 2 * PointerSize + 2 * sizeof(uint32_t))
      Entry.LSDAReloc = Reloc;
    else
      llvm_unreachable("Unexpected relocation in __compact_unwind section");
  }

  // Finally, we're ready to print the data we've gathered.
  outs() << "Contents of __compact_unwind section:\n";
  for (auto &Entry : CompactUnwinds) {
    outs() << "  Entry at offset "
           << format("0x%" PRIx32, Entry.OffsetInSection) << ":\n";

    // 1. Start of the region this entry applies to.
    outs() << "    start:                " << format("0x%" PRIx64,
                                                     Entry.FunctionAddr) << ' ';
    printUnwindRelocDest(Obj, Symbols, Entry.FunctionReloc, Entry.FunctionAddr);
    outs() << '\n';

    // 2. Length of the region this entry applies to.
    outs() << "    length:               " << format("0x%" PRIx32, Entry.Length)
           << '\n';
    // 3. The 32-bit compact encoding.
    outs() << "    compact encoding:     "
           << format("0x%08" PRIx32, Entry.CompactEncoding) << '\n';

    // 4. The personality function, if present.
    if (Entry.PersonalityReloc.getObjectFile()) {
      outs() << "    personality function: "
             << format("0x%" PRIx64, Entry.PersonalityAddr) << ' ';
      printUnwindRelocDest(Obj, Symbols, Entry.PersonalityReloc,
                           Entry.PersonalityAddr);
      outs() << '\n';
    }

    // 5. This entry's language-specific data area.
    if (Entry.LSDAReloc.getObjectFile()) {
      outs() << "    LSDA:                 " << format("0x%" PRIx64,
                                                       Entry.LSDAAddr) << ' ';
      printUnwindRelocDest(Obj, Symbols, Entry.LSDAReloc, Entry.LSDAAddr);
      outs() << '\n';
    }
  }
}

//===----------------------------------------------------------------------===//
// __unwind_info section dumping
//===----------------------------------------------------------------------===//

static void printRegularSecondLevelUnwindPage(const char *PageStart) {
  const char *Pos = PageStart;
  uint32_t Kind = readNext<uint32_t>(Pos);
  (void)Kind;
  assert(Kind == 2 && "kind for a regular 2nd level index should be 2");

  uint16_t EntriesStart = readNext<uint16_t>(Pos);
  uint16_t NumEntries = readNext<uint16_t>(Pos);

  Pos = PageStart + EntriesStart;
  for (unsigned i = 0; i < NumEntries; ++i) {
    uint32_t FunctionOffset = readNext<uint32_t>(Pos);
    uint32_t Encoding = readNext<uint32_t>(Pos);

    outs() << "      [" << i << "]: "
           << "function offset=" << format("0x%08" PRIx32, FunctionOffset)
           << ", "
           << "encoding=" << format("0x%08" PRIx32, Encoding) << '\n';
  }
}

static void printCompressedSecondLevelUnwindPage(
    const char *PageStart, uint32_t FunctionBase,
    const SmallVectorImpl<uint32_t> &CommonEncodings) {
  const char *Pos = PageStart;
  uint32_t Kind = readNext<uint32_t>(Pos);
  (void)Kind;
  assert(Kind == 3 && "kind for a compressed 2nd level index should be 3");

  uint16_t EntriesStart = readNext<uint16_t>(Pos);
  uint16_t NumEntries = readNext<uint16_t>(Pos);

  uint16_t EncodingsStart = readNext<uint16_t>(Pos);
  readNext<uint16_t>(Pos);
  const auto *PageEncodings = reinterpret_cast<const support::ulittle32_t *>(
      PageStart + EncodingsStart);

  Pos = PageStart + EntriesStart;
  for (unsigned i = 0; i < NumEntries; ++i) {
    uint32_t Entry = readNext<uint32_t>(Pos);
    uint32_t FunctionOffset = FunctionBase + (Entry & 0xffffff);
    uint32_t EncodingIdx = Entry >> 24;

    uint32_t Encoding;
    if (EncodingIdx < CommonEncodings.size())
      Encoding = CommonEncodings[EncodingIdx];
    else
      Encoding = PageEncodings[EncodingIdx - CommonEncodings.size()];

    outs() << "      [" << i << "]: "
           << "function offset=" << format("0x%08" PRIx32, FunctionOffset)
           << ", "
           << "encoding[" << EncodingIdx
           << "]=" << format("0x%08" PRIx32, Encoding) << '\n';
  }
}

static void printMachOUnwindInfoSection(const MachOObjectFile *Obj,
                                        std::map<uint64_t, SymbolRef> &Symbols,
                                        const SectionRef &UnwindInfo) {

  assert(Obj->isLittleEndian() &&
         "There should not be a big-endian .o with __unwind_info");

  outs() << "Contents of __unwind_info section:\n";

  StringRef Contents;
  UnwindInfo.getContents(Contents);
  const char *Pos = Contents.data();

  //===----------------------------------
  // Section header
  //===----------------------------------

  uint32_t Version = readNext<uint32_t>(Pos);
  outs() << "  Version:                                   "
         << format("0x%" PRIx32, Version) << '\n';
  assert(Version == 1 && "only understand version 1");

  uint32_t CommonEncodingsStart = readNext<uint32_t>(Pos);
  outs() << "  Common encodings array section offset:     "
         << format("0x%" PRIx32, CommonEncodingsStart) << '\n';
  uint32_t NumCommonEncodings = readNext<uint32_t>(Pos);
  outs() << "  Number of common encodings in array:       "
         << format("0x%" PRIx32, NumCommonEncodings) << '\n';

  uint32_t PersonalitiesStart = readNext<uint32_t>(Pos);
  outs() << "  Personality function array section offset: "
         << format("0x%" PRIx32, PersonalitiesStart) << '\n';
  uint32_t NumPersonalities = readNext<uint32_t>(Pos);
  outs() << "  Number of personality functions in array:  "
         << format("0x%" PRIx32, NumPersonalities) << '\n';

  uint32_t IndicesStart = readNext<uint32_t>(Pos);
  outs() << "  Index array section offset:                "
         << format("0x%" PRIx32, IndicesStart) << '\n';
  uint32_t NumIndices = readNext<uint32_t>(Pos);
  outs() << "  Number of indices in array:                "
         << format("0x%" PRIx32, NumIndices) << '\n';

  //===----------------------------------
  // A shared list of common encodings
  //===----------------------------------

  // These occupy indices in the range [0, N] whenever an encoding is referenced
  // from a compressed 2nd level index table. In practice the linker only
  // creates ~128 of these, so that indices are available to embed encodings in
  // the 2nd level index.

  SmallVector<uint32_t, 64> CommonEncodings;
  outs() << "  Common encodings: (count = " << NumCommonEncodings << ")\n";
  Pos = Contents.data() + CommonEncodingsStart;
  for (unsigned i = 0; i < NumCommonEncodings; ++i) {
    uint32_t Encoding = readNext<uint32_t>(Pos);
    CommonEncodings.push_back(Encoding);

    outs() << "    encoding[" << i << "]: " << format("0x%08" PRIx32, Encoding)
           << '\n';
  }

  //===----------------------------------
  // Personality functions used in this executable
  //===----------------------------------

  // There should be only a handful of these (one per source language,
  // roughly). Particularly since they only get 2 bits in the compact encoding.

  outs() << "  Personality functions: (count = " << NumPersonalities << ")\n";
  Pos = Contents.data() + PersonalitiesStart;
  for (unsigned i = 0; i < NumPersonalities; ++i) {
    uint32_t PersonalityFn = readNext<uint32_t>(Pos);
    outs() << "    personality[" << i + 1
           << "]: " << format("0x%08" PRIx32, PersonalityFn) << '\n';
  }

  //===----------------------------------
  // The level 1 index entries
  //===----------------------------------

  // These specify an approximate place to start searching for the more detailed
  // information, sorted by PC.

  struct IndexEntry {
    uint32_t FunctionOffset;
    uint32_t SecondLevelPageStart;
    uint32_t LSDAStart;
  };

  SmallVector<IndexEntry, 4> IndexEntries;

  outs() << "  Top level indices: (count = " << NumIndices << ")\n";
  Pos = Contents.data() + IndicesStart;
  for (unsigned i = 0; i < NumIndices; ++i) {
    IndexEntry Entry;

    Entry.FunctionOffset = readNext<uint32_t>(Pos);
    Entry.SecondLevelPageStart = readNext<uint32_t>(Pos);
    Entry.LSDAStart = readNext<uint32_t>(Pos);
    IndexEntries.push_back(Entry);

    outs() << "    [" << i << "]: "
           << "function offset=" << format("0x%08" PRIx32, Entry.FunctionOffset)
           << ", "
           << "2nd level page offset="
           << format("0x%08" PRIx32, Entry.SecondLevelPageStart) << ", "
           << "LSDA offset=" << format("0x%08" PRIx32, Entry.LSDAStart) << '\n';
  }

  //===----------------------------------
  // Next come the LSDA tables
  //===----------------------------------

  // The LSDA layout is rather implicit: it's a contiguous array of entries from
  // the first top-level index's LSDAOffset to the last (sentinel).

  outs() << "  LSDA descriptors:\n";
  Pos = Contents.data() + IndexEntries[0].LSDAStart;
  int NumLSDAs = (IndexEntries.back().LSDAStart - IndexEntries[0].LSDAStart) /
                 (2 * sizeof(uint32_t));
  for (int i = 0; i < NumLSDAs; ++i) {
    uint32_t FunctionOffset = readNext<uint32_t>(Pos);
    uint32_t LSDAOffset = readNext<uint32_t>(Pos);
    outs() << "    [" << i << "]: "
           << "function offset=" << format("0x%08" PRIx32, FunctionOffset)
           << ", "
           << "LSDA offset=" << format("0x%08" PRIx32, LSDAOffset) << '\n';
  }

  //===----------------------------------
  // Finally, the 2nd level indices
  //===----------------------------------

  // Generally these are 4K in size, and have 2 possible forms:
  //   + Regular stores up to 511 entries with disparate encodings
  //   + Compressed stores up to 1021 entries if few enough compact encoding
  //     values are used.
  outs() << "  Second level indices:\n";
  for (unsigned i = 0; i < IndexEntries.size() - 1; ++i) {
    // The final sentinel top-level index has no associated 2nd level page
    if (IndexEntries[i].SecondLevelPageStart == 0)
      break;

    outs() << "    Second level index[" << i << "]: "
           << "offset in section="
           << format("0x%08" PRIx32, IndexEntries[i].SecondLevelPageStart)
           << ", "
           << "base function offset="
           << format("0x%08" PRIx32, IndexEntries[i].FunctionOffset) << '\n';

    Pos = Contents.data() + IndexEntries[i].SecondLevelPageStart;
    uint32_t Kind = *reinterpret_cast<const support::ulittle32_t *>(Pos);
    if (Kind == 2)
      printRegularSecondLevelUnwindPage(Pos);
    else if (Kind == 3)
      printCompressedSecondLevelUnwindPage(Pos, IndexEntries[i].FunctionOffset,
                                           CommonEncodings);
    else
      llvm_unreachable("Do not know how to print this kind of 2nd level page");
  }
}

void llvm::printMachOUnwindInfo(const MachOObjectFile *Obj) {
  std::map<uint64_t, SymbolRef> Symbols;
  for (const SymbolRef &SymRef : Obj->symbols()) {
    // Discard any undefined or absolute symbols. They're not going to take part
    // in the convenience lookup for unwind info and just take up resources.
    section_iterator Section = Obj->section_end();
    SymRef.getSection(Section);
    if (Section == Obj->section_end())
      continue;

    uint64_t Addr;
    SymRef.getAddress(Addr);
    Symbols.insert(std::make_pair(Addr, SymRef));
  }

  for (const SectionRef &Section : Obj->sections()) {
    StringRef SectName;
    Section.getName(SectName);
    if (SectName == "__compact_unwind")
      printMachOCompactUnwindSection(Obj, Symbols, Section);
    else if (SectName == "__unwind_info")
      printMachOUnwindInfoSection(Obj, Symbols, Section);
    else if (SectName == "__eh_frame")
      outs() << "llvm-objdump: warning: unhandled __eh_frame section\n";
  }
}

static void PrintMachHeader(uint32_t magic, uint32_t cputype,
                            uint32_t cpusubtype, uint32_t filetype,
                            uint32_t ncmds, uint32_t sizeofcmds, uint32_t flags,
                            bool verbose) {
  outs() << "Mach header\n";
  outs() << "      magic cputype cpusubtype  caps    filetype ncmds "
            "sizeofcmds      flags\n";
  if (verbose) {
    if (magic == MachO::MH_MAGIC)
      outs() << "   MH_MAGIC";
    else if (magic == MachO::MH_MAGIC_64)
      outs() << "MH_MAGIC_64";
    else
      outs() << format(" 0x%08" PRIx32, magic);
    switch (cputype) {
    case MachO::CPU_TYPE_I386:
      outs() << "    I386";
      switch (cpusubtype & ~MachO::CPU_SUBTYPE_MASK) {
      case MachO::CPU_SUBTYPE_I386_ALL:
        outs() << "        ALL";
        break;
      default:
        outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
        break;
      }
      break;
    case MachO::CPU_TYPE_X86_64:
      outs() << "  X86_64";
    case MachO::CPU_SUBTYPE_X86_64_ALL:
      outs() << "        ALL";
      break;
    case MachO::CPU_SUBTYPE_X86_64_H:
      outs() << "    Haswell";
      outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
      break;
    case MachO::CPU_TYPE_ARM:
      outs() << "     ARM";
      switch (cpusubtype & ~MachO::CPU_SUBTYPE_MASK) {
      case MachO::CPU_SUBTYPE_ARM_ALL:
        outs() << "        ALL";
        break;
      case MachO::CPU_SUBTYPE_ARM_V4T:
        outs() << "        V4T";
        break;
      case MachO::CPU_SUBTYPE_ARM_V5TEJ:
        outs() << "      V5TEJ";
        break;
      case MachO::CPU_SUBTYPE_ARM_XSCALE:
        outs() << "     XSCALE";
        break;
      case MachO::CPU_SUBTYPE_ARM_V6:
        outs() << "         V6";
        break;
      case MachO::CPU_SUBTYPE_ARM_V6M:
        outs() << "        V6M";
        break;
      case MachO::CPU_SUBTYPE_ARM_V7:
        outs() << "         V7";
        break;
      case MachO::CPU_SUBTYPE_ARM_V7EM:
        outs() << "       V7EM";
        break;
      case MachO::CPU_SUBTYPE_ARM_V7K:
        outs() << "        V7K";
        break;
      case MachO::CPU_SUBTYPE_ARM_V7M:
        outs() << "        V7M";
        break;
      case MachO::CPU_SUBTYPE_ARM_V7S:
        outs() << "        V7S";
        break;
      default:
        outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
        break;
      }
      break;
    case MachO::CPU_TYPE_ARM64:
      outs() << "   ARM64";
      switch (cpusubtype & ~MachO::CPU_SUBTYPE_MASK) {
      case MachO::CPU_SUBTYPE_ARM64_ALL:
        outs() << "        ALL";
        break;
      default:
        outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
        break;
      }
      break;
    case MachO::CPU_TYPE_POWERPC:
      outs() << "     PPC";
      switch (cpusubtype & ~MachO::CPU_SUBTYPE_MASK) {
      case MachO::CPU_SUBTYPE_POWERPC_ALL:
        outs() << "        ALL";
        break;
      default:
        outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
        break;
      }
      break;
    case MachO::CPU_TYPE_POWERPC64:
      outs() << "   PPC64";
      switch (cpusubtype & ~MachO::CPU_SUBTYPE_MASK) {
      case MachO::CPU_SUBTYPE_POWERPC_ALL:
        outs() << "        ALL";
        break;
      default:
        outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
        break;
      }
      break;
    }
    if ((cpusubtype & MachO::CPU_SUBTYPE_MASK) == MachO::CPU_SUBTYPE_LIB64) {
      outs() << " LIB64";
    } else {
      outs() << format("  0x%02" PRIx32,
                       (cpusubtype & MachO::CPU_SUBTYPE_MASK) >> 24);
    }
    switch (filetype) {
    case MachO::MH_OBJECT:
      outs() << "      OBJECT";
      break;
    case MachO::MH_EXECUTE:
      outs() << "     EXECUTE";
      break;
    case MachO::MH_FVMLIB:
      outs() << "      FVMLIB";
      break;
    case MachO::MH_CORE:
      outs() << "        CORE";
      break;
    case MachO::MH_PRELOAD:
      outs() << "     PRELOAD";
      break;
    case MachO::MH_DYLIB:
      outs() << "       DYLIB";
      break;
    case MachO::MH_DYLIB_STUB:
      outs() << "  DYLIB_STUB";
      break;
    case MachO::MH_DYLINKER:
      outs() << "    DYLINKER";
      break;
    case MachO::MH_BUNDLE:
      outs() << "      BUNDLE";
      break;
    case MachO::MH_DSYM:
      outs() << "        DSYM";
      break;
    case MachO::MH_KEXT_BUNDLE:
      outs() << "  KEXTBUNDLE";
      break;
    default:
      outs() << format("  %10u", filetype);
      break;
    }
    outs() << format(" %5u", ncmds);
    outs() << format(" %10u", sizeofcmds);
    uint32_t f = flags;
    if (f & MachO::MH_NOUNDEFS) {
      outs() << "   NOUNDEFS";
      f &= ~MachO::MH_NOUNDEFS;
    }
    if (f & MachO::MH_INCRLINK) {
      outs() << " INCRLINK";
      f &= ~MachO::MH_INCRLINK;
    }
    if (f & MachO::MH_DYLDLINK) {
      outs() << " DYLDLINK";
      f &= ~MachO::MH_DYLDLINK;
    }
    if (f & MachO::MH_BINDATLOAD) {
      outs() << " BINDATLOAD";
      f &= ~MachO::MH_BINDATLOAD;
    }
    if (f & MachO::MH_PREBOUND) {
      outs() << " PREBOUND";
      f &= ~MachO::MH_PREBOUND;
    }
    if (f & MachO::MH_SPLIT_SEGS) {
      outs() << " SPLIT_SEGS";
      f &= ~MachO::MH_SPLIT_SEGS;
    }
    if (f & MachO::MH_LAZY_INIT) {
      outs() << " LAZY_INIT";
      f &= ~MachO::MH_LAZY_INIT;
    }
    if (f & MachO::MH_TWOLEVEL) {
      outs() << " TWOLEVEL";
      f &= ~MachO::MH_TWOLEVEL;
    }
    if (f & MachO::MH_FORCE_FLAT) {
      outs() << " FORCE_FLAT";
      f &= ~MachO::MH_FORCE_FLAT;
    }
    if (f & MachO::MH_NOMULTIDEFS) {
      outs() << " NOMULTIDEFS";
      f &= ~MachO::MH_NOMULTIDEFS;
    }
    if (f & MachO::MH_NOFIXPREBINDING) {
      outs() << " NOFIXPREBINDING";
      f &= ~MachO::MH_NOFIXPREBINDING;
    }
    if (f & MachO::MH_PREBINDABLE) {
      outs() << " PREBINDABLE";
      f &= ~MachO::MH_PREBINDABLE;
    }
    if (f & MachO::MH_ALLMODSBOUND) {
      outs() << " ALLMODSBOUND";
      f &= ~MachO::MH_ALLMODSBOUND;
    }
    if (f & MachO::MH_SUBSECTIONS_VIA_SYMBOLS) {
      outs() << " SUBSECTIONS_VIA_SYMBOLS";
      f &= ~MachO::MH_SUBSECTIONS_VIA_SYMBOLS;
    }
    if (f & MachO::MH_CANONICAL) {
      outs() << " CANONICAL";
      f &= ~MachO::MH_CANONICAL;
    }
    if (f & MachO::MH_WEAK_DEFINES) {
      outs() << " WEAK_DEFINES";
      f &= ~MachO::MH_WEAK_DEFINES;
    }
    if (f & MachO::MH_BINDS_TO_WEAK) {
      outs() << " BINDS_TO_WEAK";
      f &= ~MachO::MH_BINDS_TO_WEAK;
    }
    if (f & MachO::MH_ALLOW_STACK_EXECUTION) {
      outs() << " ALLOW_STACK_EXECUTION";
      f &= ~MachO::MH_ALLOW_STACK_EXECUTION;
    }
    if (f & MachO::MH_DEAD_STRIPPABLE_DYLIB) {
      outs() << " DEAD_STRIPPABLE_DYLIB";
      f &= ~MachO::MH_DEAD_STRIPPABLE_DYLIB;
    }
    if (f & MachO::MH_PIE) {
      outs() << " PIE";
      f &= ~MachO::MH_PIE;
    }
    if (f & MachO::MH_NO_REEXPORTED_DYLIBS) {
      outs() << " NO_REEXPORTED_DYLIBS";
      f &= ~MachO::MH_NO_REEXPORTED_DYLIBS;
    }
    if (f & MachO::MH_HAS_TLV_DESCRIPTORS) {
      outs() << " MH_HAS_TLV_DESCRIPTORS";
      f &= ~MachO::MH_HAS_TLV_DESCRIPTORS;
    }
    if (f & MachO::MH_NO_HEAP_EXECUTION) {
      outs() << " MH_NO_HEAP_EXECUTION";
      f &= ~MachO::MH_NO_HEAP_EXECUTION;
    }
    if (f & MachO::MH_APP_EXTENSION_SAFE) {
      outs() << " APP_EXTENSION_SAFE";
      f &= ~MachO::MH_APP_EXTENSION_SAFE;
    }
    if (f != 0 || flags == 0)
      outs() << format(" 0x%08" PRIx32, f);
  } else {
    outs() << format(" 0x%08" PRIx32, magic);
    outs() << format(" %7d", cputype);
    outs() << format(" %10d", cpusubtype & ~MachO::CPU_SUBTYPE_MASK);
    outs() << format("  0x%02" PRIx32,
                     (cpusubtype & MachO::CPU_SUBTYPE_MASK) >> 24);
    outs() << format("  %10u", filetype);
    outs() << format(" %5u", ncmds);
    outs() << format(" %10u", sizeofcmds);
    outs() << format(" 0x%08" PRIx32, flags);
  }
  outs() << "\n";
}

static void PrintSegmentCommand(uint32_t cmd, uint32_t cmdsize,
                                StringRef SegName, uint64_t vmaddr,
                                uint64_t vmsize, uint64_t fileoff,
                                uint64_t filesize, uint32_t maxprot,
                                uint32_t initprot, uint32_t nsects,
                                uint32_t flags, uint32_t object_size,
                                bool verbose) {
  uint64_t expected_cmdsize;
  if (cmd == MachO::LC_SEGMENT) {
    outs() << "      cmd LC_SEGMENT\n";
    expected_cmdsize = nsects;
    expected_cmdsize *= sizeof(struct MachO::section);
    expected_cmdsize += sizeof(struct MachO::segment_command);
  } else {
    outs() << "      cmd LC_SEGMENT_64\n";
    expected_cmdsize = nsects;
    expected_cmdsize *= sizeof(struct MachO::section_64);
    expected_cmdsize += sizeof(struct MachO::segment_command_64);
  }
  outs() << "  cmdsize " << cmdsize;
  if (cmdsize != expected_cmdsize)
    outs() << " Inconsistent size\n";
  else
    outs() << "\n";
  outs() << "  segname " << SegName << "\n";
  if (cmd == MachO::LC_SEGMENT_64) {
    outs() << "   vmaddr " << format("0x%016" PRIx64, vmaddr) << "\n";
    outs() << "   vmsize " << format("0x%016" PRIx64, vmsize) << "\n";
  } else {
    outs() << "   vmaddr " << format("0x%08" PRIx32, vmaddr) << "\n";
    outs() << "   vmsize " << format("0x%08" PRIx32, vmsize) << "\n";
  }
  outs() << "  fileoff " << fileoff;
  if (fileoff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << " filesize " << filesize;
  if (fileoff + filesize > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  if (verbose) {
    if ((maxprot &
         ~(MachO::VM_PROT_READ | MachO::VM_PROT_WRITE |
           MachO::VM_PROT_EXECUTE)) != 0)
      outs() << "  maxprot ?" << format("0x%08" PRIx32, maxprot) << "\n";
    else {
      if (maxprot & MachO::VM_PROT_READ)
        outs() << "  maxprot r";
      else
        outs() << "  maxprot -";
      if (maxprot & MachO::VM_PROT_WRITE)
        outs() << "w";
      else
        outs() << "-";
      if (maxprot & MachO::VM_PROT_EXECUTE)
        outs() << "x\n";
      else
        outs() << "-\n";
    }
    if ((initprot &
         ~(MachO::VM_PROT_READ | MachO::VM_PROT_WRITE |
           MachO::VM_PROT_EXECUTE)) != 0)
      outs() << "  initprot ?" << format("0x%08" PRIx32, initprot) << "\n";
    else {
      if (initprot & MachO::VM_PROT_READ)
        outs() << " initprot r";
      else
        outs() << " initprot -";
      if (initprot & MachO::VM_PROT_WRITE)
        outs() << "w";
      else
        outs() << "-";
      if (initprot & MachO::VM_PROT_EXECUTE)
        outs() << "x\n";
      else
        outs() << "-\n";
    }
  } else {
    outs() << "  maxprot " << format("0x%08" PRIx32, maxprot) << "\n";
    outs() << " initprot " << format("0x%08" PRIx32, initprot) << "\n";
  }
  outs() << "   nsects " << nsects << "\n";
  if (verbose) {
    outs() << "    flags";
    if (flags == 0)
      outs() << " (none)\n";
    else {
      if (flags & MachO::SG_HIGHVM) {
        outs() << " HIGHVM";
        flags &= ~MachO::SG_HIGHVM;
      }
      if (flags & MachO::SG_FVMLIB) {
        outs() << " FVMLIB";
        flags &= ~MachO::SG_FVMLIB;
      }
      if (flags & MachO::SG_NORELOC) {
        outs() << " NORELOC";
        flags &= ~MachO::SG_NORELOC;
      }
      if (flags & MachO::SG_PROTECTED_VERSION_1) {
        outs() << " PROTECTED_VERSION_1";
        flags &= ~MachO::SG_PROTECTED_VERSION_1;
      }
      if (flags)
        outs() << format(" 0x%08" PRIx32, flags) << " (unknown flags)\n";
      else
        outs() << "\n";
    }
  } else {
    outs() << "    flags " << format("0x%" PRIx32, flags) << "\n";
  }
}

static void PrintSection(const char *sectname, const char *segname,
                         uint64_t addr, uint64_t size, uint32_t offset,
                         uint32_t align, uint32_t reloff, uint32_t nreloc,
                         uint32_t flags, uint32_t reserved1, uint32_t reserved2,
                         uint32_t cmd, const char *sg_segname,
                         uint32_t filetype, uint32_t object_size,
                         bool verbose) {
  outs() << "Section\n";
  outs() << "  sectname " << format("%.16s\n", sectname);
  outs() << "   segname " << format("%.16s", segname);
  if (filetype != MachO::MH_OBJECT && strncmp(sg_segname, segname, 16) != 0)
    outs() << " (does not match segment)\n";
  else
    outs() << "\n";
  if (cmd == MachO::LC_SEGMENT_64) {
    outs() << "      addr " << format("0x%016" PRIx64, addr) << "\n";
    outs() << "      size " << format("0x%016" PRIx64, size);
  } else {
    outs() << "      addr " << format("0x%08" PRIx32, addr) << "\n";
    outs() << "      size " << format("0x%08" PRIx32, size);
  }
  if ((flags & MachO::S_ZEROFILL) != 0 && offset + size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "    offset " << offset;
  if (offset > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  uint32_t align_shifted = 1 << align;
  outs() << "     align 2^" << align << " (" << align_shifted << ")\n";
  outs() << "    reloff " << reloff;
  if (reloff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "    nreloc " << nreloc;
  if (reloff + nreloc * sizeof(struct MachO::relocation_info) > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  uint32_t section_type = flags & MachO::SECTION_TYPE;
  if (verbose) {
    outs() << "      type";
    if (section_type == MachO::S_REGULAR)
      outs() << " S_REGULAR\n";
    else if (section_type == MachO::S_ZEROFILL)
      outs() << " S_ZEROFILL\n";
    else if (section_type == MachO::S_CSTRING_LITERALS)
      outs() << " S_CSTRING_LITERALS\n";
    else if (section_type == MachO::S_4BYTE_LITERALS)
      outs() << " S_4BYTE_LITERALS\n";
    else if (section_type == MachO::S_8BYTE_LITERALS)
      outs() << " S_8BYTE_LITERALS\n";
    else if (section_type == MachO::S_16BYTE_LITERALS)
      outs() << " S_16BYTE_LITERALS\n";
    else if (section_type == MachO::S_LITERAL_POINTERS)
      outs() << " S_LITERAL_POINTERS\n";
    else if (section_type == MachO::S_NON_LAZY_SYMBOL_POINTERS)
      outs() << " S_NON_LAZY_SYMBOL_POINTERS\n";
    else if (section_type == MachO::S_LAZY_SYMBOL_POINTERS)
      outs() << " S_LAZY_SYMBOL_POINTERS\n";
    else if (section_type == MachO::S_SYMBOL_STUBS)
      outs() << " S_SYMBOL_STUBS\n";
    else if (section_type == MachO::S_MOD_INIT_FUNC_POINTERS)
      outs() << " S_MOD_INIT_FUNC_POINTERS\n";
    else if (section_type == MachO::S_MOD_TERM_FUNC_POINTERS)
      outs() << " S_MOD_TERM_FUNC_POINTERS\n";
    else if (section_type == MachO::S_COALESCED)
      outs() << " S_COALESCED\n";
    else if (section_type == MachO::S_INTERPOSING)
      outs() << " S_INTERPOSING\n";
    else if (section_type == MachO::S_DTRACE_DOF)
      outs() << " S_DTRACE_DOF\n";
    else if (section_type == MachO::S_LAZY_DYLIB_SYMBOL_POINTERS)
      outs() << " S_LAZY_DYLIB_SYMBOL_POINTERS\n";
    else if (section_type == MachO::S_THREAD_LOCAL_REGULAR)
      outs() << " S_THREAD_LOCAL_REGULAR\n";
    else if (section_type == MachO::S_THREAD_LOCAL_ZEROFILL)
      outs() << " S_THREAD_LOCAL_ZEROFILL\n";
    else if (section_type == MachO::S_THREAD_LOCAL_VARIABLES)
      outs() << " S_THREAD_LOCAL_VARIABLES\n";
    else if (section_type == MachO::S_THREAD_LOCAL_VARIABLE_POINTERS)
      outs() << " S_THREAD_LOCAL_VARIABLE_POINTERS\n";
    else if (section_type == MachO::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS)
      outs() << " S_THREAD_LOCAL_INIT_FUNCTION_POINTERS\n";
    else
      outs() << format("0x%08" PRIx32, section_type) << "\n";
    outs() << "attributes";
    uint32_t section_attributes = flags & MachO::SECTION_ATTRIBUTES;
    if (section_attributes & MachO::S_ATTR_PURE_INSTRUCTIONS)
      outs() << " PURE_INSTRUCTIONS";
    if (section_attributes & MachO::S_ATTR_NO_TOC)
      outs() << " NO_TOC";
    if (section_attributes & MachO::S_ATTR_STRIP_STATIC_SYMS)
      outs() << " STRIP_STATIC_SYMS";
    if (section_attributes & MachO::S_ATTR_NO_DEAD_STRIP)
      outs() << " NO_DEAD_STRIP";
    if (section_attributes & MachO::S_ATTR_LIVE_SUPPORT)
      outs() << " LIVE_SUPPORT";
    if (section_attributes & MachO::S_ATTR_SELF_MODIFYING_CODE)
      outs() << " SELF_MODIFYING_CODE";
    if (section_attributes & MachO::S_ATTR_DEBUG)
      outs() << " DEBUG";
    if (section_attributes & MachO::S_ATTR_SOME_INSTRUCTIONS)
      outs() << " SOME_INSTRUCTIONS";
    if (section_attributes & MachO::S_ATTR_EXT_RELOC)
      outs() << " EXT_RELOC";
    if (section_attributes & MachO::S_ATTR_LOC_RELOC)
      outs() << " LOC_RELOC";
    if (section_attributes == 0)
      outs() << " (none)";
    outs() << "\n";
  } else
    outs() << "     flags " << format("0x%08" PRIx32, flags) << "\n";
  outs() << " reserved1 " << reserved1;
  if (section_type == MachO::S_SYMBOL_STUBS ||
      section_type == MachO::S_LAZY_SYMBOL_POINTERS ||
      section_type == MachO::S_LAZY_DYLIB_SYMBOL_POINTERS ||
      section_type == MachO::S_NON_LAZY_SYMBOL_POINTERS ||
      section_type == MachO::S_THREAD_LOCAL_VARIABLE_POINTERS)
    outs() << " (index into indirect symbol table)\n";
  else
    outs() << "\n";
  outs() << " reserved2 " << reserved2;
  if (section_type == MachO::S_SYMBOL_STUBS)
    outs() << " (size of stubs)\n";
  else
    outs() << "\n";
}

static void PrintSymtabLoadCommand(MachO::symtab_command st, uint32_t cputype,
                                   uint32_t object_size) {
  outs() << "     cmd LC_SYMTAB\n";
  outs() << " cmdsize " << st.cmdsize;
  if (st.cmdsize != sizeof(struct MachO::symtab_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "  symoff " << st.symoff;
  if (st.symoff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "   nsyms " << st.nsyms;
  uint64_t big_size;
  if (cputype & MachO::CPU_ARCH_ABI64) {
    big_size = st.nsyms;
    big_size *= sizeof(struct MachO::nlist_64);
    big_size += st.symoff;
    if (big_size > object_size)
      outs() << " (past end of file)\n";
    else
      outs() << "\n";
  } else {
    big_size = st.nsyms;
    big_size *= sizeof(struct MachO::nlist);
    big_size += st.symoff;
    if (big_size > object_size)
      outs() << " (past end of file)\n";
    else
      outs() << "\n";
  }
  outs() << "  stroff " << st.stroff;
  if (st.stroff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << " strsize " << st.strsize;
  big_size = st.stroff;
  big_size += st.strsize;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
}

static void PrintDysymtabLoadCommand(MachO::dysymtab_command dyst,
                                     uint32_t nsyms, uint32_t object_size,
                                     uint32_t cputype) {
  outs() << "            cmd LC_DYSYMTAB\n";
  outs() << "        cmdsize " << dyst.cmdsize;
  if (dyst.cmdsize != sizeof(struct MachO::dysymtab_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "      ilocalsym " << dyst.ilocalsym;
  if (dyst.ilocalsym > nsyms)
    outs() << " (greater than the number of symbols)\n";
  else
    outs() << "\n";
  outs() << "      nlocalsym " << dyst.nlocalsym;
  uint64_t big_size;
  big_size = dyst.ilocalsym;
  big_size += dyst.nlocalsym;
  if (big_size > nsyms)
    outs() << " (past the end of the symbol table)\n";
  else
    outs() << "\n";
  outs() << "     iextdefsym " << dyst.iextdefsym;
  if (dyst.iextdefsym > nsyms)
    outs() << " (greater than the number of symbols)\n";
  else
    outs() << "\n";
  outs() << "     nextdefsym " << dyst.nextdefsym;
  big_size = dyst.iextdefsym;
  big_size += dyst.nextdefsym;
  if (big_size > nsyms)
    outs() << " (past the end of the symbol table)\n";
  else
    outs() << "\n";
  outs() << "      iundefsym " << dyst.iundefsym;
  if (dyst.iundefsym > nsyms)
    outs() << " (greater than the number of symbols)\n";
  else
    outs() << "\n";
  outs() << "      nundefsym " << dyst.nundefsym;
  big_size = dyst.iundefsym;
  big_size += dyst.nundefsym;
  if (big_size > nsyms)
    outs() << " (past the end of the symbol table)\n";
  else
    outs() << "\n";
  outs() << "         tocoff " << dyst.tocoff;
  if (dyst.tocoff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "           ntoc " << dyst.ntoc;
  big_size = dyst.ntoc;
  big_size *= sizeof(struct MachO::dylib_table_of_contents);
  big_size += dyst.tocoff;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "      modtaboff " << dyst.modtaboff;
  if (dyst.modtaboff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "        nmodtab " << dyst.nmodtab;
  uint64_t modtabend;
  if (cputype & MachO::CPU_ARCH_ABI64) {
    modtabend = dyst.nmodtab;
    modtabend *= sizeof(struct MachO::dylib_module_64);
    modtabend += dyst.modtaboff;
  } else {
    modtabend = dyst.nmodtab;
    modtabend *= sizeof(struct MachO::dylib_module);
    modtabend += dyst.modtaboff;
  }
  if (modtabend > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "   extrefsymoff " << dyst.extrefsymoff;
  if (dyst.extrefsymoff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "    nextrefsyms " << dyst.nextrefsyms;
  big_size = dyst.nextrefsyms;
  big_size *= sizeof(struct MachO::dylib_reference);
  big_size += dyst.extrefsymoff;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << " indirectsymoff " << dyst.indirectsymoff;
  if (dyst.indirectsymoff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "  nindirectsyms " << dyst.nindirectsyms;
  big_size = dyst.nindirectsyms;
  big_size *= sizeof(uint32_t);
  big_size += dyst.indirectsymoff;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "      extreloff " << dyst.extreloff;
  if (dyst.extreloff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "        nextrel " << dyst.nextrel;
  big_size = dyst.nextrel;
  big_size *= sizeof(struct MachO::relocation_info);
  big_size += dyst.extreloff;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "      locreloff " << dyst.locreloff;
  if (dyst.locreloff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "        nlocrel " << dyst.nlocrel;
  big_size = dyst.nlocrel;
  big_size *= sizeof(struct MachO::relocation_info);
  big_size += dyst.locreloff;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
}

static void PrintDyldInfoLoadCommand(MachO::dyld_info_command dc,
                                     uint32_t object_size) {
  if (dc.cmd == MachO::LC_DYLD_INFO)
    outs() << "            cmd LC_DYLD_INFO\n";
  else
    outs() << "            cmd LC_DYLD_INFO_ONLY\n";
  outs() << "        cmdsize " << dc.cmdsize;
  if (dc.cmdsize != sizeof(struct MachO::dyld_info_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "     rebase_off " << dc.rebase_off;
  if (dc.rebase_off > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "    rebase_size " << dc.rebase_size;
  uint64_t big_size;
  big_size = dc.rebase_off;
  big_size += dc.rebase_size;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "       bind_off " << dc.bind_off;
  if (dc.bind_off > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "      bind_size " << dc.bind_size;
  big_size = dc.bind_off;
  big_size += dc.bind_size;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "  weak_bind_off " << dc.weak_bind_off;
  if (dc.weak_bind_off > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << " weak_bind_size " << dc.weak_bind_size;
  big_size = dc.weak_bind_off;
  big_size += dc.weak_bind_size;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "  lazy_bind_off " << dc.lazy_bind_off;
  if (dc.lazy_bind_off > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << " lazy_bind_size " << dc.lazy_bind_size;
  big_size = dc.lazy_bind_off;
  big_size += dc.lazy_bind_size;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "     export_off " << dc.export_off;
  if (dc.export_off > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << "    export_size " << dc.export_size;
  big_size = dc.export_off;
  big_size += dc.export_size;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
}

static void PrintDyldLoadCommand(MachO::dylinker_command dyld,
                                 const char *Ptr) {
  if (dyld.cmd == MachO::LC_ID_DYLINKER)
    outs() << "          cmd LC_ID_DYLINKER\n";
  else if (dyld.cmd == MachO::LC_LOAD_DYLINKER)
    outs() << "          cmd LC_LOAD_DYLINKER\n";
  else if (dyld.cmd == MachO::LC_DYLD_ENVIRONMENT)
    outs() << "          cmd LC_DYLD_ENVIRONMENT\n";
  else
    outs() << "          cmd ?(" << dyld.cmd << ")\n";
  outs() << "      cmdsize " << dyld.cmdsize;
  if (dyld.cmdsize < sizeof(struct MachO::dylinker_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  if (dyld.name >= dyld.cmdsize)
    outs() << "         name ?(bad offset " << dyld.name << ")\n";
  else {
    const char *P = (const char *)(Ptr) + dyld.name;
    outs() << "         name " << P << " (offset " << dyld.name << ")\n";
  }
}

static void PrintUuidLoadCommand(MachO::uuid_command uuid) {
  outs() << "     cmd LC_UUID\n";
  outs() << " cmdsize " << uuid.cmdsize;
  if (uuid.cmdsize != sizeof(struct MachO::uuid_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "    uuid ";
  outs() << format("%02" PRIX32, uuid.uuid[0]);
  outs() << format("%02" PRIX32, uuid.uuid[1]);
  outs() << format("%02" PRIX32, uuid.uuid[2]);
  outs() << format("%02" PRIX32, uuid.uuid[3]);
  outs() << "-";
  outs() << format("%02" PRIX32, uuid.uuid[4]);
  outs() << format("%02" PRIX32, uuid.uuid[5]);
  outs() << "-";
  outs() << format("%02" PRIX32, uuid.uuid[6]);
  outs() << format("%02" PRIX32, uuid.uuid[7]);
  outs() << "-";
  outs() << format("%02" PRIX32, uuid.uuid[8]);
  outs() << format("%02" PRIX32, uuid.uuid[9]);
  outs() << "-";
  outs() << format("%02" PRIX32, uuid.uuid[10]);
  outs() << format("%02" PRIX32, uuid.uuid[11]);
  outs() << format("%02" PRIX32, uuid.uuid[12]);
  outs() << format("%02" PRIX32, uuid.uuid[13]);
  outs() << format("%02" PRIX32, uuid.uuid[14]);
  outs() << format("%02" PRIX32, uuid.uuid[15]);
  outs() << "\n";
}

static void PrintVersionMinLoadCommand(MachO::version_min_command vd) {
  if (vd.cmd == MachO::LC_VERSION_MIN_MACOSX)
    outs() << "      cmd LC_VERSION_MIN_MACOSX\n";
  else if (vd.cmd == MachO::LC_VERSION_MIN_IPHONEOS)
    outs() << "      cmd LC_VERSION_MIN_IPHONEOS\n";
  else
    outs() << "      cmd " << vd.cmd << " (?)\n";
  outs() << "  cmdsize " << vd.cmdsize;
  if (vd.cmdsize != sizeof(struct MachO::version_min_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "  version " << ((vd.version >> 16) & 0xffff) << "."
         << ((vd.version >> 8) & 0xff);
  if ((vd.version & 0xff) != 0)
    outs() << "." << (vd.version & 0xff);
  outs() << "\n";
  if (vd.sdk == 0)
    outs() << "      sdk n/a\n";
  else {
    outs() << "      sdk " << ((vd.sdk >> 16) & 0xffff) << "."
           << ((vd.sdk >> 8) & 0xff);
  }
  if ((vd.sdk & 0xff) != 0)
    outs() << "." << (vd.sdk & 0xff);
  outs() << "\n";
}

static void PrintSourceVersionCommand(MachO::source_version_command sd) {
  outs() << "      cmd LC_SOURCE_VERSION\n";
  outs() << "  cmdsize " << sd.cmdsize;
  if (sd.cmdsize != sizeof(struct MachO::source_version_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  uint64_t a = (sd.version >> 40) & 0xffffff;
  uint64_t b = (sd.version >> 30) & 0x3ff;
  uint64_t c = (sd.version >> 20) & 0x3ff;
  uint64_t d = (sd.version >> 10) & 0x3ff;
  uint64_t e = sd.version & 0x3ff;
  outs() << "  version " << a << "." << b;
  if (e != 0)
    outs() << "." << c << "." << d << "." << e;
  else if (d != 0)
    outs() << "." << c << "." << d;
  else if (c != 0)
    outs() << "." << c;
  outs() << "\n";
}

static void PrintEntryPointCommand(MachO::entry_point_command ep) {
  outs() << "       cmd LC_MAIN\n";
  outs() << "   cmdsize " << ep.cmdsize;
  if (ep.cmdsize != sizeof(struct MachO::entry_point_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "  entryoff " << ep.entryoff << "\n";
  outs() << " stacksize " << ep.stacksize << "\n";
}

static void PrintDylibCommand(MachO::dylib_command dl, const char *Ptr) {
  if (dl.cmd == MachO::LC_ID_DYLIB)
    outs() << "          cmd LC_ID_DYLIB\n";
  else if (dl.cmd == MachO::LC_LOAD_DYLIB)
    outs() << "          cmd LC_LOAD_DYLIB\n";
  else if (dl.cmd == MachO::LC_LOAD_WEAK_DYLIB)
    outs() << "          cmd LC_LOAD_WEAK_DYLIB\n";
  else if (dl.cmd == MachO::LC_REEXPORT_DYLIB)
    outs() << "          cmd LC_REEXPORT_DYLIB\n";
  else if (dl.cmd == MachO::LC_LAZY_LOAD_DYLIB)
    outs() << "          cmd LC_LAZY_LOAD_DYLIB\n";
  else if (dl.cmd == MachO::LC_LOAD_UPWARD_DYLIB)
    outs() << "          cmd LC_LOAD_UPWARD_DYLIB\n";
  else
    outs() << "          cmd " << dl.cmd << " (unknown)\n";
  outs() << "      cmdsize " << dl.cmdsize;
  if (dl.cmdsize < sizeof(struct MachO::dylib_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  if (dl.dylib.name < dl.cmdsize) {
    const char *P = (const char *)(Ptr) + dl.dylib.name;
    outs() << "         name " << P << " (offset " << dl.dylib.name << ")\n";
  } else {
    outs() << "         name ?(bad offset " << dl.dylib.name << ")\n";
  }
  outs() << "   time stamp " << dl.dylib.timestamp << " ";
  time_t t = dl.dylib.timestamp;
  outs() << ctime(&t);
  outs() << "      current version ";
  if (dl.dylib.current_version == 0xffffffff)
    outs() << "n/a\n";
  else
    outs() << ((dl.dylib.current_version >> 16) & 0xffff) << "."
           << ((dl.dylib.current_version >> 8) & 0xff) << "."
           << (dl.dylib.current_version & 0xff) << "\n";
  outs() << "compatibility version ";
  if (dl.dylib.compatibility_version == 0xffffffff)
    outs() << "n/a\n";
  else
    outs() << ((dl.dylib.compatibility_version >> 16) & 0xffff) << "."
           << ((dl.dylib.compatibility_version >> 8) & 0xff) << "."
           << (dl.dylib.compatibility_version & 0xff) << "\n";
}

static void PrintLinkEditDataCommand(MachO::linkedit_data_command ld,
                                     uint32_t object_size) {
  if (ld.cmd == MachO::LC_CODE_SIGNATURE)
    outs() << "      cmd LC_FUNCTION_STARTS\n";
  else if (ld.cmd == MachO::LC_SEGMENT_SPLIT_INFO)
    outs() << "      cmd LC_SEGMENT_SPLIT_INFO\n";
  else if (ld.cmd == MachO::LC_FUNCTION_STARTS)
    outs() << "      cmd LC_FUNCTION_STARTS\n";
  else if (ld.cmd == MachO::LC_DATA_IN_CODE)
    outs() << "      cmd LC_DATA_IN_CODE\n";
  else if (ld.cmd == MachO::LC_DYLIB_CODE_SIGN_DRS)
    outs() << "      cmd LC_DYLIB_CODE_SIGN_DRS\n";
  else if (ld.cmd == MachO::LC_LINKER_OPTIMIZATION_HINT)
    outs() << "      cmd LC_LINKER_OPTIMIZATION_HINT\n";
  else
    outs() << "      cmd " << ld.cmd << " (?)\n";
  outs() << "  cmdsize " << ld.cmdsize;
  if (ld.cmdsize != sizeof(struct MachO::linkedit_data_command))
    outs() << " Incorrect size\n";
  else
    outs() << "\n";
  outs() << "  dataoff " << ld.dataoff;
  if (ld.dataoff > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
  outs() << " datasize " << ld.datasize;
  uint64_t big_size = ld.dataoff;
  big_size += ld.datasize;
  if (big_size > object_size)
    outs() << " (past end of file)\n";
  else
    outs() << "\n";
}

static void PrintLoadCommands(const MachOObjectFile *Obj, uint32_t ncmds,
                              uint32_t filetype, uint32_t cputype,
                              bool verbose) {
  StringRef Buf = Obj->getData();
  MachOObjectFile::LoadCommandInfo Command = Obj->getFirstLoadCommandInfo();
  for (unsigned i = 0;; ++i) {
    outs() << "Load command " << i << "\n";
    if (Command.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command SLC = Obj->getSegmentLoadCommand(Command);
      const char *sg_segname = SLC.segname;
      PrintSegmentCommand(SLC.cmd, SLC.cmdsize, SLC.segname, SLC.vmaddr,
                          SLC.vmsize, SLC.fileoff, SLC.filesize, SLC.maxprot,
                          SLC.initprot, SLC.nsects, SLC.flags, Buf.size(),
                          verbose);
      for (unsigned j = 0; j < SLC.nsects; j++) {
        MachO::section_64 S = Obj->getSection64(Command, j);
        PrintSection(S.sectname, S.segname, S.addr, S.size, S.offset, S.align,
                     S.reloff, S.nreloc, S.flags, S.reserved1, S.reserved2,
                     SLC.cmd, sg_segname, filetype, Buf.size(), verbose);
      }
    } else if (Command.C.cmd == MachO::LC_SEGMENT_64) {
      MachO::segment_command_64 SLC_64 = Obj->getSegment64LoadCommand(Command);
      const char *sg_segname = SLC_64.segname;
      PrintSegmentCommand(SLC_64.cmd, SLC_64.cmdsize, SLC_64.segname,
                          SLC_64.vmaddr, SLC_64.vmsize, SLC_64.fileoff,
                          SLC_64.filesize, SLC_64.maxprot, SLC_64.initprot,
                          SLC_64.nsects, SLC_64.flags, Buf.size(), verbose);
      for (unsigned j = 0; j < SLC_64.nsects; j++) {
        MachO::section_64 S_64 = Obj->getSection64(Command, j);
        PrintSection(S_64.sectname, S_64.segname, S_64.addr, S_64.size,
                     S_64.offset, S_64.align, S_64.reloff, S_64.nreloc,
                     S_64.flags, S_64.reserved1, S_64.reserved2, SLC_64.cmd,
                     sg_segname, filetype, Buf.size(), verbose);
      }
    } else if (Command.C.cmd == MachO::LC_SYMTAB) {
      MachO::symtab_command Symtab = Obj->getSymtabLoadCommand();
      PrintSymtabLoadCommand(Symtab, cputype, Buf.size());
    } else if (Command.C.cmd == MachO::LC_DYSYMTAB) {
      MachO::dysymtab_command Dysymtab = Obj->getDysymtabLoadCommand();
      MachO::symtab_command Symtab = Obj->getSymtabLoadCommand();
      PrintDysymtabLoadCommand(Dysymtab, Symtab.nsyms, Buf.size(), cputype);
    } else if (Command.C.cmd == MachO::LC_DYLD_INFO ||
               Command.C.cmd == MachO::LC_DYLD_INFO_ONLY) {
      MachO::dyld_info_command DyldInfo = Obj->getDyldInfoLoadCommand(Command);
      PrintDyldInfoLoadCommand(DyldInfo, Buf.size());
    } else if (Command.C.cmd == MachO::LC_LOAD_DYLINKER ||
               Command.C.cmd == MachO::LC_ID_DYLINKER ||
               Command.C.cmd == MachO::LC_DYLD_ENVIRONMENT) {
      MachO::dylinker_command Dyld = Obj->getDylinkerCommand(Command);
      PrintDyldLoadCommand(Dyld, Command.Ptr);
    } else if (Command.C.cmd == MachO::LC_UUID) {
      MachO::uuid_command Uuid = Obj->getUuidCommand(Command);
      PrintUuidLoadCommand(Uuid);
    } else if (Command.C.cmd == MachO::LC_VERSION_MIN_MACOSX) {
      MachO::version_min_command Vd = Obj->getVersionMinLoadCommand(Command);
      PrintVersionMinLoadCommand(Vd);
    } else if (Command.C.cmd == MachO::LC_SOURCE_VERSION) {
      MachO::source_version_command Sd = Obj->getSourceVersionCommand(Command);
      PrintSourceVersionCommand(Sd);
    } else if (Command.C.cmd == MachO::LC_MAIN) {
      MachO::entry_point_command Ep = Obj->getEntryPointCommand(Command);
      PrintEntryPointCommand(Ep);
    } else if (Command.C.cmd == MachO::LC_LOAD_DYLIB ||
               Command.C.cmd == MachO::LC_ID_DYLIB ||
               Command.C.cmd == MachO::LC_LOAD_WEAK_DYLIB ||
               Command.C.cmd == MachO::LC_REEXPORT_DYLIB ||
               Command.C.cmd == MachO::LC_LAZY_LOAD_DYLIB ||
               Command.C.cmd == MachO::LC_LOAD_UPWARD_DYLIB) {
      MachO::dylib_command Dl = Obj->getDylibIDLoadCommand(Command);
      PrintDylibCommand(Dl, Command.Ptr);
    } else if (Command.C.cmd == MachO::LC_CODE_SIGNATURE ||
               Command.C.cmd == MachO::LC_SEGMENT_SPLIT_INFO ||
               Command.C.cmd == MachO::LC_FUNCTION_STARTS ||
               Command.C.cmd == MachO::LC_DATA_IN_CODE ||
               Command.C.cmd == MachO::LC_DYLIB_CODE_SIGN_DRS ||
               Command.C.cmd == MachO::LC_LINKER_OPTIMIZATION_HINT) {
      MachO::linkedit_data_command Ld =
          Obj->getLinkeditDataLoadCommand(Command);
      PrintLinkEditDataCommand(Ld, Buf.size());
    } else {
      outs() << "      cmd ?(" << format("0x%08" PRIx32, Command.C.cmd)
             << ")\n";
      outs() << "  cmdsize " << Command.C.cmdsize << "\n";
      // TODO: get and print the raw bytes of the load command.
    }
    // TODO: print all the other kinds of load commands.
    if (i == ncmds - 1)
      break;
    else
      Command = Obj->getNextLoadCommandInfo(Command);
  }
}

static void getAndPrintMachHeader(const MachOObjectFile *Obj, uint32_t &ncmds,
                                  uint32_t &filetype, uint32_t &cputype,
                                  bool verbose) {
  if (Obj->is64Bit()) {
    MachO::mach_header_64 H_64;
    H_64 = Obj->getHeader64();
    PrintMachHeader(H_64.magic, H_64.cputype, H_64.cpusubtype, H_64.filetype,
                    H_64.ncmds, H_64.sizeofcmds, H_64.flags, verbose);
    ncmds = H_64.ncmds;
    filetype = H_64.filetype;
    cputype = H_64.cputype;
  } else {
    MachO::mach_header H;
    H = Obj->getHeader();
    PrintMachHeader(H.magic, H.cputype, H.cpusubtype, H.filetype, H.ncmds,
                    H.sizeofcmds, H.flags, verbose);
    ncmds = H.ncmds;
    filetype = H.filetype;
    cputype = H.cputype;
  }
}

void llvm::printMachOFileHeader(const object::ObjectFile *Obj) {
  const MachOObjectFile *file = dyn_cast<const MachOObjectFile>(Obj);
  uint32_t ncmds = 0;
  uint32_t filetype = 0;
  uint32_t cputype = 0;
  getAndPrintMachHeader(file, ncmds, filetype, cputype, true);
  PrintLoadCommands(file, ncmds, filetype, cputype, true);
}

//===----------------------------------------------------------------------===//
// export trie dumping
//===----------------------------------------------------------------------===//

void llvm::printMachOExportsTrie(const object::MachOObjectFile *Obj) {
  for (const llvm::object::ExportEntry &Entry : Obj->exports()) {
    uint64_t Flags = Entry.flags();
    bool ReExport = (Flags & MachO::EXPORT_SYMBOL_FLAGS_REEXPORT);
    bool WeakDef = (Flags & MachO::EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION);
    bool ThreadLocal = ((Flags & MachO::EXPORT_SYMBOL_FLAGS_KIND_MASK) ==
                        MachO::EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL);
    bool Abs = ((Flags & MachO::EXPORT_SYMBOL_FLAGS_KIND_MASK) ==
                MachO::EXPORT_SYMBOL_FLAGS_KIND_ABSOLUTE);
    bool Resolver = (Flags & MachO::EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER);
    if (ReExport)
      outs() << "[re-export] ";
    else
      outs() << format("0x%08llX  ",
                       Entry.address()); // FIXME:add in base address
    outs() << Entry.name();
    if (WeakDef || ThreadLocal || Resolver || Abs) {
      bool NeedsComma = false;
      outs() << " [";
      if (WeakDef) {
        outs() << "weak_def";
        NeedsComma = true;
      }
      if (ThreadLocal) {
        if (NeedsComma)
          outs() << ", ";
        outs() << "per-thread";
        NeedsComma = true;
      }
      if (Abs) {
        if (NeedsComma)
          outs() << ", ";
        outs() << "absolute";
        NeedsComma = true;
      }
      if (Resolver) {
        if (NeedsComma)
          outs() << ", ";
        outs() << format("resolver=0x%08llX", Entry.other());
        NeedsComma = true;
      }
      outs() << "]";
    }
    if (ReExport) {
      StringRef DylibName = "unknown";
      int Ordinal = Entry.other() - 1;
      Obj->getLibraryShortNameByIndex(Ordinal, DylibName);
      if (Entry.otherName().empty())
        outs() << " (from " << DylibName << ")";
      else
        outs() << " (" << Entry.otherName() << " from " << DylibName << ")";
    }
    outs() << "\n";
  }
}

//===----------------------------------------------------------------------===//
// rebase table dumping
//===----------------------------------------------------------------------===//

namespace {
class SegInfo {
public:
  SegInfo(const object::MachOObjectFile *Obj);

  StringRef segmentName(uint32_t SegIndex);
  StringRef sectionName(uint32_t SegIndex, uint64_t SegOffset);
  uint64_t address(uint32_t SegIndex, uint64_t SegOffset);

private:
  struct SectionInfo {
    uint64_t Address;
    uint64_t Size;
    StringRef SectionName;
    StringRef SegmentName;
    uint64_t OffsetInSegment;
    uint64_t SegmentStartAddress;
    uint32_t SegmentIndex;
  };
  const SectionInfo &findSection(uint32_t SegIndex, uint64_t SegOffset);
  SmallVector<SectionInfo, 32> Sections;
};
}

SegInfo::SegInfo(const object::MachOObjectFile *Obj) {
  // Build table of sections so segIndex/offset pairs can be translated.
  uint32_t CurSegIndex = Obj->hasPageZeroSegment() ? 1 : 0;
  StringRef CurSegName;
  uint64_t CurSegAddress;
  for (const SectionRef &Section : Obj->sections()) {
    SectionInfo Info;
    if (error(Section.getName(Info.SectionName)))
      return;
    Info.Address = Section.getAddress();
    Info.Size = Section.getSize();
    Info.SegmentName =
        Obj->getSectionFinalSegmentName(Section.getRawDataRefImpl());
    if (!Info.SegmentName.equals(CurSegName)) {
      ++CurSegIndex;
      CurSegName = Info.SegmentName;
      CurSegAddress = Info.Address;
    }
    Info.SegmentIndex = CurSegIndex - 1;
    Info.OffsetInSegment = Info.Address - CurSegAddress;
    Info.SegmentStartAddress = CurSegAddress;
    Sections.push_back(Info);
  }
}

StringRef SegInfo::segmentName(uint32_t SegIndex) {
  for (const SectionInfo &SI : Sections) {
    if (SI.SegmentIndex == SegIndex)
      return SI.SegmentName;
  }
  llvm_unreachable("invalid segIndex");
}

const SegInfo::SectionInfo &SegInfo::findSection(uint32_t SegIndex,
                                                 uint64_t OffsetInSeg) {
  for (const SectionInfo &SI : Sections) {
    if (SI.SegmentIndex != SegIndex)
      continue;
    if (SI.OffsetInSegment > OffsetInSeg)
      continue;
    if (OffsetInSeg >= (SI.OffsetInSegment + SI.Size))
      continue;
    return SI;
  }
  llvm_unreachable("segIndex and offset not in any section");
}

StringRef SegInfo::sectionName(uint32_t SegIndex, uint64_t OffsetInSeg) {
  return findSection(SegIndex, OffsetInSeg).SectionName;
}

uint64_t SegInfo::address(uint32_t SegIndex, uint64_t OffsetInSeg) {
  const SectionInfo &SI = findSection(SegIndex, OffsetInSeg);
  return SI.SegmentStartAddress + OffsetInSeg;
}

void llvm::printMachORebaseTable(const object::MachOObjectFile *Obj) {
  // Build table of sections so names can used in final output.
  SegInfo sectionTable(Obj);

  outs() << "segment  section            address     type\n";
  for (const llvm::object::MachORebaseEntry &Entry : Obj->rebaseTable()) {
    uint32_t SegIndex = Entry.segmentIndex();
    uint64_t OffsetInSeg = Entry.segmentOffset();
    StringRef SegmentName = sectionTable.segmentName(SegIndex);
    StringRef SectionName = sectionTable.sectionName(SegIndex, OffsetInSeg);
    uint64_t Address = sectionTable.address(SegIndex, OffsetInSeg);

    // Table lines look like: __DATA  __nl_symbol_ptr  0x0000F00C  pointer
    outs() << format("%-8s %-18s 0x%08" PRIX64 "  %s\n",
                     SegmentName.str().c_str(), SectionName.str().c_str(),
                     Address, Entry.typeName().str().c_str());
  }
}

static StringRef ordinalName(const object::MachOObjectFile *Obj, int Ordinal) {
  StringRef DylibName;
  switch (Ordinal) {
  case MachO::BIND_SPECIAL_DYLIB_SELF:
    return "this-image";
  case MachO::BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE:
    return "main-executable";
  case MachO::BIND_SPECIAL_DYLIB_FLAT_LOOKUP:
    return "flat-namespace";
  default:
    if (Ordinal > 0) {
      std::error_code EC =
          Obj->getLibraryShortNameByIndex(Ordinal - 1, DylibName);
      if (EC)
        return "<<bad library ordinal>>";
      return DylibName;
    }
  }
  return "<<unknown special ordinal>>";
}

//===----------------------------------------------------------------------===//
// bind table dumping
//===----------------------------------------------------------------------===//

void llvm::printMachOBindTable(const object::MachOObjectFile *Obj) {
  // Build table of sections so names can used in final output.
  SegInfo sectionTable(Obj);

  outs() << "segment  section            address    type       "
            "addend dylib            symbol\n";
  for (const llvm::object::MachOBindEntry &Entry : Obj->bindTable()) {
    uint32_t SegIndex = Entry.segmentIndex();
    uint64_t OffsetInSeg = Entry.segmentOffset();
    StringRef SegmentName = sectionTable.segmentName(SegIndex);
    StringRef SectionName = sectionTable.sectionName(SegIndex, OffsetInSeg);
    uint64_t Address = sectionTable.address(SegIndex, OffsetInSeg);

    // Table lines look like:
    //  __DATA  __got  0x00012010    pointer   0 libSystem ___stack_chk_guard
    StringRef Attr;
    if (Entry.flags() & MachO::BIND_SYMBOL_FLAGS_WEAK_IMPORT)
      Attr = " (weak_import)";
    outs() << left_justify(SegmentName, 8) << " "
           << left_justify(SectionName, 18) << " "
           << format_hex(Address, 10, true) << " "
           << left_justify(Entry.typeName(), 8) << " "
           << format_decimal(Entry.addend(), 8) << " "
           << left_justify(ordinalName(Obj, Entry.ordinal()), 16) << " "
           << Entry.symbolName() << Attr << "\n";
  }
}

//===----------------------------------------------------------------------===//
// lazy bind table dumping
//===----------------------------------------------------------------------===//

void llvm::printMachOLazyBindTable(const object::MachOObjectFile *Obj) {
  // Build table of sections so names can used in final output.
  SegInfo sectionTable(Obj);

  outs() << "segment  section            address     "
            "dylib            symbol\n";
  for (const llvm::object::MachOBindEntry &Entry : Obj->lazyBindTable()) {
    uint32_t SegIndex = Entry.segmentIndex();
    uint64_t OffsetInSeg = Entry.segmentOffset();
    StringRef SegmentName = sectionTable.segmentName(SegIndex);
    StringRef SectionName = sectionTable.sectionName(SegIndex, OffsetInSeg);
    uint64_t Address = sectionTable.address(SegIndex, OffsetInSeg);

    // Table lines look like:
    //  __DATA  __got  0x00012010 libSystem ___stack_chk_guard
    outs() << left_justify(SegmentName, 8) << " "
           << left_justify(SectionName, 18) << " "
           << format_hex(Address, 10, true) << " "
           << left_justify(ordinalName(Obj, Entry.ordinal()), 16) << " "
           << Entry.symbolName() << "\n";
  }
}

//===----------------------------------------------------------------------===//
// weak bind table dumping
//===----------------------------------------------------------------------===//

void llvm::printMachOWeakBindTable(const object::MachOObjectFile *Obj) {
  // Build table of sections so names can used in final output.
  SegInfo sectionTable(Obj);

  outs() << "segment  section            address     "
            "type       addend   symbol\n";
  for (const llvm::object::MachOBindEntry &Entry : Obj->weakBindTable()) {
    // Strong symbols don't have a location to update.
    if (Entry.flags() & MachO::BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION) {
      outs() << "                                        strong              "
             << Entry.symbolName() << "\n";
      continue;
    }
    uint32_t SegIndex = Entry.segmentIndex();
    uint64_t OffsetInSeg = Entry.segmentOffset();
    StringRef SegmentName = sectionTable.segmentName(SegIndex);
    StringRef SectionName = sectionTable.sectionName(SegIndex, OffsetInSeg);
    uint64_t Address = sectionTable.address(SegIndex, OffsetInSeg);

    // Table lines look like:
    // __DATA  __data  0x00001000  pointer    0   _foo
    outs() << left_justify(SegmentName, 8) << " "
           << left_justify(SectionName, 18) << " "
           << format_hex(Address, 10, true) << " "
           << left_justify(Entry.typeName(), 8) << " "
           << format_decimal(Entry.addend(), 8) << "   " << Entry.symbolName()
           << "\n";
  }
}

// get_dyld_bind_info_symbolname() is used for disassembly and passed an
// address, ReferenceValue, in the Mach-O file and looks in the dyld bind
// information for that address. If the address is found its binding symbol
// name is returned.  If not nullptr is returned.
static const char *get_dyld_bind_info_symbolname(uint64_t ReferenceValue,
                                                 struct DisassembleInfo *info) {
  if (info->bindtable == nullptr) {
    info->bindtable = new (BindTable);
    SegInfo sectionTable(info->O);
    for (const llvm::object::MachOBindEntry &Entry : info->O->bindTable()) {
      uint32_t SegIndex = Entry.segmentIndex();
      uint64_t OffsetInSeg = Entry.segmentOffset();
      uint64_t Address = sectionTable.address(SegIndex, OffsetInSeg);
      const char *SymbolName = nullptr;
      StringRef name = Entry.symbolName();
      if (!name.empty())
        SymbolName = name.data();
      info->bindtable->push_back(std::make_pair(Address, SymbolName));
    }
  }
  for (bind_table_iterator BI = info->bindtable->begin(),
                           BE = info->bindtable->end();
       BI != BE; ++BI) {
    uint64_t Address = BI->first;
    if (ReferenceValue == Address) {
      const char *SymbolName = BI->second;
      return SymbolName;
    }
  }
  return nullptr;
}
