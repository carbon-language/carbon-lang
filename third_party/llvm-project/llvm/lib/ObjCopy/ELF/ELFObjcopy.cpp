//===- ELFObjcopy.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjCopy/ELF/ELFObjcopy.h"
#include "ELFObject.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/ObjCopy/CommonConfig.h"
#include "llvm/ObjCopy/ELF/ELFConfig.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::objcopy;
using namespace llvm::objcopy::elf;
using namespace llvm::object;

using SectionPred = std::function<bool(const SectionBase &Sec)>;

static bool isDebugSection(const SectionBase &Sec) {
  return StringRef(Sec.Name).startswith(".debug") ||
         StringRef(Sec.Name).startswith(".zdebug") || Sec.Name == ".gdb_index";
}

static bool isDWOSection(const SectionBase &Sec) {
  return StringRef(Sec.Name).endswith(".dwo");
}

static bool onlyKeepDWOPred(const Object &Obj, const SectionBase &Sec) {
  // We can't remove the section header string table.
  if (&Sec == Obj.SectionNames)
    return false;
  // Short of keeping the string table we want to keep everything that is a DWO
  // section and remove everything else.
  return !isDWOSection(Sec);
}

static uint64_t getNewShfFlags(SectionFlag AllFlags) {
  uint64_t NewFlags = 0;
  if (AllFlags & SectionFlag::SecAlloc)
    NewFlags |= ELF::SHF_ALLOC;
  if (!(AllFlags & SectionFlag::SecReadonly))
    NewFlags |= ELF::SHF_WRITE;
  if (AllFlags & SectionFlag::SecCode)
    NewFlags |= ELF::SHF_EXECINSTR;
  if (AllFlags & SectionFlag::SecMerge)
    NewFlags |= ELF::SHF_MERGE;
  if (AllFlags & SectionFlag::SecStrings)
    NewFlags |= ELF::SHF_STRINGS;
  if (AllFlags & SectionFlag::SecExclude)
    NewFlags |= ELF::SHF_EXCLUDE;
  return NewFlags;
}

static uint64_t getSectionFlagsPreserveMask(uint64_t OldFlags,
                                            uint64_t NewFlags) {
  // Preserve some flags which should not be dropped when setting flags.
  // Also, preserve anything OS/processor dependant.
  const uint64_t PreserveMask =
      (ELF::SHF_COMPRESSED | ELF::SHF_GROUP | ELF::SHF_LINK_ORDER |
       ELF::SHF_MASKOS | ELF::SHF_MASKPROC | ELF::SHF_TLS |
       ELF::SHF_INFO_LINK) &
      ~ELF::SHF_EXCLUDE;
  return (OldFlags & PreserveMask) | (NewFlags & ~PreserveMask);
}

static void setSectionFlagsAndType(SectionBase &Sec, SectionFlag Flags) {
  Sec.Flags = getSectionFlagsPreserveMask(Sec.Flags, getNewShfFlags(Flags));

  // In GNU objcopy, certain flags promote SHT_NOBITS to SHT_PROGBITS. This rule
  // may promote more non-ALLOC sections than GNU objcopy, but it is fine as
  // non-ALLOC SHT_NOBITS sections do not make much sense.
  if (Sec.Type == SHT_NOBITS &&
      (!(Sec.Flags & ELF::SHF_ALLOC) ||
       Flags & (SectionFlag::SecContents | SectionFlag::SecLoad)))
    Sec.Type = SHT_PROGBITS;
}

static ElfType getOutputElfType(const Binary &Bin) {
  // Infer output ELF type from the input ELF object
  if (isa<ELFObjectFile<ELF32LE>>(Bin))
    return ELFT_ELF32LE;
  if (isa<ELFObjectFile<ELF64LE>>(Bin))
    return ELFT_ELF64LE;
  if (isa<ELFObjectFile<ELF32BE>>(Bin))
    return ELFT_ELF32BE;
  if (isa<ELFObjectFile<ELF64BE>>(Bin))
    return ELFT_ELF64BE;
  llvm_unreachable("Invalid ELFType");
}

static ElfType getOutputElfType(const MachineInfo &MI) {
  // Infer output ELF type from the binary arch specified
  if (MI.Is64Bit)
    return MI.IsLittleEndian ? ELFT_ELF64LE : ELFT_ELF64BE;
  else
    return MI.IsLittleEndian ? ELFT_ELF32LE : ELFT_ELF32BE;
}

static std::unique_ptr<Writer> createELFWriter(const CommonConfig &Config,
                                               Object &Obj, raw_ostream &Out,
                                               ElfType OutputElfType) {
  // Depending on the initial ELFT and OutputFormat we need a different Writer.
  switch (OutputElfType) {
  case ELFT_ELF32LE:
    return std::make_unique<ELFWriter<ELF32LE>>(Obj, Out, !Config.StripSections,
                                                Config.OnlyKeepDebug);
  case ELFT_ELF64LE:
    return std::make_unique<ELFWriter<ELF64LE>>(Obj, Out, !Config.StripSections,
                                                Config.OnlyKeepDebug);
  case ELFT_ELF32BE:
    return std::make_unique<ELFWriter<ELF32BE>>(Obj, Out, !Config.StripSections,
                                                Config.OnlyKeepDebug);
  case ELFT_ELF64BE:
    return std::make_unique<ELFWriter<ELF64BE>>(Obj, Out, !Config.StripSections,
                                                Config.OnlyKeepDebug);
  }
  llvm_unreachable("Invalid output format");
}

static std::unique_ptr<Writer> createWriter(const CommonConfig &Config,
                                            Object &Obj, raw_ostream &Out,
                                            ElfType OutputElfType) {
  switch (Config.OutputFormat) {
  case FileFormat::Binary:
    return std::make_unique<BinaryWriter>(Obj, Out);
  case FileFormat::IHex:
    return std::make_unique<IHexWriter>(Obj, Out);
  default:
    return createELFWriter(Config, Obj, Out, OutputElfType);
  }
}

template <class... Ts>
static Error makeStringError(std::error_code EC, const Twine &Msg,
                             Ts &&...Args) {
  std::string FullMsg = (EC.message() + ": " + Msg).str();
  return createStringError(EC, FullMsg.c_str(), std::forward<Ts>(Args)...);
}

static Error dumpSectionToFile(StringRef SecName, StringRef Filename,
                               Object &Obj) {
  for (auto &Sec : Obj.sections()) {
    if (Sec.Name == SecName) {
      if (Sec.Type == SHT_NOBITS)
        return createStringError(object_error::parse_failed,
                                 "cannot dump section '%s': it has no contents",
                                 SecName.str().c_str());
      Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
          FileOutputBuffer::create(Filename, Sec.OriginalData.size());
      if (!BufferOrErr)
        return BufferOrErr.takeError();
      std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufferOrErr);
      std::copy(Sec.OriginalData.begin(), Sec.OriginalData.end(),
                Buf->getBufferStart());
      if (Error E = Buf->commit())
        return E;
      return Error::success();
    }
  }
  return createStringError(object_error::parse_failed, "section '%s' not found",
                           SecName.str().c_str());
}

static bool isCompressable(const SectionBase &Sec) {
  return !(Sec.Flags & ELF::SHF_COMPRESSED) &&
         StringRef(Sec.Name).startswith(".debug");
}

static Error replaceDebugSections(
    Object &Obj, function_ref<bool(const SectionBase &)> ShouldReplace,
    function_ref<Expected<SectionBase *>(const SectionBase *)> AddSection) {
  // Build a list of the debug sections we are going to replace.
  // We can't call `AddSection` while iterating over sections,
  // because it would mutate the sections array.
  SmallVector<SectionBase *, 13> ToReplace;
  for (auto &Sec : Obj.sections())
    if (ShouldReplace(Sec))
      ToReplace.push_back(&Sec);

  // Build a mapping from original section to a new one.
  DenseMap<SectionBase *, SectionBase *> FromTo;
  for (SectionBase *S : ToReplace) {
    Expected<SectionBase *> NewSection = AddSection(S);
    if (!NewSection)
      return NewSection.takeError();

    FromTo[S] = *NewSection;
  }

  return Obj.replaceSections(FromTo);
}

static bool isAArch64MappingSymbol(const Symbol &Sym) {
  if (Sym.Binding != STB_LOCAL || Sym.Type != STT_NOTYPE ||
      Sym.getShndx() == SHN_UNDEF)
    return false;
  StringRef Name = Sym.Name;
  if (!Name.consume_front("$x") && !Name.consume_front("$d"))
    return false;
  return Name.empty() || Name.startswith(".");
}

static bool isArmMappingSymbol(const Symbol &Sym) {
  if (Sym.Binding != STB_LOCAL || Sym.Type != STT_NOTYPE ||
      Sym.getShndx() == SHN_UNDEF)
    return false;
  StringRef Name = Sym.Name;
  if (!Name.consume_front("$a") && !Name.consume_front("$d") &&
      !Name.consume_front("$t"))
    return false;
  return Name.empty() || Name.startswith(".");
}

// Check if the symbol should be preserved because it is required by ABI.
static bool isRequiredByABISymbol(const Object &Obj, const Symbol &Sym) {
  switch (Obj.Machine) {
  case EM_AARCH64:
    // Mapping symbols should be preserved for a relocatable object file.
    return Obj.isRelocatable() && isAArch64MappingSymbol(Sym);
  case EM_ARM:
    // Mapping symbols should be preserved for a relocatable object file.
    return Obj.isRelocatable() && isArmMappingSymbol(Sym);
  default:
    return false;
  }
}

static bool isUnneededSymbol(const Symbol &Sym) {
  return !Sym.Referenced &&
         (Sym.Binding == STB_LOCAL || Sym.getShndx() == SHN_UNDEF) &&
         Sym.Type != STT_SECTION;
}

static Error updateAndRemoveSymbols(const CommonConfig &Config,
                                    const ELFConfig &ELFConfig, Object &Obj) {
  // TODO: update or remove symbols only if there is an option that affects
  // them.
  if (!Obj.SymbolTable)
    return Error::success();

  Obj.SymbolTable->updateSymbols([&](Symbol &Sym) {
    // Common and undefined symbols don't make sense as local symbols, and can
    // even cause crashes if we localize those, so skip them.
    if (!Sym.isCommon() && Sym.getShndx() != SHN_UNDEF &&
        ((ELFConfig.LocalizeHidden &&
          (Sym.Visibility == STV_HIDDEN || Sym.Visibility == STV_INTERNAL)) ||
         Config.SymbolsToLocalize.matches(Sym.Name)))
      Sym.Binding = STB_LOCAL;

    // Note: these two globalize flags have very similar names but different
    // meanings:
    //
    // --globalize-symbol: promote a symbol to global
    // --keep-global-symbol: all symbols except for these should be made local
    //
    // If --globalize-symbol is specified for a given symbol, it will be
    // global in the output file even if it is not included via
    // --keep-global-symbol. Because of that, make sure to check
    // --globalize-symbol second.
    if (!Config.SymbolsToKeepGlobal.empty() &&
        !Config.SymbolsToKeepGlobal.matches(Sym.Name) &&
        Sym.getShndx() != SHN_UNDEF)
      Sym.Binding = STB_LOCAL;

    if (Config.SymbolsToGlobalize.matches(Sym.Name) &&
        Sym.getShndx() != SHN_UNDEF)
      Sym.Binding = STB_GLOBAL;

    if (Config.SymbolsToWeaken.matches(Sym.Name) && Sym.Binding == STB_GLOBAL)
      Sym.Binding = STB_WEAK;

    if (Config.Weaken && Sym.Binding == STB_GLOBAL &&
        Sym.getShndx() != SHN_UNDEF)
      Sym.Binding = STB_WEAK;

    const auto I = Config.SymbolsToRename.find(Sym.Name);
    if (I != Config.SymbolsToRename.end())
      Sym.Name = std::string(I->getValue());

    if (!Config.SymbolsPrefix.empty() && Sym.Type != STT_SECTION)
      Sym.Name = (Config.SymbolsPrefix + Sym.Name).str();
  });

  // The purpose of this loop is to mark symbols referenced by sections
  // (like GroupSection or RelocationSection). This way, we know which
  // symbols are still 'needed' and which are not.
  if (Config.StripUnneeded || !Config.UnneededSymbolsToRemove.empty() ||
      !Config.OnlySection.empty()) {
    for (SectionBase &Sec : Obj.sections())
      Sec.markSymbols();
  }

  auto RemoveSymbolsPred = [&](const Symbol &Sym) {
    if (Config.SymbolsToKeep.matches(Sym.Name) ||
        (ELFConfig.KeepFileSymbols && Sym.Type == STT_FILE))
      return false;

    if (Config.SymbolsToRemove.matches(Sym.Name))
      return true;

    if (Config.StripAll || Config.StripAllGNU)
      return true;

    if (isRequiredByABISymbol(Obj, Sym))
      return false;

    if (Config.StripDebug && Sym.Type == STT_FILE)
      return true;

    if ((Config.DiscardMode == DiscardType::All ||
         (Config.DiscardMode == DiscardType::Locals &&
          StringRef(Sym.Name).startswith(".L"))) &&
        Sym.Binding == STB_LOCAL && Sym.getShndx() != SHN_UNDEF &&
        Sym.Type != STT_FILE && Sym.Type != STT_SECTION)
      return true;

    if ((Config.StripUnneeded ||
         Config.UnneededSymbolsToRemove.matches(Sym.Name)) &&
        (!Obj.isRelocatable() || isUnneededSymbol(Sym)))
      return true;

    // We want to remove undefined symbols if all references have been stripped.
    if (!Config.OnlySection.empty() && !Sym.Referenced &&
        Sym.getShndx() == SHN_UNDEF)
      return true;

    return false;
  };

  return Obj.removeSymbols(RemoveSymbolsPred);
}

static Error replaceAndRemoveSections(const CommonConfig &Config,
                                      const ELFConfig &ELFConfig, Object &Obj) {
  SectionPred RemovePred = [](const SectionBase &) { return false; };

  // Removes:
  if (!Config.ToRemove.empty()) {
    RemovePred = [&Config](const SectionBase &Sec) {
      return Config.ToRemove.matches(Sec.Name);
    };
  }

  if (Config.StripDWO)
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return isDWOSection(Sec) || RemovePred(Sec);
    };

  if (Config.ExtractDWO)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      return onlyKeepDWOPred(Obj, Sec) || RemovePred(Sec);
    };

  if (Config.StripAllGNU)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if ((Sec.Flags & SHF_ALLOC) != 0)
        return false;
      if (&Sec == Obj.SectionNames)
        return false;
      switch (Sec.Type) {
      case SHT_SYMTAB:
      case SHT_REL:
      case SHT_RELA:
      case SHT_STRTAB:
        return true;
      }
      return isDebugSection(Sec);
    };

  if (Config.StripSections) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || Sec.ParentSegment == nullptr;
    };
  }

  if (Config.StripDebug || Config.StripUnneeded) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || isDebugSection(Sec);
    };
  }

  if (Config.StripNonAlloc)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj.SectionNames)
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0 && Sec.ParentSegment == nullptr;
    };

  if (Config.StripAll)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj.SectionNames)
        return false;
      if (StringRef(Sec.Name).startswith(".gnu.warning"))
        return false;
      // We keep the .ARM.attribute section to maintain compatibility
      // with Debian derived distributions. This is a bug in their
      // patchset as documented here:
      // https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=943798
      if (Sec.Type == SHT_ARM_ATTRIBUTES)
        return false;
      if (Sec.ParentSegment != nullptr)
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  if (Config.ExtractPartition || Config.ExtractMainPartition) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (Sec.Type == SHT_LLVM_PART_EHDR || Sec.Type == SHT_LLVM_PART_PHDR)
        return true;
      return (Sec.Flags & SHF_ALLOC) != 0 && !Sec.ParentSegment;
    };
  }

  // Explicit copies:
  if (!Config.OnlySection.empty()) {
    RemovePred = [&Config, RemovePred, &Obj](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (Config.OnlySection.matches(Sec.Name))
        return false;

      // Allow all implicit removes.
      if (RemovePred(Sec))
        return true;

      // Keep special sections.
      if (Obj.SectionNames == &Sec)
        return false;
      if (Obj.SymbolTable == &Sec ||
          (Obj.SymbolTable && Obj.SymbolTable->getStrTab() == &Sec))
        return false;

      // Remove everything else.
      return true;
    };
  }

  if (!Config.KeepSection.empty()) {
    RemovePred = [&Config, RemovePred](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (Config.KeepSection.matches(Sec.Name))
        return false;
      // Otherwise defer to RemovePred.
      return RemovePred(Sec);
    };
  }

  // This has to be the last predicate assignment.
  // If the option --keep-symbol has been specified
  // and at least one of those symbols is present
  // (equivalently, the updated symbol table is not empty)
  // the symbol table and the string table should not be removed.
  if ((!Config.SymbolsToKeep.empty() || ELFConfig.KeepFileSymbols) &&
      Obj.SymbolTable && !Obj.SymbolTable->empty()) {
    RemovePred = [&Obj, RemovePred](const SectionBase &Sec) {
      if (&Sec == Obj.SymbolTable || &Sec == Obj.SymbolTable->getStrTab())
        return false;
      return RemovePred(Sec);
    };
  }

  if (Error E = Obj.removeSections(ELFConfig.AllowBrokenLinks, RemovePred))
    return E;

  if (Config.CompressionType != DebugCompressionType::None) {
    if (Error Err = replaceDebugSections(
            Obj, isCompressable,
            [&Config, &Obj](const SectionBase *S) -> Expected<SectionBase *> {
              Expected<CompressedSection> NewSection =
                  CompressedSection::create(*S, Config.CompressionType);
              if (!NewSection)
                return NewSection.takeError();

              return &Obj.addSection<CompressedSection>(std::move(*NewSection));
            }))
      return Err;
  } else if (Config.DecompressDebugSections) {
    if (Error Err = replaceDebugSections(
            Obj,
            [](const SectionBase &S) { return isa<CompressedSection>(&S); },
            [&Obj](const SectionBase *S) {
              const CompressedSection *CS = cast<CompressedSection>(S);
              return &Obj.addSection<DecompressedSection>(*CS);
            }))
      return Err;
  }

  return Error::success();
}

// Add symbol to the Object symbol table with the specified properties.
static void addSymbol(Object &Obj, const NewSymbolInfo &SymInfo,
                      uint8_t DefaultVisibility) {
  SectionBase *Sec = Obj.findSection(SymInfo.SectionName);
  uint64_t Value = Sec ? Sec->Addr + SymInfo.Value : SymInfo.Value;

  uint8_t Bind = ELF::STB_GLOBAL;
  uint8_t Type = ELF::STT_NOTYPE;
  uint8_t Visibility = DefaultVisibility;

  for (SymbolFlag FlagValue : SymInfo.Flags)
    switch (FlagValue) {
    case SymbolFlag::Global:
      Bind = ELF::STB_GLOBAL;
      break;
    case SymbolFlag::Local:
      Bind = ELF::STB_LOCAL;
      break;
    case SymbolFlag::Weak:
      Bind = ELF::STB_WEAK;
      break;
    case SymbolFlag::Default:
      Visibility = ELF::STV_DEFAULT;
      break;
    case SymbolFlag::Hidden:
      Visibility = ELF::STV_HIDDEN;
      break;
    case SymbolFlag::Protected:
      Visibility = ELF::STV_PROTECTED;
      break;
    case SymbolFlag::File:
      Type = ELF::STT_FILE;
      break;
    case SymbolFlag::Section:
      Type = ELF::STT_SECTION;
      break;
    case SymbolFlag::Object:
      Type = ELF::STT_OBJECT;
      break;
    case SymbolFlag::Function:
      Type = ELF::STT_FUNC;
      break;
    case SymbolFlag::IndirectFunction:
      Type = ELF::STT_GNU_IFUNC;
      break;
    default: /* Other flag values are ignored for ELF. */
      break;
    };

  Obj.SymbolTable->addSymbol(
      SymInfo.SymbolName, Bind, Type, Sec, Value, Visibility,
      Sec ? (uint16_t)SYMBOL_SIMPLE_INDEX : (uint16_t)SHN_ABS, 0);
}

static Error
handleUserSection(StringRef Flag,
                  function_ref<Error(StringRef, ArrayRef<uint8_t>)> F) {
  std::pair<StringRef, StringRef> SecPair = Flag.split("=");
  StringRef SecName = SecPair.first;
  StringRef File = SecPair.second;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr = MemoryBuffer::getFile(File);
  if (!BufOrErr)
    return createFileError(File, errorCodeToError(BufOrErr.getError()));
  std::unique_ptr<MemoryBuffer> Buf = std::move(*BufOrErr);
  ArrayRef<uint8_t> Data(
      reinterpret_cast<const uint8_t *>(Buf->getBufferStart()),
      Buf->getBufferSize());
  return F(SecName, Data);
}

// This function handles the high level operations of GNU objcopy including
// handling command line options. It's important to outline certain properties
// we expect to hold of the command line operations. Any operation that "keeps"
// should keep regardless of a remove. Additionally any removal should respect
// any previous removals. Lastly whether or not something is removed shouldn't
// depend a) on the order the options occur in or b) on some opaque priority
// system. The only priority is that keeps/copies overrule removes.
static Error handleArgs(const CommonConfig &Config, const ELFConfig &ELFConfig,
                        Object &Obj) {
  if (Config.OutputArch) {
    Obj.Machine = Config.OutputArch.getValue().EMachine;
    Obj.OSABI = Config.OutputArch.getValue().OSABI;
  }

  if (!Config.SplitDWO.empty() && Config.ExtractDWO) {
    return Obj.removeSections(
        ELFConfig.AllowBrokenLinks,
        [&Obj](const SectionBase &Sec) { return onlyKeepDWOPred(Obj, Sec); });
  }

  // Dump sections before add/remove for compatibility with GNU objcopy.
  for (StringRef Flag : Config.DumpSection) {
    StringRef SectionName;
    StringRef FileName;
    std::tie(SectionName, FileName) = Flag.split('=');
    if (Error E = dumpSectionToFile(SectionName, FileName, Obj))
      return E;
  }

  // It is important to remove the sections first. For example, we want to
  // remove the relocation sections before removing the symbols. That allows
  // us to avoid reporting the inappropriate errors about removing symbols
  // named in relocations.
  if (Error E = replaceAndRemoveSections(Config, ELFConfig, Obj))
    return E;

  if (Error E = updateAndRemoveSymbols(Config, ELFConfig, Obj))
    return E;

  if (!Config.SectionsToRename.empty()) {
    std::vector<RelocationSectionBase *> RelocSections;
    DenseSet<SectionBase *> RenamedSections;
    for (SectionBase &Sec : Obj.sections()) {
      auto *RelocSec = dyn_cast<RelocationSectionBase>(&Sec);
      const auto Iter = Config.SectionsToRename.find(Sec.Name);
      if (Iter != Config.SectionsToRename.end()) {
        const SectionRename &SR = Iter->second;
        Sec.Name = std::string(SR.NewName);
        if (SR.NewFlags.hasValue())
          setSectionFlagsAndType(Sec, SR.NewFlags.getValue());
        RenamedSections.insert(&Sec);
      } else if (RelocSec && !(Sec.Flags & SHF_ALLOC))
        // Postpone processing relocation sections which are not specified in
        // their explicit '--rename-section' commands until after their target
        // sections are renamed.
        // Dynamic relocation sections (i.e. ones with SHF_ALLOC) should be
        // renamed only explicitly. Otherwise, renaming, for example, '.got.plt'
        // would affect '.rela.plt', which is not desirable.
        RelocSections.push_back(RelocSec);
    }

    // Rename relocation sections according to their target sections.
    for (RelocationSectionBase *RelocSec : RelocSections) {
      auto Iter = RenamedSections.find(RelocSec->getSection());
      if (Iter != RenamedSections.end())
        RelocSec->Name = (RelocSec->getNamePrefix() + (*Iter)->Name).str();
    }
  }

  // Add a prefix to allocated sections and their relocation sections. This
  // should be done after renaming the section by Config.SectionToRename to
  // imitate the GNU objcopy behavior.
  if (!Config.AllocSectionsPrefix.empty()) {
    DenseSet<SectionBase *> PrefixedSections;
    for (SectionBase &Sec : Obj.sections()) {
      if (Sec.Flags & SHF_ALLOC) {
        Sec.Name = (Config.AllocSectionsPrefix + Sec.Name).str();
        PrefixedSections.insert(&Sec);
      } else if (auto *RelocSec = dyn_cast<RelocationSectionBase>(&Sec)) {
        // Rename relocation sections associated to the allocated sections.
        // For example, if we rename .text to .prefix.text, we also rename
        // .rel.text to .rel.prefix.text.
        //
        // Dynamic relocation sections (SHT_REL[A] with SHF_ALLOC) are handled
        // above, e.g., .rela.plt is renamed to .prefix.rela.plt, not
        // .rela.prefix.plt since GNU objcopy does so.
        const SectionBase *TargetSec = RelocSec->getSection();
        if (TargetSec && (TargetSec->Flags & SHF_ALLOC)) {
          // If the relocation section comes *after* the target section, we
          // don't add Config.AllocSectionsPrefix because we've already added
          // the prefix to TargetSec->Name. Otherwise, if the relocation
          // section comes *before* the target section, we add the prefix.
          if (PrefixedSections.count(TargetSec))
            Sec.Name = (RelocSec->getNamePrefix() + TargetSec->Name).str();
          else
            Sec.Name = (RelocSec->getNamePrefix() + Config.AllocSectionsPrefix +
                        TargetSec->Name)
                           .str();
        }
      }
    }
  }

  if (!Config.SetSectionAlignment.empty()) {
    for (SectionBase &Sec : Obj.sections()) {
      auto I = Config.SetSectionAlignment.find(Sec.Name);
      if (I != Config.SetSectionAlignment.end())
        Sec.Align = I->second;
    }
  }

  if (Config.OnlyKeepDebug)
    for (auto &Sec : Obj.sections())
      if (Sec.Flags & SHF_ALLOC && Sec.Type != SHT_NOTE)
        Sec.Type = SHT_NOBITS;

  for (const auto &Flag : Config.AddSection) {
    auto AddSection = [&](StringRef Name, ArrayRef<uint8_t> Data) {
      OwnedDataSection &NewSection =
          Obj.addSection<OwnedDataSection>(Name, Data);
      if (Name.startswith(".note") && Name != ".note.GNU-stack")
        NewSection.Type = SHT_NOTE;
      return Error::success();
    };
    if (Error E = handleUserSection(Flag, AddSection))
      return E;
  }

  for (StringRef Flag : Config.UpdateSection) {
    auto UpdateSection = [&](StringRef Name, ArrayRef<uint8_t> Data) {
      return Obj.updateSection(Name, Data);
    };
    if (Error E = handleUserSection(Flag, UpdateSection))
      return E;
  }

  if (!Config.AddGnuDebugLink.empty())
    Obj.addSection<GnuDebugLinkSection>(Config.AddGnuDebugLink,
                                        Config.GnuDebugLinkCRC32);

  // If the symbol table was previously removed, we need to create a new one
  // before adding new symbols.
  if (!Obj.SymbolTable && !Config.SymbolsToAdd.empty())
    if (Error E = Obj.addNewSymbolTable())
      return E;

  for (const NewSymbolInfo &SI : Config.SymbolsToAdd)
    addSymbol(Obj, SI, ELFConfig.NewSymbolVisibility);

  // --set-section-flags works with sections added by --add-section.
  if (!Config.SetSectionFlags.empty()) {
    for (auto &Sec : Obj.sections()) {
      const auto Iter = Config.SetSectionFlags.find(Sec.Name);
      if (Iter != Config.SetSectionFlags.end()) {
        const SectionFlagsUpdate &SFU = Iter->second;
        setSectionFlagsAndType(Sec, SFU.NewFlags);
      }
    }
  }

  if (ELFConfig.EntryExpr)
    Obj.Entry = ELFConfig.EntryExpr(Obj.Entry);
  return Error::success();
}

static Error writeOutput(const CommonConfig &Config, Object &Obj,
                         raw_ostream &Out, ElfType OutputElfType) {
  std::unique_ptr<Writer> Writer =
      createWriter(Config, Obj, Out, OutputElfType);
  if (Error E = Writer->finalize())
    return E;
  return Writer->write();
}

Error objcopy::elf::executeObjcopyOnIHex(const CommonConfig &Config,
                                         const ELFConfig &ELFConfig,
                                         MemoryBuffer &In, raw_ostream &Out) {
  IHexReader Reader(&In);
  Expected<std::unique_ptr<Object>> Obj = Reader.create(true);
  if (!Obj)
    return Obj.takeError();

  const ElfType OutputElfType =
      getOutputElfType(Config.OutputArch.getValueOr(MachineInfo()));
  if (Error E = handleArgs(Config, ELFConfig, **Obj))
    return E;
  return writeOutput(Config, **Obj, Out, OutputElfType);
}

Error objcopy::elf::executeObjcopyOnRawBinary(const CommonConfig &Config,
                                              const ELFConfig &ELFConfig,
                                              MemoryBuffer &In,
                                              raw_ostream &Out) {
  BinaryReader Reader(&In, ELFConfig.NewSymbolVisibility);
  Expected<std::unique_ptr<Object>> Obj = Reader.create(true);
  if (!Obj)
    return Obj.takeError();

  // Prefer OutputArch (-O<format>) if set, otherwise fallback to BinaryArch
  // (-B<arch>).
  const ElfType OutputElfType =
      getOutputElfType(Config.OutputArch.getValueOr(MachineInfo()));
  if (Error E = handleArgs(Config, ELFConfig, **Obj))
    return E;
  return writeOutput(Config, **Obj, Out, OutputElfType);
}

Error objcopy::elf::executeObjcopyOnBinary(const CommonConfig &Config,
                                           const ELFConfig &ELFConfig,
                                           object::ELFObjectFileBase &In,
                                           raw_ostream &Out) {
  ELFReader Reader(&In, Config.ExtractPartition);
  Expected<std::unique_ptr<Object>> Obj =
      Reader.create(!Config.SymbolsToAdd.empty());
  if (!Obj)
    return Obj.takeError();
  // Prefer OutputArch (-O<format>) if set, otherwise infer it from the input.
  const ElfType OutputElfType =
      Config.OutputArch ? getOutputElfType(Config.OutputArch.getValue())
                        : getOutputElfType(In);

  if (Error E = handleArgs(Config, ELFConfig, **Obj))
    return createFileError(Config.InputFilename, std::move(E));

  if (Error E = writeOutput(Config, **Obj, Out, OutputElfType))
    return createFileError(Config.InputFilename, std::move(E));

  return Error::success();
}
