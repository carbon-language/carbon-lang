//===-- RuntimeDyldELF.cpp - Run-time dynamic linker for MC-JIT -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of ELF support for the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dyld"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/IntervalMap.h"
#include "RuntimeDyldELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {


void RuntimeDyldELF::resolveX86_64Relocation(uint8_t *LocalAddress,
                                             uint64_t FinalAddress,
                                             uint64_t Value,
                                             uint32_t Type,
                                             int64_t Addend) {
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
  break;
  case ELF::R_X86_64_64: {
    uint64_t *Target = (uint64_t*)(LocalAddress);
    *Target = Value + Addend;
    break;
  }
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S: {
    Value += Addend;
    // FIXME: Handle the possibility of this assertion failing
    assert((Type == ELF::R_X86_64_32 && !(Value & 0xFFFFFFFF00000000ULL)) ||
           (Type == ELF::R_X86_64_32S &&
            (Value & 0xFFFFFFFF00000000ULL) == 0xFFFFFFFF00000000ULL));
    uint32_t TruncatedAddr = (Value & 0xFFFFFFFF);
    uint32_t *Target = reinterpret_cast<uint32_t*>(LocalAddress);
    *Target = TruncatedAddr;
    break;
  }
  case ELF::R_X86_64_PC32: {
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(LocalAddress);
    int64_t RealOffset = *Placeholder + Value + Addend - FinalAddress;
    assert(RealOffset <= 214783647 && RealOffset >= -214783648);
    int32_t TruncOffset = (RealOffset & 0xFFFFFFFF);
    *Placeholder = TruncOffset;
    break;
  }
  }
}

void RuntimeDyldELF::resolveX86Relocation(uint8_t *LocalAddress,
                                          uint32_t FinalAddress,
                                          uint32_t Value,
                                          uint32_t Type,
                                          int32_t Addend) {
  switch (Type) {
  case ELF::R_386_32: {
    uint32_t *Target = (uint32_t*)(LocalAddress);
    *Target = Value + Addend;
    break;
  }
  case ELF::R_386_PC32: {
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(LocalAddress);
    uint32_t RealOffset = *Placeholder + Value + Addend - FinalAddress;
    *Placeholder = RealOffset;
    break;
    }
    default:
      // There are other relocation types, but it appears these are the
      //  only ones currently used by the LLVM ELF object writer
      llvm_unreachable("Relocation type not implemented yet!");
      break;
  }
}

void RuntimeDyldELF::resolveARMRelocation(uint8_t *LocalAddress,
                                          uint32_t FinalAddress,
                                          uint32_t Value,
                                          uint32_t Type,
                                          int32_t Addend) {
  // TODO: Add Thumb relocations.
  uint32_t* TargetPtr = (uint32_t*)LocalAddress;
  Value += Addend;

  DEBUG(dbgs() << "resolveARMRelocation, LocalAddress: " << LocalAddress
               << " FinalAddress: " << format("%p",FinalAddress)
               << " Value: " << format("%x",Value)
               << " Type: " << format("%x",Type)
               << " Addend: " << format("%x",Addend)
               << "\n");

  switch(Type) {
  default:
    llvm_unreachable("Not implemented relocation type!");

  // Just write 32bit value to relocation address
  case ELF::R_ARM_ABS32 :
    *TargetPtr = Value;
    break;

  // Write first 16 bit of 32 bit value to the mov instruction.
  // Last 4 bit should be shifted.
  case ELF::R_ARM_MOVW_ABS_NC :
    Value = Value & 0xFFFF;
    *TargetPtr |= Value & 0xFFF;
    *TargetPtr |= ((Value >> 12) & 0xF) << 16;
    break;

  // Write last 16 bit of 32 bit value to the mov instruction.
  // Last 4 bit should be shifted.
  case ELF::R_ARM_MOVT_ABS :
    Value = (Value >> 16) & 0xFFFF;
    *TargetPtr |= Value & 0xFFF;
    *TargetPtr |= ((Value >> 12) & 0xF) << 16;
    break;

  // Write 24 bit relative value to the branch instruction.
  case ELF::R_ARM_PC24 :    // Fall through.
  case ELF::R_ARM_CALL :    // Fall through.
  case ELF::R_ARM_JUMP24 :
    int32_t RelValue = static_cast<int32_t>(Value - FinalAddress - 8);
    RelValue = (RelValue & 0x03FFFFFC) >> 2;
    *TargetPtr &= 0xFF000000;
    *TargetPtr |= RelValue;
    break;
  }
}

void RuntimeDyldELF::resolveRelocation(uint8_t *LocalAddress,
                                       uint64_t FinalAddress,
                                       uint64_t Value,
                                       uint32_t Type,
                                       int64_t Addend) {
  switch (Arch) {
  case Triple::x86_64:
    resolveX86_64Relocation(LocalAddress, FinalAddress, Value, Type, Addend);
    break;
  case Triple::x86:
    resolveX86Relocation(LocalAddress, (uint32_t)(FinalAddress & 0xffffffffL),
                         (uint32_t)(Value & 0xffffffffL), Type,
                         (uint32_t)(Addend & 0xffffffffL));
    break;
  case Triple::arm:    // Fall through.
  case Triple::thumb:
    resolveARMRelocation(LocalAddress, (uint32_t)(FinalAddress & 0xffffffffL),
                         (uint32_t)(Value & 0xffffffffL), Type,
                         (uint32_t)(Addend & 0xffffffffL));
    break;
  default: llvm_unreachable("Unsupported CPU type!");
  }
}

void RuntimeDyldELF::
processRelocationRef(const ObjRelocationInfo &Rel, const ObjectFile &Obj,
                     ObjSectionToIDMap &ObjSectionToID,
                     LocalSymbolMap &Symbols, StubMap &Stubs) {

  uint32_t RelType = (uint32_t)(Rel.Type & 0xffffffffL);
  intptr_t Addend = (intptr_t)Rel.AdditionalInfo;
  RelocationValueRef Value;
  StringRef TargetName;
  const SymbolRef &Symbol = Rel.Symbol;
  Symbol.getName(TargetName);
  DEBUG(dbgs() << "\t\tRelType: " << RelType
               << " Addend: " << Addend
               << " TargetName: " << TargetName
               << "\n");
  // First look the symbol in object file symbols.
  LocalSymbolMap::iterator it = Symbols.find(TargetName.data());
  if (it != Symbols.end()) {
    Value.SectionID = it->second.first;
    Value.Addend = it->second.second;
  } else {
    // Second look the symbol in global symbol table.
    StringMap<SymbolLoc>::iterator itS = SymbolTable.find(TargetName.data());
    if (itS != SymbolTable.end()) {
      Value.SectionID = itS->second.first;
      Value.Addend = itS->second.second;
    } else {
      SymbolRef::Type SymType;
      Symbol.getType(SymType);
      switch (SymType) {
        case SymbolRef::ST_Debug: {
          // TODO: Now ELF SymbolRef::ST_Debug = STT_SECTION, it's not obviously
          // and can be changed by another developers. Maybe best way is add
          // a new symbol type ST_Section to SymbolRef and use it.
          section_iterator sIt = Obj.end_sections();
          Symbol.getSection(sIt);
          if (sIt == Obj.end_sections())
            llvm_unreachable("Symbol section not found, bad object file format!");
          DEBUG(dbgs() << "\t\tThis is section symbol\n");
          Value.SectionID = findOrEmitSection((*sIt), true, ObjSectionToID);
          Value.Addend = Addend;
          break;
        }
        case SymbolRef::ST_Unknown: {
          Value.SymbolName = TargetName.data();
          Value.Addend = Addend;
          break;
        }
        default:
          llvm_unreachable("Unresolved symbol type!");
          break;
      }
    }
  }
  DEBUG(dbgs() << "\t\tRel.SectionID: " << Rel.SectionID
               << " Rel.Offset: " << Rel.Offset
               << "\n");
  if (Arch == Triple::arm &&
      (RelType == ELF::R_ARM_PC24 ||
       RelType == ELF::R_ARM_CALL ||
       RelType == ELF::R_ARM_JUMP24)) {
    // This is an ARM branch relocation, need to use a stub function.
    DEBUG(dbgs() << "\t\tThis is an ARM branch relocation.");
    SectionEntry &Section = Sections[Rel.SectionID];
    uint8_t *Target = Section.Address + Rel.Offset;

    //  Look up for existing stub.
    StubMap::const_iterator stubIt = Stubs.find(Value);
    if (stubIt != Stubs.end()) {
      resolveRelocation(Target, Section.LoadAddress, (uint64_t)Section.Address +
                        stubIt->second, RelType, 0);
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                   Section.StubOffset);
      AddRelocation(Value, Rel.SectionID,
                    StubTargetAddr - Section.Address, ELF::R_ARM_ABS32);
      resolveRelocation(Target, Section.LoadAddress, (uint64_t)Section.Address +
                        Section.StubOffset, RelType, 0);
      Section.StubOffset += getMaxStubSize();
    }
  } else
    AddRelocation(Value, Rel.SectionID, Rel.Offset, RelType);
}

bool RuntimeDyldELF::isCompatibleFormat(const MemoryBuffer *InputBuffer) const {
  StringRef Magic = InputBuffer->getBuffer().slice(0, ELF::EI_NIDENT);
  return (memcmp(Magic.data(), ELF::ElfMagic, strlen(ELF::ElfMagic))) == 0;
}
} // namespace llvm
