//===-- InstrProfCorrelator.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfCorrelator.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFLocationExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "correlator"

using namespace llvm;

/// Get the __llvm_prf_cnts section.
Expected<object::SectionRef> getCountersSection(const object::ObjectFile &Obj) {
  for (auto &Section : Obj.sections())
    if (auto SectionName = Section.getName())
      if (SectionName.get() == INSTR_PROF_CNTS_SECT_NAME)
        return Section;
  return make_error<InstrProfError>(
      instrprof_error::unable_to_correlate_profile,
      "could not find counter section (" INSTR_PROF_CNTS_SECT_NAME ")");
}

const char *InstrProfCorrelator::FunctionNameAttributeName = "Function Name";
const char *InstrProfCorrelator::CFGHashAttributeName = "CFG Hash";
const char *InstrProfCorrelator::NumCountersAttributeName = "Num Counters";

llvm::Expected<std::unique_ptr<InstrProfCorrelator::Context>>
InstrProfCorrelator::Context::get(std::unique_ptr<MemoryBuffer> Buffer,
                                  const object::ObjectFile &Obj) {
  auto CountersSection = getCountersSection(Obj);
  if (auto Err = CountersSection.takeError())
    return std::move(Err);
  auto C = std::make_unique<Context>();
  C->Buffer = std::move(Buffer);
  C->CountersSectionStart = CountersSection->getAddress();
  C->CountersSectionEnd = C->CountersSectionStart + CountersSection->getSize();
  C->ShouldSwapBytes = Obj.isLittleEndian() != sys::IsLittleEndianHost;
  return Expected<std::unique_ptr<Context>>(std::move(C));
}

llvm::Expected<std::unique_ptr<InstrProfCorrelator>>
InstrProfCorrelator::get(StringRef DebugInfoFilename) {
  auto DsymObjectsOrErr =
      object::MachOObjectFile::findDsymObjectMembers(DebugInfoFilename);
  if (auto Err = DsymObjectsOrErr.takeError())
    return std::move(Err);
  if (!DsymObjectsOrErr->empty()) {
    // TODO: Enable profile correlation when there are multiple objects in a
    // dSYM bundle.
    if (DsymObjectsOrErr->size() > 1)
      return make_error<InstrProfError>(
          instrprof_error::unable_to_correlate_profile,
          "using multiple objects is not yet supported");
    DebugInfoFilename = *DsymObjectsOrErr->begin();
  }
  auto BufferOrErr =
      errorOrToExpected(MemoryBuffer::getFile(DebugInfoFilename));
  if (auto Err = BufferOrErr.takeError())
    return std::move(Err);

  return get(std::move(*BufferOrErr));
}

llvm::Expected<std::unique_ptr<InstrProfCorrelator>>
InstrProfCorrelator::get(std::unique_ptr<MemoryBuffer> Buffer) {
  auto BinOrErr = object::createBinary(*Buffer);
  if (auto Err = BinOrErr.takeError())
    return std::move(Err);

  if (auto *Obj = dyn_cast<object::ObjectFile>(BinOrErr->get())) {
    auto CtxOrErr = Context::get(std::move(Buffer), *Obj);
    if (auto Err = CtxOrErr.takeError())
      return std::move(Err);
    auto T = Obj->makeTriple();
    if (T.isArch64Bit())
      return InstrProfCorrelatorImpl<uint64_t>::get(std::move(*CtxOrErr), *Obj);
    if (T.isArch32Bit())
      return InstrProfCorrelatorImpl<uint32_t>::get(std::move(*CtxOrErr), *Obj);
  }
  return make_error<InstrProfError>(
      instrprof_error::unable_to_correlate_profile, "not an object file");
}

Optional<size_t> InstrProfCorrelator::getDataSize() const {
  if (auto *C = dyn_cast<InstrProfCorrelatorImpl<uint32_t>>(this)) {
    return C->getDataSize();
  } else if (auto *C = dyn_cast<InstrProfCorrelatorImpl<uint64_t>>(this)) {
    return C->getDataSize();
  }
  return {};
}

namespace llvm {

template <>
InstrProfCorrelatorImpl<uint32_t>::InstrProfCorrelatorImpl(
    std::unique_ptr<InstrProfCorrelator::Context> Ctx)
    : InstrProfCorrelatorImpl(InstrProfCorrelatorKind::CK_32Bit,
                              std::move(Ctx)) {}
template <>
InstrProfCorrelatorImpl<uint64_t>::InstrProfCorrelatorImpl(
    std::unique_ptr<InstrProfCorrelator::Context> Ctx)
    : InstrProfCorrelatorImpl(InstrProfCorrelatorKind::CK_64Bit,
                              std::move(Ctx)) {}
template <>
bool InstrProfCorrelatorImpl<uint32_t>::classof(const InstrProfCorrelator *C) {
  return C->getKind() == InstrProfCorrelatorKind::CK_32Bit;
}
template <>
bool InstrProfCorrelatorImpl<uint64_t>::classof(const InstrProfCorrelator *C) {
  return C->getKind() == InstrProfCorrelatorKind::CK_64Bit;
}

} // end namespace llvm

template <class IntPtrT>
llvm::Expected<std::unique_ptr<InstrProfCorrelatorImpl<IntPtrT>>>
InstrProfCorrelatorImpl<IntPtrT>::get(
    std::unique_ptr<InstrProfCorrelator::Context> Ctx,
    const object::ObjectFile &Obj) {
  if (Obj.isELF() || Obj.isMachO()) {
    auto DICtx = DWARFContext::create(Obj);
    return std::make_unique<DwarfInstrProfCorrelator<IntPtrT>>(std::move(DICtx),
                                                               std::move(Ctx));
  }
  return make_error<InstrProfError>(
      instrprof_error::unable_to_correlate_profile,
      "unsupported debug info format (only DWARF is supported)");
}

template <class IntPtrT>
Error InstrProfCorrelatorImpl<IntPtrT>::correlateProfileData() {
  assert(Data.empty() && Names.empty() && NamesVec.empty());
  correlateProfileDataImpl();
  if (Data.empty() || NamesVec.empty())
    return make_error<InstrProfError>(
        instrprof_error::unable_to_correlate_profile,
        "could not find any profile metadata in debug info");
  auto Result =
      collectPGOFuncNameStrings(NamesVec, /*doCompression=*/false, Names);
  CounterOffsets.clear();
  NamesVec.clear();
  return Result;
}

template <class IntPtrT>
void InstrProfCorrelatorImpl<IntPtrT>::addProbe(StringRef FunctionName,
                                                uint64_t CFGHash,
                                                IntPtrT CounterOffset,
                                                IntPtrT FunctionPtr,
                                                uint32_t NumCounters) {
  // Check if a probe was already added for this counter offset.
  if (!CounterOffsets.insert(CounterOffset).second)
    return;
  Data.push_back({
      maybeSwap<uint64_t>(IndexedInstrProf::ComputeHash(FunctionName)),
      maybeSwap<uint64_t>(CFGHash),
      // In this mode, CounterPtr actually stores the section relative address
      // of the counter.
      maybeSwap<IntPtrT>(CounterOffset),
      maybeSwap<IntPtrT>(FunctionPtr),
      // TODO: Value profiling is not yet supported.
      /*ValuesPtr=*/maybeSwap<IntPtrT>(0),
      maybeSwap<uint32_t>(NumCounters),
      /*NumValueSites=*/{maybeSwap<uint16_t>(0), maybeSwap<uint16_t>(0)},
  });
  NamesVec.push_back(FunctionName.str());
}

template <class IntPtrT>
llvm::Optional<uint64_t>
DwarfInstrProfCorrelator<IntPtrT>::getLocation(const DWARFDie &Die) const {
  auto Locations = Die.getLocations(dwarf::DW_AT_location);
  if (!Locations) {
    consumeError(Locations.takeError());
    return {};
  }
  auto &DU = *Die.getDwarfUnit();
  auto AddressSize = DU.getAddressByteSize();
  for (auto &Location : *Locations) {
    DataExtractor Data(Location.Expr, DICtx->isLittleEndian(), AddressSize);
    DWARFExpression Expr(Data, AddressSize);
    for (auto &Op : Expr) {
      if (Op.getCode() == dwarf::DW_OP_addr) {
        return Op.getRawOperand(0);
      } else if (Op.getCode() == dwarf::DW_OP_addrx) {
        uint64_t Index = Op.getRawOperand(0);
        if (auto SA = DU.getAddrOffsetSectionItem(Index))
          return SA->Address;
      }
    }
  }
  return {};
}

template <class IntPtrT>
bool DwarfInstrProfCorrelator<IntPtrT>::isDIEOfProbe(const DWARFDie &Die) {
  const auto &ParentDie = Die.getParent();
  if (!Die.isValid() || !ParentDie.isValid() || Die.isNULL())
    return false;
  if (Die.getTag() != dwarf::DW_TAG_variable)
    return false;
  if (!ParentDie.isSubprogramDIE())
    return false;
  if (!Die.hasChildren())
    return false;
  if (const char *Name = Die.getName(DINameKind::ShortName))
    return StringRef(Name).startswith(getInstrProfCountersVarPrefix());
  return false;
}

template <class IntPtrT>
void DwarfInstrProfCorrelator<IntPtrT>::correlateProfileDataImpl() {
  auto maybeAddProbe = [&](DWARFDie Die) {
    if (!isDIEOfProbe(Die))
      return;
    Optional<const char *> FunctionName;
    Optional<uint64_t> CFGHash;
    Optional<uint64_t> CounterPtr = getLocation(Die);
    auto FunctionPtr =
        dwarf::toAddress(Die.getParent().find(dwarf::DW_AT_low_pc));
    Optional<uint64_t> NumCounters;
    for (const DWARFDie &Child : Die.children()) {
      if (Child.getTag() != dwarf::DW_TAG_LLVM_annotation)
        continue;
      auto AnnotationFormName = Child.find(dwarf::DW_AT_name);
      auto AnnotationFormValue = Child.find(dwarf::DW_AT_const_value);
      if (!AnnotationFormName || !AnnotationFormValue)
        continue;
      auto AnnotationNameOrErr = AnnotationFormName->getAsCString();
      if (auto Err = AnnotationNameOrErr.takeError()) {
        consumeError(std::move(Err));
        continue;
      }
      StringRef AnnotationName = *AnnotationNameOrErr;
      if (AnnotationName.compare(
              InstrProfCorrelator::FunctionNameAttributeName) == 0) {
        if (auto EC =
                AnnotationFormValue->getAsCString().moveInto(FunctionName))
          consumeError(std::move(EC));
      } else if (AnnotationName.compare(
                     InstrProfCorrelator::CFGHashAttributeName) == 0) {
        CFGHash = AnnotationFormValue->getAsUnsignedConstant();
      } else if (AnnotationName.compare(
                     InstrProfCorrelator::NumCountersAttributeName) == 0) {
        NumCounters = AnnotationFormValue->getAsUnsignedConstant();
      }
    }
    if (!FunctionName || !CFGHash || !CounterPtr || !NumCounters) {
      LLVM_DEBUG(dbgs() << "Incomplete DIE for probe\n\tFunctionName: "
                        << FunctionName << "\n\tCFGHash: " << CFGHash
                        << "\n\tCounterPtr: " << CounterPtr
                        << "\n\tNumCounters: " << NumCounters);
      LLVM_DEBUG(Die.dump(dbgs()));
      return;
    }
    uint64_t CountersStart = this->Ctx->CountersSectionStart;
    uint64_t CountersEnd = this->Ctx->CountersSectionEnd;
    if (*CounterPtr < CountersStart || *CounterPtr >= CountersEnd) {
      LLVM_DEBUG(
          dbgs() << "CounterPtr out of range for probe\n\tFunction Name: "
                 << FunctionName << "\n\tExpected: [0x"
                 << Twine::utohexstr(CountersStart) << ", 0x"
                 << Twine::utohexstr(CountersEnd) << ")\n\tActual: 0x"
                 << Twine::utohexstr(*CounterPtr));
      LLVM_DEBUG(Die.dump(dbgs()));
      return;
    }
    if (!FunctionPtr) {
      LLVM_DEBUG(dbgs() << "Could not find address of " << *FunctionName
                        << "\n");
      LLVM_DEBUG(Die.dump(dbgs()));
    }
    this->addProbe(*FunctionName, *CFGHash, *CounterPtr - CountersStart,
                   FunctionPtr.getValueOr(0), *NumCounters);
  };
  for (auto &CU : DICtx->normal_units())
    for (const auto &Entry : CU->dies())
      maybeAddProbe(DWARFDie(CU.get(), &Entry));
  for (auto &CU : DICtx->dwo_units())
    for (const auto &Entry : CU->dies())
      maybeAddProbe(DWARFDie(CU.get(), &Entry));
}
