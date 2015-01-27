//===- DIASymbol.cpp - Dump debug info from a PDB file ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-pdbdump.h"
#include "DIASymbol.h"

using namespace llvm::sys::windows;

DIASymbol::DIASymbol(IDiaSymbol *DiaSymbol) : Symbol(DiaSymbol) {
  Symbol->AddRef();
}

DIASymbol::~DIASymbol() { Symbol->Release(); }

void DIASymbol::fullDump(int IndentLevel) {
  getLexicalParent().dump("Lexical parent", IndentLevel);
  getClassParent().dump("Class parent", IndentLevel);
  getType().dump("Type", IndentLevel);
  getArrayIndexType().dump("Array index type", IndentLevel);
  getVirtualTableShape().dump("Vtable shape", IndentLevel);
  getLowerBound().dump("Lower bound", IndentLevel);
  getUpperBound().dump("Upper bound", IndentLevel);
  getObjectPointerType().dump("Object pointer type", IndentLevel);
  getContainer().dump("Container", IndentLevel);
  getVirtualBaseTableType().dump("Virtual base table type", IndentLevel);
  getUnmodifiedType().dump("Unmodified type", IndentLevel);
  getSubType().dump("Sub type", IndentLevel);
  getBaseSymbol().dump("Base symbol", IndentLevel);

#if (_MSC_FULL_VER >= 180031101)
  // These methods are only available on VS 2013 SP4 and higher.
  isPGO().dump("Is PGO", IndentLevel);
  hasValidPGOCounts().dump("Has valid PGO counts", IndentLevel);
  isOptimizedForSpeed().dump("Is optimized for speed", IndentLevel);
  getPGOEntryCount().dump("PGO entry count", IndentLevel);
  getPGOEdgeCount().dump("PGO edge count", IndentLevel);
  getPGODynamicInstructionCount().dump("PGO dynamic instruction count",
                                       IndentLevel);
  getStaticSize().dump("Static size", IndentLevel);
  getFinalLiveStaticSize().dump("Final live static size", IndentLevel);
  getPhaseName().dump("Phase name", IndentLevel);
  hasControlFlowCheck().dump("Has control flow check", IndentLevel);
#endif

  getAccess().dump("Access", IndentLevel);
  getAddressOffset().dump("Address Offset", IndentLevel);
  getAddressSection().dump("Address Section", IndentLevel);
  getAddressTaken().dump("Address Taken", IndentLevel);
  getAge().dump("Age", IndentLevel);
  getArrayIndexTypeId().dump("Array Index Type Id", IndentLevel);
  getBackEndMajor().dump("Back End Major", IndentLevel);
  getBackEndMinor().dump("Back End Minor", IndentLevel);
  getBackEndBuild().dump("Back End Build", IndentLevel);
  getBaseDataOffset().dump("Base Data Offset", IndentLevel);
  getBaseDataSlot().dump("Base Data Slot", IndentLevel);
  getBaseSymbolId().dump("Base Symbol Id", IndentLevel);
  getBaseType().dump("Base Type", IndentLevel);
  getBitPosition().dump("Bit Position", IndentLevel);
  getBuiltInKind().dump("Built In Kind", IndentLevel);
  getCallingConvention().dump("Calling Convention", IndentLevel);
  getClassParentId().dump("Class Parent Id", IndentLevel);
  isCode().dump("Code", IndentLevel);
  isCompilerGenerated().dump("Compiler Generated", IndentLevel);
  getCompilerName().dump("Compiler Name", IndentLevel);
  hasConstructor().dump("Constructor", IndentLevel);
  isConstType().dump("Const Type", IndentLevel);
  getCount().dump("Count", IndentLevel);
  getCountLiveRanges().dump("Count Live Ranges", IndentLevel);
  hasCustomCallingConvention().dump("Custom Calling Convention", IndentLevel);
  getDataKind().dump("Data Kind", IndentLevel);
  isEditAndContinueEnabled().dump("Edit and Continue Enabled", IndentLevel);
  hasFarReturn().dump("Far Return", IndentLevel);
  getFrontEndMajor().dump("Front End Major", IndentLevel);
  getFrontEndMinor().dump("Front End Minor", IndentLevel);
  getFrontEndBuild().dump("Front End Build", IndentLevel);
  isFunction().dump("Function", IndentLevel);
  getGuid().dump("GUID", IndentLevel);
  hasAlloca().dump("Has Alloca", IndentLevel);
  hasAssignmentOperator().dump("Has Assignment Operator", IndentLevel);
  hasCastOperator().dump("Has Cast Operator", IndentLevel);
  hasDebugInfo().dump("Has Debug Info", IndentLevel);
  hasEH().dump("Has EH", IndentLevel);
  hasEHa().dump("Has EHa", IndentLevel);
  hasInlAsm().dump("Has Inline Asm", IndentLevel);
  hasLongJump().dump("Has longjmp", IndentLevel);
  hasManagedCode().dump("Has Managed Code", IndentLevel);
  hasNestedTypes().dump("Has Nested Types", IndentLevel);
  hasSecurityChecks().dump("Has Security Checks", IndentLevel);
  hasSEH().dump("Has SEH", IndentLevel);
  hasSetJump().dump("Has setjmp", IndentLevel);
  isIndirectVirtualBaseClass().dump("Is indirect virtual base", IndentLevel);
  hasInlineAttribute().dump("Has inline declspec", IndentLevel);
  hasInterruptReturn().dump("Has interrupt return", IndentLevel);
  isBaseVirtualFunction().dump("Is base virtual function", IndentLevel);
  isAcceleratorGroupSharedLocal().dump("Is Accelerator Group Shared Local",
                                       IndentLevel);
  isAcceleratorPointerTagLiveRange().dump(
      "Is Accelerator Pointer Tag Live Range", IndentLevel);
  isAcceleratorStubFunction().dump("Is Accelerator Stub Function", IndentLevel);
  isAggregated().dump("Is aggregated", IndentLevel);
  hasCTypes().dump("Has C types", IndentLevel);
  isCVTCIL().dump("Was converted from MSIL", IndentLevel);
  isDataAligned().dump("Is data aligned", IndentLevel);
  isHLSLData().dump("Is HLSL data", IndentLevel);
  isHotpatchable().dump("Is hot-patchable", IndentLevel);
  isLTCG().dump("Is LTCG", IndentLevel);
  isMatrixRowMajor().dump("Is matrix row major", IndentLevel);
  isMSILNetmodule().dump("Is MSIL .netmodule", IndentLevel);
  isMultipleInheritance().dump("Is multiple inheritance", IndentLevel);
  isNaked().dump("Is naked", IndentLevel);
  isOptimizedAway().dump("Is optimized away", IndentLevel);
  isPointerBasedOnSymbolValue().dump("Is pointer based on symbol value",
                                     IndentLevel);
  isPointerToDataMember().dump("Is pointer to data member", IndentLevel);
  isPointerToMemberFunction().dump("Is pointer to member function",
                                   IndentLevel);
  isReturnValue().dump("Is return value", IndentLevel);
  isSdl().dump("Is SDL", IndentLevel);
  isSingleInheritance().dump("Is single inheritance", IndentLevel);
  isSplitted().dump("Is splitted", IndentLevel);
  isStatic().dump("Is staic", IndentLevel);
  isStripped().dump("Is stripped", IndentLevel);
  isVirtualInheritance().dump("Is virtual inheritance", IndentLevel);
  getLanguage().dump("Language", IndentLevel);
  getLength().dump("Length", IndentLevel);
  getLexicalParentId().dump("Lexical parent id", IndentLevel);
  getLibraryName().dump("Library name", IndentLevel);
  getLiveRangeLength().dump("Live range length", IndentLevel);
  getLiveRangeStartAddressSection().dump("Live range start address section",
                                         IndentLevel);
  getLiveRangeStartAddressOffset().dump("Live range start address offset",
                                        IndentLevel);
  getLiveRangeStartRelativeVirtualAddress().dump("Live range start address RVA",
                                                 IndentLevel);
  getLocationType().dump("Location type", IndentLevel);
  getLowerBoundId().dump("Lower bound id", IndentLevel);
  getMachineType().dump("Machine type", IndentLevel);
  isManagedRef().dump("Managed", IndentLevel);
  getMemorySpaceKind().dump("Memory space kind", IndentLevel);
  isMsilRef().dump("MSIL", IndentLevel);
  getName().dump("Name", IndentLevel);
  isNested().dump("Nested", IndentLevel);
  hasNoInlineAttribute().dump("Has noinline declspec", IndentLevel);
  hasNoReturnAttribute().dump("Has noreturn declspec", IndentLevel);
  getNoStackOrdering().dump("No stack ordering", IndentLevel);
  isUnreached().dump("Not reached", IndentLevel);
  getNumberOfAcceleratorPointerTags().dump("Number of accelerator pointer tags",
                                           IndentLevel);
  getNumberOfModifiers().dump("Number of modifiers", IndentLevel);
  getNumberOfRegisterIndices().dump("Number of register indices", IndentLevel);
  getNumberOfRows().dump("Number of rows", IndentLevel);
  getNumberOfColumns().dump("Number of columns", IndentLevel);
  getObjectFileName().dump("Object file name", IndentLevel);
  getOemSymbolId().dump("OEM symbol id", IndentLevel);
  getOffset().dump("Offset", IndentLevel);
  hasOptimizedCodeDebugInfo().dump("Optimized code debug info", IndentLevel);
  hasOverloadedOperator().dump("Overloaded operator", IndentLevel);
  isPacked().dump("Packed", IndentLevel);
  getPlatform().dump("Platform", IndentLevel);
  isPureVirtual().dump("Pure", IndentLevel);
  getRank().dump("Rank", IndentLevel);
  isReference().dump("Reference", IndentLevel);
  getRegisterId().dump("Register ID", IndentLevel);
  getRegisterType().dump("Register type", IndentLevel);
  getRelativeVirtualAddress().dump("RVA", IndentLevel);
  isRestrictedType().dump("Restricted type", IndentLevel);
  getSamplerSlot().dump("Sampler slot", IndentLevel);
  isScoped().dump("Scoped", IndentLevel);
  getSignature().dump("Signature", IndentLevel);
  getSizeInUdt().dump("Size in UDT", IndentLevel);
  getSlot().dump("Slot", IndentLevel);
  getSourceFileName().dump("Source file name", IndentLevel);
  getStride().dump("Stride", IndentLevel);
  getSubTypeId().dump("Sub type ID", IndentLevel);
  getSymbolsFileName().dump("File name", IndentLevel);
  getSymIndexId().dump("Sym index ID", IndentLevel);
  getSymTag().dump("Sym tag", IndentLevel);
  getTargetOffset().dump("Target offset", IndentLevel);
  getTargetRelativeVirtualAddress().dump("Target RVA", IndentLevel);
  getTargetSection().dump("Target section", IndentLevel);
  getTargetVirtualAddress().dump("Target virtual address", IndentLevel);
  getTextureSlot().dump("Texture slot", IndentLevel);
  getThisAdjust().dump("This adjust", IndentLevel);
  getThunkOrdinal().dump("Thunk ordinal", IndentLevel);
  getTimeStamp().dump("Time stamp", IndentLevel);
  getToken().dump("Token", IndentLevel);
  getUavSlot().dump("UAV slot", IndentLevel);
  getUdtKind().dump("UDT kind", IndentLevel);
  isUnalignedType().dump("Unaligned type", IndentLevel);
  getUndecoratedName().dump("Undecorated name", IndentLevel);
  getUnmodifiedTypeId().dump("Unmodified type id", IndentLevel);
  getUpperBoundId().dump("Upper bound id", IndentLevel);
  isVirtual().dump("Virtual", IndentLevel);
  getVirtualAddress().dump("Virtual address", IndentLevel);
  isVirtualBaseClass().dump("Virtual base class", IndentLevel);
  getVirtualBaseDispIndex().dump("Virtual base disp index", IndentLevel);
  getVirtualBaseOffset().dump("Virtual base offset", IndentLevel);
  getVirtualBasePointerOffset().dump("Virtual base pointer offset",
                                     IndentLevel);
  getVirtualTableShapeId().dump("Vtable shape ID", IndentLevel);
  isVolatileType().dump("Volatile type", IndentLevel);
}

#if (_MSC_FULL_VER >= 180031101)
DIAResult<BOOL> DIASymbol::isPGO() {
  return InternalGetDIAValue(&IDiaSymbol::get_isPGO);
}

DIAResult<BOOL> DIASymbol::hasValidPGOCounts() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasValidPGOCounts);
}

DIAResult<BOOL> DIASymbol::isOptimizedForSpeed() {
  return InternalGetDIAValue(&IDiaSymbol::get_isOptimizedForSpeed);
}

DIAResult<DWORD> DIASymbol::getPGOEntryCount() {
  return InternalGetDIAValue(&IDiaSymbol::get_PGOEntryCount);
}

DIAResult<DWORD> DIASymbol::getPGOEdgeCount() {
  return InternalGetDIAValue(&IDiaSymbol::get_PGOEdgeCount);
}

DIAResult<ULONGLONG> DIASymbol::getPGODynamicInstructionCount() {
  return InternalGetDIAValue(&IDiaSymbol::get_PGODynamicInstructionCount);
}

DIAResult<DWORD> DIASymbol::getStaticSize() {
  return InternalGetDIAValue(&IDiaSymbol::get_staticSize);
}

DIAResult<DWORD> DIASymbol::getFinalLiveStaticSize() {
  return InternalGetDIAValue(&IDiaSymbol::get_finalLiveStaticSize);
}

DIAResult<DIAString> DIASymbol::getPhaseName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_phaseName);
}

DIAResult<BOOL> DIASymbol::hasControlFlowCheck() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasControlFlowCheck);
}
#endif

DIAResult<DiaSymbolPtr> DIASymbol::getClassParent() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_classParent);
}

DIAResult<DiaSymbolPtr> DIASymbol::getType() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(&IDiaSymbol::get_type);
}

DIAResult<DiaSymbolPtr> DIASymbol::getArrayIndexType() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_arrayIndexType);
}

DIAResult<DiaSymbolPtr> DIASymbol::getVirtualTableShape() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_virtualTableShape);
}

DIAResult<DiaSymbolPtr> DIASymbol::getLowerBound() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_lowerBound);
}

DIAResult<DiaSymbolPtr> DIASymbol::getUpperBound() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_upperBound);
}

DIAResult<DiaSymbolPtr> DIASymbol::getObjectPointerType() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_objectPointerType);
}

DIAResult<DiaSymbolPtr> DIASymbol::getContainer() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_container);
}

DIAResult<DiaSymbolPtr> DIASymbol::getVirtualBaseTableType() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_virtualBaseTableType);
}

DIAResult<DiaSymbolPtr> DIASymbol::getUnmodifiedType() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_unmodifiedType);
}

DIAResult<DiaSymbolPtr> DIASymbol::getSubType() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_subType);
}

DIAResult<DiaSymbolPtr> DIASymbol::getBaseSymbol() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_baseSymbol);
}

DIAResult<DWORD> DIASymbol::getAccess() {
  return InternalGetDIAValue(&IDiaSymbol::get_access);
}

DIAResult<DWORD> DIASymbol::getAddressOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_addressOffset);
}

DIAResult<DWORD> DIASymbol::getAddressSection() {
  return InternalGetDIAValue(&IDiaSymbol::get_addressSection);
}

DIAResult<BOOL> DIASymbol::getAddressTaken() {
  return InternalGetDIAValue(&IDiaSymbol::get_addressTaken);
}

DIAResult<DWORD> DIASymbol::getAge() {
  return InternalGetDIAValue(&IDiaSymbol::get_age);
}

DIAResult<DWORD> DIASymbol::getArrayIndexTypeId() {
  return InternalGetDIAValue(&IDiaSymbol::get_arrayIndexTypeId);
}

DIAResult<DWORD> DIASymbol::getBackEndMajor() {
  return InternalGetDIAValue(&IDiaSymbol::get_backEndMajor);
}

DIAResult<DWORD> DIASymbol::getBackEndMinor() {

  return InternalGetDIAValue(&IDiaSymbol::get_backEndMinor);
}

DIAResult<DWORD> DIASymbol::getBackEndBuild() {
  return InternalGetDIAValue(&IDiaSymbol::get_backEndBuild);
}

DIAResult<DWORD> DIASymbol::getBackEndQFE() {
  return InternalGetDIAValue(&IDiaSymbol::get_backEndQFE);
}

DIAResult<DWORD> DIASymbol::getBaseDataOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_baseDataOffset);
}

DIAResult<DWORD> DIASymbol::getBaseDataSlot() {
  return InternalGetDIAValue(&IDiaSymbol::get_baseDataSlot);
}

DIAResult<DWORD> DIASymbol::getBaseSymbolId() {
  return InternalGetDIAValue(&IDiaSymbol::get_baseSymbolId);
}

DIAResult<DWORD> DIASymbol::getBaseType() {
  return InternalGetDIAValue(&IDiaSymbol::get_baseType);
}

DIAResult<DWORD> DIASymbol::getBitPosition() {
  return InternalGetDIAValue(&IDiaSymbol::get_bitPosition);
}

DIAResult<DWORD> DIASymbol::getBuiltInKind() {
  return InternalGetDIAValue(&IDiaSymbol::get_builtInKind);
}

DIAResult<CV_call_e> DIASymbol::getCallingConvention() {
  return InternalGetDIAValue<DWORD, CV_call_e>(
      &IDiaSymbol::get_callingConvention);
}

DIAResult<DWORD> DIASymbol::getClassParentId() {
  return InternalGetDIAValue(&IDiaSymbol::get_classParentId);
}

DIAResult<BOOL> DIASymbol::isCode() {
  return InternalGetDIAValue(&IDiaSymbol::get_code);
}

DIAResult<BOOL> DIASymbol::isCompilerGenerated() {
  return InternalGetDIAValue(&IDiaSymbol::get_compilerGenerated);
}

DIAResult<DIAString> DIASymbol::getCompilerName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_compilerName);
}

DIAResult<BOOL> DIASymbol::hasConstructor() {
  return InternalGetDIAValue(&IDiaSymbol::get_constructor);
}

DIAResult<BOOL> DIASymbol::isConstType() {
  return InternalGetDIAValue(&IDiaSymbol::get_constType);
}

DIAResult<DWORD> DIASymbol::getCount() {
  return InternalGetDIAValue(&IDiaSymbol::get_count);
}

DIAResult<DWORD> DIASymbol::getCountLiveRanges() {
  return InternalGetDIAValue(&IDiaSymbol::get_countLiveRanges);
}

DIAResult<BOOL> DIASymbol::hasCustomCallingConvention() {
  return InternalGetDIAValue(&IDiaSymbol::get_customCallingConvention);
}

DIAResult<DataKind> DIASymbol::getDataKind() {
  return InternalGetDIAValue<DWORD, DataKind>(&IDiaSymbol::get_dataKind);
}

DIAResult<BOOL> DIASymbol::isEditAndContinueEnabled() {
  return InternalGetDIAValue(&IDiaSymbol::get_editAndContinueEnabled);
}

DIAResult<BOOL> DIASymbol::hasFarReturn() {
  return InternalGetDIAValue(&IDiaSymbol::get_farReturn);
}

DIAResult<DWORD> DIASymbol::getFrontEndMajor() {
  return InternalGetDIAValue(&IDiaSymbol::get_frontEndMajor);
}

DIAResult<DWORD> DIASymbol::getFrontEndMinor() {
  return InternalGetDIAValue(&IDiaSymbol::get_frontEndMinor);
}

DIAResult<DWORD> DIASymbol::getFrontEndBuild() {
  return InternalGetDIAValue(&IDiaSymbol::get_frontEndBuild);
}

DIAResult<DWORD> DIASymbol::getFrontEndQFE() {
  return InternalGetDIAValue(&IDiaSymbol::get_frontEndQFE);
}

DIAResult<BOOL> DIASymbol::isFunction() {
  return InternalGetDIAValue(&IDiaSymbol::get_function);
}

DIAResult<GUID> DIASymbol::getGuid() {
  return InternalGetDIAValue(&IDiaSymbol::get_guid);
}

DIAResult<BOOL> DIASymbol::hasAlloca() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasAlloca);
}

DIAResult<BOOL> DIASymbol::hasAssignmentOperator() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasAssignmentOperator);
}

DIAResult<BOOL> DIASymbol::hasCastOperator() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasCastOperator);
}

DIAResult<BOOL> DIASymbol::hasDebugInfo() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasDebugInfo);
}

DIAResult<BOOL> DIASymbol::hasEH() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasEH);
}

DIAResult<BOOL> DIASymbol::hasEHa() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasEHa);
}

DIAResult<BOOL> DIASymbol::hasInlAsm() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasInlAsm);
}

DIAResult<BOOL> DIASymbol::hasLongJump() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasLongJump);
}

DIAResult<BOOL> DIASymbol::hasManagedCode() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasManagedCode);
}

DIAResult<BOOL> DIASymbol::hasNestedTypes() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasNestedTypes);
}

DIAResult<BOOL> DIASymbol::hasSecurityChecks() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasSecurityChecks);
}

DIAResult<BOOL> DIASymbol::hasSEH() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasSEH);
}

DIAResult<BOOL> DIASymbol::hasSetJump() {
  return InternalGetDIAValue(&IDiaSymbol::get_hasSetJump);
}

DIAResult<BOOL> DIASymbol::isIndirectVirtualBaseClass() {
  return InternalGetDIAValue(&IDiaSymbol::get_indirectVirtualBaseClass);
}

DIAResult<BOOL> DIASymbol::hasInlineAttribute() {
  return InternalGetDIAValue(&IDiaSymbol::get_inlSpec);
}

DIAResult<BOOL> DIASymbol::hasInterruptReturn() {
  return InternalGetDIAValue(&IDiaSymbol::get_interruptReturn);
}

DIAResult<BOOL> DIASymbol::isBaseVirtualFunction() {
  return InternalGetDIAValue(&IDiaSymbol::get_intro);
}

DIAResult<BOOL> DIASymbol::isIntrinsic() {
  return InternalGetDIAValue(&IDiaSymbol::get_intrinsic);
}

DIAResult<BOOL> DIASymbol::isAcceleratorGroupSharedLocal() {
  return InternalGetDIAValue(&IDiaSymbol::get_isAcceleratorGroupSharedLocal);
}

DIAResult<BOOL> DIASymbol::isAcceleratorPointerTagLiveRange() {
  return InternalGetDIAValue(&IDiaSymbol::get_isAcceleratorPointerTagLiveRange);
}

DIAResult<BOOL> DIASymbol::isAcceleratorStubFunction() {
  return InternalGetDIAValue(&IDiaSymbol::get_isAcceleratorStubFunction);
}

DIAResult<BOOL> DIASymbol::isAggregated() {
  return InternalGetDIAValue(&IDiaSymbol::get_isAggregated);
}

DIAResult<BOOL> DIASymbol::isConstructorVirtualBase() {
  return InternalGetDIAValue(&IDiaSymbol::get_isConstructorVirtualBase);
}

DIAResult<BOOL> DIASymbol::hasStrictGSCheck() {
  return InternalGetDIAValue(&IDiaSymbol::get_strictGSCheck);
}

DIAResult<BOOL> DIASymbol::isLocationControlFlowDependent() {
  return InternalGetDIAValue(&IDiaSymbol::get_isLocationControlFlowDependent);
}

DIAResult<BOOL> DIASymbol::hasCTypes() {
  return InternalGetDIAValue(&IDiaSymbol::get_isCTypes);
}

DIAResult<BOOL> DIASymbol::isCVTCIL() {
  return InternalGetDIAValue(&IDiaSymbol::get_isCVTCIL);
}

DIAResult<BOOL> DIASymbol::isDataAligned() {
  return InternalGetDIAValue(&IDiaSymbol::get_isDataAligned);
}

DIAResult<BOOL> DIASymbol::isHLSLData() {
  return InternalGetDIAValue(&IDiaSymbol::get_isHLSLData);
}

DIAResult<BOOL> DIASymbol::isHotpatchable() {
  return InternalGetDIAValue(&IDiaSymbol::get_isHotpatchable);
}

DIAResult<BOOL> DIASymbol::isLTCG() {
  return InternalGetDIAValue(&IDiaSymbol::get_isLTCG);
}

DIAResult<BOOL> DIASymbol::isMatrixRowMajor() {
  return InternalGetDIAValue(&IDiaSymbol::get_isMatrixRowMajor);
}

DIAResult<BOOL> DIASymbol::isMSILNetmodule() {
  return InternalGetDIAValue(&IDiaSymbol::get_isMSILNetmodule);
}

DIAResult<BOOL> DIASymbol::isMultipleInheritance() {
  return InternalGetDIAValue(&IDiaSymbol::get_isMultipleInheritance);
}

DIAResult<BOOL> DIASymbol::isNaked() {
  return InternalGetDIAValue(&IDiaSymbol::get_isNaked);
}

DIAResult<BOOL> DIASymbol::isOptimizedAway() {
  return InternalGetDIAValue(&IDiaSymbol::get_isOptimizedAway);
}

DIAResult<BOOL> DIASymbol::isPointerBasedOnSymbolValue() {
  return InternalGetDIAValue(&IDiaSymbol::get_isPointerBasedOnSymbolValue);
}

DIAResult<BOOL> DIASymbol::isPointerToDataMember() {
  return InternalGetDIAValue(&IDiaSymbol::get_isPointerToDataMember);
}

DIAResult<BOOL> DIASymbol::isPointerToMemberFunction() {
  return InternalGetDIAValue(&IDiaSymbol::get_isPointerToMemberFunction);
}

DIAResult<BOOL> DIASymbol::isReturnValue() {
  return InternalGetDIAValue(&IDiaSymbol::get_isReturnValue);
}

DIAResult<BOOL> DIASymbol::isSdl() {
  return InternalGetDIAValue(&IDiaSymbol::get_isSdl);
}

DIAResult<BOOL> DIASymbol::isSingleInheritance() {
  return InternalGetDIAValue(&IDiaSymbol::get_isSingleInheritance);
}

DIAResult<BOOL> DIASymbol::isSplitted() {
  return InternalGetDIAValue(&IDiaSymbol::get_isSplitted);
}

DIAResult<BOOL> DIASymbol::isStatic() {
  return InternalGetDIAValue(&IDiaSymbol::get_isStatic);
}

DIAResult<BOOL> DIASymbol::isStripped() {
  return InternalGetDIAValue(&IDiaSymbol::get_isStripped);
}

DIAResult<BOOL> DIASymbol::isVirtualInheritance() {
  return InternalGetDIAValue(&IDiaSymbol::get_isVirtualInheritance);
}

DIAResult<CV_CFL_LANG> DIASymbol::getLanguage() {
  return InternalGetDIAValue<DWORD, CV_CFL_LANG>(&IDiaSymbol::get_language);
}

DIAResult<BOOL> DIASymbol::isSafeBuffers() {
  return InternalGetDIAValue(&IDiaSymbol::get_isSafeBuffers);
}

DIAResult<ULONGLONG> DIASymbol::getLength() {
  return InternalGetDIAValue(&IDiaSymbol::get_length);
}

DIAResult<DWORD> DIASymbol::getLexicalParentId() {
  return InternalGetDIAValue(&IDiaSymbol::get_lexicalParentId);
}

DIAResult<DiaSymbolPtr> DIASymbol::getLexicalParent() {
  return InternalGetDIAValue<IDiaSymbol *, DiaSymbolPtr>(
      &IDiaSymbol::get_lexicalParent);
}

DIAResult<DWORD> DIASymbol::getLocalBasePointerRegisterId() {
  return InternalGetDIAValue(&IDiaSymbol::get_localBasePointerRegisterId);
}

DIAResult<DIAString> DIASymbol::getLibraryName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_libraryName);
}

DIAResult<ULONGLONG> DIASymbol::getLiveRangeLength() {
  return InternalGetDIAValue(&IDiaSymbol::get_liveRangeLength);
}

DIAResult<DWORD> DIASymbol::getLiveRangeStartAddressSection() {
  return InternalGetDIAValue(&IDiaSymbol::get_liveRangeStartAddressSection);
}

DIAResult<DWORD> DIASymbol::getLiveRangeStartAddressOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_liveRangeStartAddressOffset);
}

DIAResult<DWORD> DIASymbol::getLiveRangeStartRelativeVirtualAddress() {
  return InternalGetDIAValue(
      &IDiaSymbol::get_liveRangeStartRelativeVirtualAddress);
}

DIAResult<LocationType> DIASymbol::getLocationType() {
  return InternalGetDIAValue<DWORD, LocationType>(
      &IDiaSymbol::get_locationType);
}

DIAResult<DWORD> DIASymbol::getLowerBoundId() {
  return InternalGetDIAValue(&IDiaSymbol::get_lowerBoundId);
}

DIAResult<MachineTypeEnum> DIASymbol::getMachineType() {
  return InternalGetDIAValue<DWORD, MachineTypeEnum>(
      &IDiaSymbol::get_machineType);
}

DIAResult<BOOL> DIASymbol::isManagedRef() {
  return InternalGetDIAValue(&IDiaSymbol::get_managed);
}

DIAResult<DWORD> DIASymbol::getMemorySpaceKind() {
  return InternalGetDIAValue(&IDiaSymbol::get_memorySpaceKind);
}

DIAResult<BOOL> DIASymbol::isMsilRef() {
  return InternalGetDIAValue(&IDiaSymbol::get_msil);
}

DIAResult<DIAString> DIASymbol::getName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_name);
}

DIAResult<BOOL> DIASymbol::isNested() {
  return InternalGetDIAValue(&IDiaSymbol::get_nested);
}

DIAResult<BOOL> DIASymbol::hasNoInlineAttribute() {
  return InternalGetDIAValue(&IDiaSymbol::get_noInline);
}

DIAResult<BOOL> DIASymbol::hasNoReturnAttribute() {
  return InternalGetDIAValue(&IDiaSymbol::get_noReturn);
}

DIAResult<BOOL> DIASymbol::getNoStackOrdering() {
  return InternalGetDIAValue(&IDiaSymbol::get_noStackOrdering);
}

DIAResult<BOOL> DIASymbol::isUnreached() {
  return InternalGetDIAValue(&IDiaSymbol::get_notReached);
}

DIAResult<DWORD> DIASymbol::getNumberOfAcceleratorPointerTags() {
  return InternalGetDIAValue(&IDiaSymbol::get_numberOfAcceleratorPointerTags);
}

DIAResult<DWORD> DIASymbol::getNumberOfModifiers() {
  return InternalGetDIAValue(&IDiaSymbol::get_numberOfModifiers);
}

DIAResult<DWORD> DIASymbol::getNumberOfRegisterIndices() {
  return InternalGetDIAValue(&IDiaSymbol::get_numberOfRegisterIndices);
}

DIAResult<DWORD> DIASymbol::getNumberOfRows() {
  return InternalGetDIAValue(&IDiaSymbol::get_numberOfRows);
}

DIAResult<DWORD> DIASymbol::getNumberOfColumns() {
  return InternalGetDIAValue(&IDiaSymbol::get_numberOfColumns);
}

DIAResult<DIAString> DIASymbol::getObjectFileName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_objectFileName);
}

DIAResult<DWORD> DIASymbol::getOemSymbolId() {
  return InternalGetDIAValue(&IDiaSymbol::get_oemSymbolId);
}

DIAResult<LONG> DIASymbol::getOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_offset);
}

DIAResult<DWORD> DIASymbol::getOffsetInUdt() {
  return InternalGetDIAValue(&IDiaSymbol::get_offsetInUdt);
}

DIAResult<BOOL> DIASymbol::hasOptimizedCodeDebugInfo() {
  return InternalGetDIAValue(&IDiaSymbol::get_optimizedCodeDebugInfo);
}

DIAResult<BOOL> DIASymbol::hasOverloadedOperator() {
  return InternalGetDIAValue(&IDiaSymbol::get_overloadedOperator);
}

DIAResult<BOOL> DIASymbol::isPacked() {
  return InternalGetDIAValue(&IDiaSymbol::get_packed);
}

DIAResult<CV_CPU_TYPE_e> DIASymbol::getPlatform() {
  return InternalGetDIAValue<DWORD, CV_CPU_TYPE_e>(&IDiaSymbol::get_platform);
}

DIAResult<BOOL> DIASymbol::isPureVirtual() {
  return InternalGetDIAValue(&IDiaSymbol::get_pure);
}

DIAResult<DWORD> DIASymbol::getRank() {
  return InternalGetDIAValue(&IDiaSymbol::get_rank);
}

DIAResult<BOOL> DIASymbol::isReference() {
  return InternalGetDIAValue(&IDiaSymbol::get_reference);
}

DIAResult<BOOL> DIASymbol::isRValueReference() {
  return InternalGetDIAValue(&IDiaSymbol::get_RValueReference);
}

DIAResult<DWORD> DIASymbol::getRegisterId() {
  return InternalGetDIAValue(&IDiaSymbol::get_registerId);
}

DIAResult<DWORD> DIASymbol::getRegisterType() {
  return InternalGetDIAValue(&IDiaSymbol::get_registerType);
}

DIAResult<DWORD> DIASymbol::getRelativeVirtualAddress() {
  return InternalGetDIAValue(&IDiaSymbol::get_relativeVirtualAddress);
}

DIAResult<BOOL> DIASymbol::isRestrictedType() {
  return InternalGetDIAValue(&IDiaSymbol::get_restrictedType);
}

DIAResult<DWORD> DIASymbol::getSamplerSlot() {
  return InternalGetDIAValue(&IDiaSymbol::get_samplerSlot);
}

DIAResult<BOOL> DIASymbol::isScoped() {
  return InternalGetDIAValue(&IDiaSymbol::get_scoped);
}

DIAResult<BOOL> DIASymbol::isRefUdt() {
  return InternalGetDIAValue(&IDiaSymbol::get_isRefUdt);
}

DIAResult<BOOL> DIASymbol::isValueUdt() {
  return InternalGetDIAValue(&IDiaSymbol::get_isValueUdt);
}

DIAResult<BOOL> DIASymbol::isInterfaceUdt() {
  return InternalGetDIAValue(&IDiaSymbol::get_isInterfaceUdt);
}

DIAResult<BOOL> DIASymbol::isCxxReturnUdt() {
  return InternalGetDIAValue(&IDiaSymbol::get_isCxxReturnUdt);
}

DIAResult<DWORD> DIASymbol::getSignature() {
  return InternalGetDIAValue(&IDiaSymbol::get_signature);
}

DIAResult<DWORD> DIASymbol::getSizeInUdt() {
  return InternalGetDIAValue(&IDiaSymbol::get_sizeInUdt);
}

DIAResult<DWORD> DIASymbol::getSlot() {
  return InternalGetDIAValue(&IDiaSymbol::get_slot);
}

DIAResult<DIAString> DIASymbol::getSourceFileName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_sourceFileName);
}

DIAResult<DWORD> DIASymbol::getStride() {
  return InternalGetDIAValue(&IDiaSymbol::get_stride);
}

DIAResult<DWORD> DIASymbol::getSubTypeId() {
  return InternalGetDIAValue(&IDiaSymbol::get_subTypeId);
}

DIAResult<DIAString> DIASymbol::getSymbolsFileName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_symbolsFileName);
}

DIAResult<DWORD> DIASymbol::getSymIndexId() {
  return InternalGetDIAValue(&IDiaSymbol::get_symIndexId);
}

DIAResult<DiaSymTagEnum> DIASymbol::getSymTag() {
  return InternalGetDIAValue<DWORD, DiaSymTagEnum>(&IDiaSymbol::get_symTag);
}

DIAResult<DWORD> DIASymbol::getTargetOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_targetOffset);
}

DIAResult<DWORD> DIASymbol::getTargetRelativeVirtualAddress() {
  return InternalGetDIAValue(&IDiaSymbol::get_targetRelativeVirtualAddress);
}

DIAResult<DWORD> DIASymbol::getTargetSection() {
  return InternalGetDIAValue(&IDiaSymbol::get_targetSection);
}

DIAResult<ULONGLONG> DIASymbol::getTargetVirtualAddress() {
  return InternalGetDIAValue(&IDiaSymbol::get_targetVirtualAddress);
}

DIAResult<DWORD> DIASymbol::getTextureSlot() {
  return InternalGetDIAValue(&IDiaSymbol::get_textureSlot);
}

DIAResult<LONG> DIASymbol::getThisAdjust() {
  return InternalGetDIAValue(&IDiaSymbol::get_thisAdjust);
}

DIAResult<THUNK_ORDINAL> DIASymbol::getThunkOrdinal() {
  return InternalGetDIAValue<DWORD, THUNK_ORDINAL>(
      &IDiaSymbol::get_thunkOrdinal);
}

DIAResult<DWORD> DIASymbol::getTimeStamp() {
  return InternalGetDIAValue(&IDiaSymbol::get_timeStamp);
}

DIAResult<DWORD> DIASymbol::getToken() {
  return InternalGetDIAValue(&IDiaSymbol::get_token);
}

DIAResult<DWORD> DIASymbol::getUavSlot() {
  return InternalGetDIAValue(&IDiaSymbol::get_uavSlot);
}

DIAResult<UdtKind> DIASymbol::getUdtKind() {
  return InternalGetDIAValue<DWORD, UdtKind>(&IDiaSymbol::get_udtKind);
}

DIAResult<BOOL> DIASymbol::isUnalignedType() {
  return InternalGetDIAValue(&IDiaSymbol::get_unalignedType);
}

DIAResult<DIAString> DIASymbol::getUndecoratedName() {
  return InternalGetDIAStringValue(&IDiaSymbol::get_undecoratedName);
}

DIAResult<DWORD> DIASymbol::getUnmodifiedTypeId() {
  return InternalGetDIAValue(&IDiaSymbol::get_unmodifiedTypeId);
}

DIAResult<DWORD> DIASymbol::getUpperBoundId() {
  return InternalGetDIAValue(&IDiaSymbol::get_upperBoundId);
}

DIAResult<BOOL> DIASymbol::isVirtual() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtual);
}

DIAResult<ULONGLONG> DIASymbol::getVirtualAddress() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtualAddress);
}

DIAResult<BOOL> DIASymbol::isVirtualBaseClass() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtualBaseClass);
}

DIAResult<DWORD> DIASymbol::getVirtualBaseDispIndex() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtualBaseDispIndex);
}

DIAResult<DWORD> DIASymbol::getVirtualBaseOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtualBaseOffset);
}

DIAResult<LONG> DIASymbol::getVirtualBasePointerOffset() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtualBasePointerOffset);
}

DIAResult<DWORD> DIASymbol::getVirtualTableShapeId() {
  return InternalGetDIAValue(&IDiaSymbol::get_virtualTableShapeId);
}

DIAResult<BOOL> DIASymbol::isVolatileType() {
  return InternalGetDIAValue(&IDiaSymbol::get_volatileType);
}
