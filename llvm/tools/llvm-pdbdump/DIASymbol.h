//===- DIASymbol.h - Dump debug info from a PDB file ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Provides a wrapper around the IDiaSymbol interface.  IDiaSymbol is used to
// represent any kind of symbol from functions, to compilands, to source files.
// It provides a monolithic interface of close to 200 operations, and the set
// of operations that are valid depends on the type of the symbol.  Since it is
// not clearly documented which set of operations is valid for which type of
// symbol, the best way of figuring it out is to dump every method for every
// symbol, and see which methods return errors.  This wrapper provides a clean
// way of doing this without involving needing to embed lots of unsightly
// HRESULT checking at every callsite.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_DIASYMBOL_H
#define LLVM_TOOLS_LLVMPDBDUMP_DIASYMBOL_H

#include "DIAExtras.h"
#include "llvm/ADT/SmallString.h"

namespace llvm {
namespace sys {
namespace windows {

class DIASymbol {
public:
  DIASymbol(IDiaSymbol *DiaSymbol);
  ~DIASymbol();

  /// Dumps the value of every property (if it exists) with a default name.
  /// This is useful for understanding what symbol types support what methods
  /// during development time.
  void fullDump(int IndentLevel);

// TODO: The following methods are present on IDiaSymbol but do not yet have
// wrapper methods.
//
// HRESULT get_value(VARIANT *pRetVal) = 0;
// HRESULT get_undecoratedNameEx(DWORD undecorateOptions, BSTR *name) = 0;
// HRESULT getSrcLineOnTypeDefn(IDiaLineNumber **ppResult) = 0;
// HRESULT get_dataBytes(DWORD cbData, DWORD *pcbData, BYTE *pbData) = 0;
// HRESULT get_types(DWORD cTypes, DWORD *pcTypes, IDiaSymbol **pTypes) = 0;
// HRESULT get_typeIds(DWORD cTypeIds, DWORD *pcTypeIds, DWORD *pdwTypeIds) = 0;
// HRESULT get_numericProperties(DWORD cnt, DWORD *pcnt,
//                               DWORD *pProperties) = 0;
// HRESULT get_modifierValues(DWORD cnt, DWORD *pcnt, WORD *pModifiers) = 0;
// HRESULT get_acceleratorPointerTags(DWORD cnt, DWORD *pcnt, DWORD
//                                    *pPointerTags) = 0;
// HRESULT get_hfaFloat(BOOL *pRetVal) = 0;
// HRESULT get_hfaDouble(BOOL *pRetVal) = 0;
// HRESULT get_paramBasePointerRegisterId(DWORD *pRetVal) = 0;
// HRESULT get_isWinRTPointer(BOOL *pRetVal) = 0;

#if (_MSC_FULL_VER >= 180031101)
  // These methods are only available on VS 2013 SP 4 and higher.
  DIAResult<BOOL> isPGO();
  DIAResult<BOOL> hasValidPGOCounts();
  DIAResult<BOOL> isOptimizedForSpeed();
  DIAResult<DWORD> getPGOEntryCount();
  DIAResult<DWORD> getPGOEdgeCount();
  DIAResult<ULONGLONG> getPGODynamicInstructionCount();
  DIAResult<DWORD> getStaticSize();
  DIAResult<DWORD> getFinalLiveStaticSize();
  DIAResult<DIAString> getPhaseName();
  DIAResult<BOOL> hasControlFlowCheck();
#endif

  DIAResult<DiaSymbolPtr> getLexicalParent();
  DIAResult<DiaSymbolPtr> getClassParent();
  DIAResult<DiaSymbolPtr> getType();
  DIAResult<DiaSymbolPtr> getArrayIndexType();
  DIAResult<DiaSymbolPtr> getVirtualTableShape();
  DIAResult<DiaSymbolPtr> getLowerBound();
  DIAResult<DiaSymbolPtr> getUpperBound();
  DIAResult<DiaSymbolPtr> getObjectPointerType();
  DIAResult<DiaSymbolPtr> getContainer();
  DIAResult<DiaSymbolPtr> getVirtualBaseTableType();
  DIAResult<DiaSymbolPtr> getUnmodifiedType();
  DIAResult<DiaSymbolPtr> getSubType();
  DIAResult<DiaSymbolPtr> getBaseSymbol();

  DIAResult<DWORD> getAccess();
  DIAResult<DWORD> getAddressOffset();
  DIAResult<DWORD> getAddressSection();
  DIAResult<DWORD> getAge();
  DIAResult<DWORD> getArrayIndexTypeId();
  DIAResult<DWORD> getBackEndBuild();
  DIAResult<DWORD> getBackEndMajor();
  DIAResult<DWORD> getBackEndMinor();
  DIAResult<DWORD> getBackEndQFE();
  DIAResult<DWORD> getBaseDataOffset();
  DIAResult<DWORD> getBaseDataSlot();
  DIAResult<DWORD> getBaseSymbolId();
  DIAResult<DWORD> getBaseType();
  DIAResult<DWORD> getBitPosition();
  DIAResult<DWORD> getBuiltInKind();
  DIAResult<CV_call_e> getCallingConvention();
  DIAResult<DWORD> getClassParentId();
  DIAResult<DIAString> getCompilerName();
  DIAResult<DWORD> getCount();
  DIAResult<DWORD> getCountLiveRanges();
  DIAResult<DWORD> getFrontEndBuild();
  DIAResult<DWORD> getFrontEndMajor();
  DIAResult<DWORD> getFrontEndMinor();
  DIAResult<DWORD> getFrontEndQFE();
  DIAResult<CV_CFL_LANG> getLanguage();
  DIAResult<DWORD> getLexicalParentId();
  DIAResult<DIAString> getLibraryName();
  DIAResult<DWORD> getLiveRangeStartAddressOffset();
  DIAResult<DWORD> getLiveRangeStartAddressSection();
  DIAResult<DWORD> getLiveRangeStartRelativeVirtualAddress();
  DIAResult<DWORD> getLocalBasePointerRegisterId();
  DIAResult<DWORD> getLowerBoundId();
  DIAResult<DWORD> getMemorySpaceKind();
  DIAResult<DIAString> getName();
  DIAResult<DWORD> getNumberOfAcceleratorPointerTags();
  DIAResult<DWORD> getNumberOfColumns();
  DIAResult<DWORD> getNumberOfModifiers();
  DIAResult<DWORD> getNumberOfRegisterIndices();
  DIAResult<DWORD> getNumberOfRows();
  DIAResult<DIAString> getObjectFileName();
  DIAResult<DWORD> getOemSymbolId();
  DIAResult<DWORD> getOffsetInUdt();
  DIAResult<CV_CPU_TYPE_e> getPlatform();
  DIAResult<DWORD> getRank();
  DIAResult<DWORD> getRegisterId();
  DIAResult<DWORD> getRegisterType();
  DIAResult<DWORD> getRelativeVirtualAddress();
  DIAResult<DWORD> getSamplerSlot();
  DIAResult<DWORD> getSignature();
  DIAResult<DWORD> getSizeInUdt();
  DIAResult<DWORD> getSlot();
  DIAResult<DIAString> getSourceFileName();
  DIAResult<DWORD> getStride();
  DIAResult<DWORD> getSubTypeId();
  DIAResult<DIAString> getSymbolsFileName();
  DIAResult<DWORD> getSymIndexId();
  DIAResult<DWORD> getTargetOffset();
  DIAResult<DWORD> getTargetRelativeVirtualAddress();
  DIAResult<DWORD> getTargetSection();
  DIAResult<DWORD> getTextureSlot();
  DIAResult<DWORD> getTimeStamp();
  DIAResult<DWORD> getToken();
  DIAResult<DWORD> getUavSlot();
  DIAResult<DIAString> getUndecoratedName();
  DIAResult<DWORD> getUnmodifiedTypeId();
  DIAResult<DWORD> getUpperBoundId();
  DIAResult<DWORD> getVirtualBaseDispIndex();
  DIAResult<DWORD> getVirtualBaseOffset();
  DIAResult<DWORD> getVirtualTableShapeId();
  DIAResult<DataKind> getDataKind();
  DIAResult<DiaSymTagEnum> getSymTag();
  DIAResult<GUID> getGuid();
  DIAResult<LONG> getOffset();
  DIAResult<LONG> getThisAdjust();
  DIAResult<LONG> getVirtualBasePointerOffset();
  DIAResult<LocationType> getLocationType();
  DIAResult<MachineTypeEnum> getMachineType();
  DIAResult<THUNK_ORDINAL> getThunkOrdinal();
  DIAResult<ULONGLONG> getLength();
  DIAResult<ULONGLONG> getLiveRangeLength();
  DIAResult<ULONGLONG> getTargetVirtualAddress();
  DIAResult<ULONGLONG> getVirtualAddress();
  DIAResult<UdtKind> getUdtKind();
  DIAResult<BOOL> hasConstructor();
  DIAResult<BOOL> hasCustomCallingConvention();
  DIAResult<BOOL> hasFarReturn();
  DIAResult<BOOL> isCode();
  DIAResult<BOOL> isCompilerGenerated();
  DIAResult<BOOL> isConstType();
  DIAResult<BOOL> isEditAndContinueEnabled();
  DIAResult<BOOL> isFunction();
  DIAResult<BOOL> getAddressTaken();
  DIAResult<BOOL> getNoStackOrdering();
  DIAResult<BOOL> hasAlloca();
  DIAResult<BOOL> hasAssignmentOperator();
  DIAResult<BOOL> hasCTypes();
  DIAResult<BOOL> hasCastOperator();
  DIAResult<BOOL> hasDebugInfo();
  DIAResult<BOOL> hasEH();
  DIAResult<BOOL> hasEHa();
  DIAResult<BOOL> hasInlAsm();
  DIAResult<BOOL> hasInlineAttribute();
  DIAResult<BOOL> hasInterruptReturn();
  DIAResult<BOOL> hasLongJump();
  DIAResult<BOOL> hasManagedCode();
  DIAResult<BOOL> hasNestedTypes();
  DIAResult<BOOL> hasNoInlineAttribute();
  DIAResult<BOOL> hasNoReturnAttribute();
  DIAResult<BOOL> hasOptimizedCodeDebugInfo();
  DIAResult<BOOL> hasOverloadedOperator();
  DIAResult<BOOL> hasSEH();
  DIAResult<BOOL> hasSecurityChecks();
  DIAResult<BOOL> hasSetJump();
  DIAResult<BOOL> hasStrictGSCheck();
  DIAResult<BOOL> isAcceleratorGroupSharedLocal();
  DIAResult<BOOL> isAcceleratorPointerTagLiveRange();
  DIAResult<BOOL> isAcceleratorStubFunction();
  DIAResult<BOOL> isAggregated();
  DIAResult<BOOL> isBaseVirtualFunction();
  DIAResult<BOOL> isCVTCIL();
  DIAResult<BOOL> isConstructorVirtualBase();
  DIAResult<BOOL> isCxxReturnUdt();
  DIAResult<BOOL> isDataAligned();
  DIAResult<BOOL> isHLSLData();
  DIAResult<BOOL> isHotpatchable();
  DIAResult<BOOL> isIndirectVirtualBaseClass();
  DIAResult<BOOL> isInterfaceUdt();
  DIAResult<BOOL> isIntrinsic();
  DIAResult<BOOL> isLTCG();
  DIAResult<BOOL> isLocationControlFlowDependent();
  DIAResult<BOOL> isMSILNetmodule();
  DIAResult<BOOL> isManagedRef();
  DIAResult<BOOL> isMatrixRowMajor();
  DIAResult<BOOL> isMsilRef();
  DIAResult<BOOL> isMultipleInheritance();
  DIAResult<BOOL> isNaked();
  DIAResult<BOOL> isNested();
  DIAResult<BOOL> isOptimizedAway();
  DIAResult<BOOL> isPacked();
  DIAResult<BOOL> isPointerBasedOnSymbolValue();
  DIAResult<BOOL> isPointerToDataMember();
  DIAResult<BOOL> isPointerToMemberFunction();
  DIAResult<BOOL> isPureVirtual();
  DIAResult<BOOL> isRValueReference();
  DIAResult<BOOL> isRefUdt();
  DIAResult<BOOL> isReference();
  DIAResult<BOOL> isRestrictedType();
  DIAResult<BOOL> isReturnValue();
  DIAResult<BOOL> isSafeBuffers();
  DIAResult<BOOL> isScoped();
  DIAResult<BOOL> isSdl();
  DIAResult<BOOL> isSingleInheritance();
  DIAResult<BOOL> isSplitted();
  DIAResult<BOOL> isStatic();
  DIAResult<BOOL> isStripped();
  DIAResult<BOOL> isUnalignedType();
  DIAResult<BOOL> isUnreached();
  DIAResult<BOOL> isValueUdt();
  DIAResult<BOOL> isVirtual();
  DIAResult<BOOL> isVirtualBaseClass();
  DIAResult<BOOL> isVirtualInheritance();
  DIAResult<BOOL> isVolatileType();

private:
  template <class T, class U = T>
  DIAResult<U>
  InternalGetDIAValue(HRESULT (__stdcall IDiaSymbol::*Method)(T *)) {
    T Value;
    if (S_OK == (Symbol->*Method)(&Value))
      return DIAResult<U>(U(Value));
    else
      return DIAResult<U>();
  }

  DIAResult<DIAString>
  InternalGetDIAStringValue(HRESULT (__stdcall IDiaSymbol::*Method)(BSTR *)) {
    BSTR String16;
    if (S_OK == (Symbol->*Method)(&String16)) {
      std::string String8;
      llvm::sys::windows::BSTRToUTF8(String16, String8);
      SysFreeString(String16);
      return DIAResult<DIAString>(DIAString(String8));
    } else
      return DIAResult<DIAString>();
  }

  IDiaSymbol *Symbol;
};

} // namespace windows
} // namespace sys
} // namespace llvm

#endif
