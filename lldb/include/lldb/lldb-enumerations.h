//===-- lldb-enumerations.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_enumerations_h_
#define LLDB_enumerations_h_

#include "llvm/System/Host.h"

namespace lldb {

//----------------------------------------------------------------------
// Process and Thread States
//----------------------------------------------------------------------
typedef enum StateType
{
    eStateInvalid = 0,
    eStateUnloaded,
    eStateAttaching,
    eStateLaunching,
    eStateStopped,
    eStateRunning,
    eStateStepping,
    eStateCrashed,
    eStateDetached,
    eStateExited,
    eStateSuspended
} StateType;

//----------------------------------------------------------------------
// Thread Step Types
//----------------------------------------------------------------------
typedef enum StepType
{
    eStepTypeNone,
    eStepTypeTrace,     ///< Single step one instruction.
    eStepTypeTraceOver, ///< Single step one instruction, stepping over.
    eStepTypeInto,      ///< Single step into a specified context.
    eStepTypeOver,      ///< Single step over a specified context.
    eStepTypeOut        ///< Single step out a specified context.
} StepType;

//----------------------------------------------------------------------
// Thread Run Modes
//----------------------------------------------------------------------
typedef enum RunMode {
    eOnlyThisThread,
    eAllThreads,
    eOnlyDuringStepping
} RunMode;

//----------------------------------------------------------------------
// Address Types
//----------------------------------------------------------------------
typedef enum AddressType
{
    eAddressTypeInvalid = 0,
    eAddressTypeFile, ///< Address is an address as found in an object or symbol file
    eAddressTypeLoad, ///< Address is an address as in the current target inferior process
    eAddressTypeHost  ///< Address is an address in the process that is running this code
} AddressType;

//----------------------------------------------------------------------
// Byte ordering definitions
//----------------------------------------------------------------------
typedef enum ByteOrder
{
    eByteOrderInvalid   = 0,
    eByteOrderLittle    = 1234,
    eByteOrderBig       = 4321,
    eByteOrderPDP       = 3412
} ByteOrder;

inline ByteOrder getHostByteOrder() {
  if (llvm::sys::isLittleEndianHost())
    return eByteOrderLittle;
  return eByteOrderBig;
}

// FIXME: Replace uses of eByteOrderHost with getHostByteOrder()!
#define eByteOrderHost getHostByteOrder()

//----------------------------------------------------------------------
// Register encoding definitions
//----------------------------------------------------------------------
typedef enum Encoding
{
    eEncodingInvalid = 0,
    eEncodingUint,               // unsigned integer
    eEncodingSint,               // signed integer
    eEncodingIEEE754,            // float
    eEncodingVector              // vector registers
} Encoding;

//----------------------------------------------------------------------
// Display format definitions
//----------------------------------------------------------------------
typedef enum Format
{
    eFormatDefault = 0,
    eFormatInvalid = 0,
    eFormatBoolean,
    eFormatBinary,
    eFormatBytes,
    eFormatBytesWithASCII,
    eFormatChar,
    eFormatCharPrintable,   // Only printable characters, space if not printable
    eFormatComplex,
    eFormatCString,         // NULL terminated C strings
    eFormatDecimal,
    eFormatEnum,
    eFormatHex,
    eFormatFloat,
    eFormatOctal,
    eFormatUnicode16,
    eFormatUnicode32,
    eFormatUnsigned,
    eFormatPointer,
    eFormatVectorOfChar,
    eFormatVectorOfSInt8,
    eFormatVectorOfUInt8,
    eFormatVectorOfSInt16,
    eFormatVectorOfUInt16,
    eFormatVectorOfSInt32,
    eFormatVectorOfUInt32,
    eFormatVectorOfSInt64,
    eFormatVectorOfUInt64,
    eFormatVectorOfFloat32,
    eFormatVectorOfFloat64,
    eFormatVectorOfUInt128

} Format;

//----------------------------------------------------------------------
// Description levels for "void GetDescription(Stream *, DescriptionLevel)" calls
//----------------------------------------------------------------------
typedef enum DescriptionLevel
{
    eDescriptionLevelBrief = 0,
    eDescriptionLevelFull,
    eDescriptionLevelVerbose,
    kNumDescriptionLevels
} DescriptionLevel;

//----------------------------------------------------------------------
// Script interpreter types
//----------------------------------------------------------------------
typedef enum ScriptLanguage
{
    eScriptLanguageNone,
    eScriptLanguagePython,
    eScriptLanguageDefault = eScriptLanguagePython
} ScriptLanguage;

//----------------------------------------------------------------------
// Register numbering types
//----------------------------------------------------------------------
typedef enum RegisterKind
{
    eRegisterKindGCC = 0,
    eRegisterKindDWARF,
    eRegisterKindGeneric,
    eRegisterKindGDB,
    kNumRegisterKinds
} RegisterKind;

//----------------------------------------------------------------------
// Thread stop reasons
//----------------------------------------------------------------------
typedef enum StopReason
{
    eStopReasonInvalid = 0,
    eStopReasonNone,
    eStopReasonTrace,
    eStopReasonBreakpoint,
    eStopReasonWatchpoint,
    eStopReasonSignal,
    eStopReasonException,
    eStopReasonPlanComplete
} StopReason;

//----------------------------------------------------------------------
// Votes - Need a tri-state, yes, no, no opinion...
//----------------------------------------------------------------------
typedef enum Vote
{
    eVoteNo         = -1,
    eVoteNoOpinion  =  0,
    eVoteYes        =  1
} Vote;

//----------------------------------------------------------------------
// Symbol types
//----------------------------------------------------------------------
typedef enum SymbolType
{
    eSymbolTypeAny = 0,
    eSymbolTypeInvalid = 0,
    eSymbolTypeAbsolute,
    eSymbolTypeExtern,
    eSymbolTypeCode,
    eSymbolTypeData,
    eSymbolTypeTrampoline,
    eSymbolTypeRuntime,
    eSymbolTypeException,
    eSymbolTypeSourceFile,
    eSymbolTypeHeaderFile,
    eSymbolTypeObjectFile,
    eSymbolTypeFunction,
    eSymbolTypeFunctionEnd,
    eSymbolTypeCommonBlock,
    eSymbolTypeBlock,
    eSymbolTypeStatic,
    eSymbolTypeGlobal,
    eSymbolTypeLocal,
    eSymbolTypeParam,
    eSymbolTypeVariable,
    eSymbolTypeVariableType,
    eSymbolTypeLineEntry,
    eSymbolTypeLineHeader,
    eSymbolTypeScopeBegin,
    eSymbolTypeScopeEnd,
    eSymbolTypeAdditional, // When symbols take more than one entry, the extra entries get this type
    eSymbolTypeCompiler,
    eSymbolTypeInstrumentation,
    eSymbolTypeUndefined
} SymbolType;


//----------------------------------------------------------------------
// Command Return Status Types
//----------------------------------------------------------------------
typedef enum ReturnStatus
{
    eReturnStatusInvalid,
    eReturnStatusSuccessFinishNoResult,
    eReturnStatusSuccessFinishResult,
    eReturnStatusSuccessContinuingNoResult,
    eReturnStatusSuccessContinuingResult,
    eReturnStatusStarted,
    eReturnStatusFailed,
    eReturnStatusQuit
} ReturnStatus;


//----------------------------------------------------------------------
// Connection Status Types
//----------------------------------------------------------------------
typedef enum ConnectionStatus
{
    eConnectionStatusSuccess,         // Success
    eConnectionStatusError,           // Check GetError() for details
    eConnectionStatusTimedOut,        // Request timed out
    eConnectionStatusNoConnection,    // No connection
    eConnectionStatusLostConnection   // Lost connection while connected to a valid connection
} ConnectionStatus;


typedef enum ErrorType
{
    eErrorTypeInvalid,
    eErrorTypeGeneric,      ///< Generic errors that can be any value.
    eErrorTypeMachKernel,   ///< Mach kernel error codes.
    eErrorTypePOSIX         ///< POSIX error codes.
} ErrorType;


typedef enum ValueType
{
    eValueTypeInvalid           = 0,
    eValueTypeVariableGlobal    = 1,    // globals variable
    eValueTypeVariableStatic    = 2,    // static variable
    eValueTypeVariableArgument  = 3,    // function argument variables
    eValueTypeVariableLocal     = 4,    // function local variables
    eValueTypeRegister          = 5,    // stack frame register value
    eValueTypeRegisterSet       = 6     // A collection of stack frame register values
} ValueType;

//----------------------------------------------------------------------
// Token size/granularities for Input Readers
//----------------------------------------------------------------------

typedef enum InputReaderGranularity
{
    eInputReaderGranularityInvalid = 0,
    eInputReaderGranularityByte,
    eInputReaderGranularityWord,
    eInputReaderGranularityLine,
    eInputReaderGranularityAll
} InputReaderGranularity;

//------------------------------------------------------------------
/// These mask bits allow a common interface for queries that can
/// limit the amount of information that gets parsed to only the
/// information that is requested. These bits also can indicate what
/// actually did get resolved during query function calls.
///
/// Each definition corresponds to a one of the member variables
/// in this class, and requests that that item be resolved, or
/// indicates that the member did get resolved.
//------------------------------------------------------------------
typedef enum SymbolContextItem
{
    eSymbolContextTarget     = (1 << 0), ///< Set when \a target is requested from a query, or was located in query results
    eSymbolContextModule     = (1 << 1), ///< Set when \a module is requested from a query, or was located in query results
    eSymbolContextCompUnit   = (1 << 2), ///< Set when \a comp_unit is requested from a query, or was located in query results
    eSymbolContextFunction   = (1 << 3), ///< Set when \a function is requested from a query, or was located in query results
    eSymbolContextBlock      = (1 << 4), ///< Set when the deepest \a block is requested from a query, or was located in query results
    eSymbolContextLineEntry  = (1 << 5), ///< Set when \a line_entry is requested from a query, or was located in query results
    eSymbolContextSymbol     = (1 << 6), ///< Set when \a symbol is requested from a query, or was located in query results
    eSymbolContextEverything = ((eSymbolContextSymbol << 1) - 1)  ///< Indicates to try and lookup everything up during a query.
} SymbolContextItem;

typedef enum Permissions
{
    ePermissionsWritable = (1 << 0),
    ePermissionsReadable = (1 << 1),
    ePermissionsExecutable = (1 << 2)
} Permissions;

typedef enum SectionType
{
    eSectionTypeInvalid,
    eSectionTypeCode,
    eSectionTypeContainer,              // The section contains child sections
    eSectionTypeData,
    eSectionTypeDataCString,            // Inlined C string data
    eSectionTypeDataCStringPointers,    // Pointers to C string data
    eSectionTypeDataSymbolAddress,      // Address of a symbol in the symbol table
    eSectionTypeData4,
    eSectionTypeData8,
    eSectionTypeData16,
    eSectionTypeDataPointers,
    eSectionTypeDebug,
    eSectionTypeZeroFill,
    eSectionTypeDataObjCMessageRefs,    // Pointer to function pointer + selector
    eSectionTypeDataObjCCFStrings,      // Objective C const CFString/NSString objects
    eSectionTypeOther

} SectionType;


typedef enum InputReaderAction
{
    eInputReaderActivate,   // reader is newly pushed onto the reader stack 
    eInputReaderReactivate, // reader is on top of the stack again after another reader was popped off 
    eInputReaderDeactivate, // another reader was pushed on the stack 
    eInputReaderGotToken,   // reader got one of its tokens (granularity)
    eInputReaderDone        // reader was just popped off the stack and is done
} InputReaderAction;

} // namespace lldb


#endif  // LLDB_enumerations_h_
