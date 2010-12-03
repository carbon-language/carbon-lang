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

#if !defined (__APPLE__)

#include <endian.h>

#endif

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
// Launch Flags
//----------------------------------------------------------------------
typedef enum LaunchFlags
{
    eLaunchFlagNone         = 0u,
    eLaunchFlagDisableASLR  = (1u << 0),  ///< Disable Address Space Layout Randomization
    eLaunchFlagDisableSTDIO = (1u << 1)   /// Disable stdio for inferior process (e.g. for a GUI app)
} LaunchFlags;
    
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
    eByteOrderPDP       = 3412,

#if defined (__APPLE__)

// On Mac OS X there are preprocessor defines automatically defined
// for the byte order that we can rely on.

#if   defined (__LITTLE_ENDIAN__)
    eByteOrderHost      = eByteOrderLittle
#elif defined (__BIG_ENDIAN__)
    eByteOrderHost      = eByteOrderBig
#elif defined (__PDP_ENDIAN__)
    eByteOrderHost      = eByteOrderPDP
#else
#error unable to detect endianness
#endif

#else

// On linux we rely upon the defines in <endian.h>

#if __BYTE_ORDER == __LITTLE_ENDIAN
    eByteOrderHost      = eByteOrderLittle
#elif __BYTE_ORDER == __BIG_ENDIAN
    eByteOrderHost      = eByteOrderBig
#elif __BYTE_ORDER == __PDP_ENDIAN
    eByteOrderHost      = eByteOrderPDP
#else
#error unable to detect endianness
#endif

#endif

} ByteOrder;

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
    eRegisterKindGCC = 0,    // the register numbers seen in eh_frame
    eRegisterKindDWARF,      // the register numbers seen DWARF
    eRegisterKindGeneric,    // insn ptr reg, stack ptr reg, etc not specific to any particular target
    eRegisterKindGDB,        // the register numbers gdb uses (matches stabs numbers?)
    eRegisterKindLLDB,       // lldb's internal register numbers
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
    eSymbolTypeCommonBlock,
    eSymbolTypeBlock,
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
    eConnectionStatusEndOfFile,       // End-of-file encountered
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
    eValueTypeRegisterSet       = 6,    // A collection of stack frame register values
    eValueTypeConstResult       = 7,    // constant result variables
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
    eSectionTypeDWARFDebugAbbrev,
    eSectionTypeDWARFDebugAranges,
    eSectionTypeDWARFDebugFrame,
    eSectionTypeDWARFDebugInfo,
    eSectionTypeDWARFDebugLine,
    eSectionTypeDWARFDebugLoc,
    eSectionTypeDWARFDebugMacInfo,
    eSectionTypeDWARFDebugPubNames,
    eSectionTypeDWARFDebugPubTypes,
    eSectionTypeDWARFDebugRanges,
    eSectionTypeDWARFDebugStr,
    eSectionTypeEHFrame,
    eSectionTypeOther

} SectionType;


typedef enum InputReaderAction
{
    eInputReaderActivate,   // reader is newly pushed onto the reader stack 
    eInputReaderReactivate, // reader is on top of the stack again after another reader was popped off 
    eInputReaderDeactivate, // another reader was pushed on the stack 
    eInputReaderGotToken,   // reader got one of its tokens (granularity)
    eInputReaderInterrupt,  // reader received an interrupt signal (probably from a control-c)
    eInputReaderEndOfFile,  // reader received an EOF char (probably from a control-d)
    eInputReaderDone        // reader was just popped off the stack and is done
} InputReaderAction;


typedef enum ArchitectureType 
{
    eArchTypeInvalid,
    eArchTypeMachO,
    eArchTypeELF,
    kNumArchTypes
} ArchitectureType;

typedef enum FunctionNameType 
{
    eFunctionNameTypeNone       = 0u,
    eFunctionNameTypeAuto       = (1u << 1),    // Automatically figure out which FunctionNameType
                                                // bits to set based on the function name.
    eFunctionNameTypeFull       = (1u << 2),    // The function name.
                                                // For C this is the same as just the name of the function
                                                // For C++ this is the demangled version of the mangled name.
                                                // For ObjC this is the full function signature with the + or
                                                // - and the square brackets and the class and selector
    eFunctionNameTypeBase       = (1u << 3),    // The function name only, no namespaces or arguments and no class 
                                                // methods or selectors will be searched.
    eFunctionNameTypeMethod     = (1u << 4),    // Find function by method name (C++) with no namespace or arguments
    eFunctionNameTypeSelector   = (1u << 5),    // Find function by selector name (ObjC) names
} FunctionNameType;


typedef enum BreakpointEventType
{
    eBreakpointEventTypeInvalidType         = (1u << 0),
    eBreakpointEventTypeAdded               = (1u << 1),
    eBreakpointEventTypeRemoved             = (1u << 2),
    eBreakpointEventTypeLocationsAdded      = (1u << 3),
    eBreakpointEventTypeLocationsRemoved    = (1u << 4),
    eBreakpointEventTypeLocationsResolved   = (1u << 5)
} BreakpointEventType;


//----------------------------------------------------------------------
/// Programming language type.
///
/// These enumerations use the same language enumerations as the DWARF
/// specification for ease of use and consistency.
//----------------------------------------------------------------------
typedef enum LanguageType
{
    eLanguageTypeUnknown         = 0x0000,   ///< Unknown or invalid language value.
    eLanguageTypeC89             = 0x0001,   ///< ISO C:1989.
    eLanguageTypeC               = 0x0002,   ///< Non-standardized C, such as K&R.
    eLanguageTypeAda83           = 0x0003,   ///< ISO Ada:1983.
    eLanguageTypeC_plus_plus     = 0x0004,   ///< ISO C++:1998.
    eLanguageTypeCobol74         = 0x0005,   ///< ISO Cobol:1974.
    eLanguageTypeCobol85         = 0x0006,   ///< ISO Cobol:1985.
    eLanguageTypeFortran77       = 0x0007,   ///< ISO Fortran 77.
    eLanguageTypeFortran90       = 0x0008,   ///< ISO Fortran 90.
    eLanguageTypePascal83        = 0x0009,   ///< ISO Pascal:1983.
    eLanguageTypeModula2         = 0x000a,   ///< ISO Modula-2:1996.
    eLanguageTypeJava            = 0x000b,   ///< Java.
    eLanguageTypeC99             = 0x000c,   ///< ISO C:1999.
    eLanguageTypeAda95           = 0x000d,   ///< ISO Ada:1995.
    eLanguageTypeFortran95       = 0x000e,   ///< ISO Fortran 95.
    eLanguageTypePLI             = 0x000f,   ///< ANSI PL/I:1976.
    eLanguageTypeObjC            = 0x0010,   ///< Objective-C.
    eLanguageTypeObjC_plus_plus  = 0x0011,   ///< Objective-C++.
    eLanguageTypeUPC             = 0x0012,   ///< Unified Parallel C.
    eLanguageTypeD               = 0x0013,   ///< D.
    eLanguageTypePython          = 0x0014    ///< Python.
} LanguageType;


typedef enum AccessType
{
    eAccessNone,
    eAccessPublic,
    eAccessPrivate,
    eAccessProtected,
    eAccessPackage
} AccessType;

//----------------------------------------------------------------------
/// Settable state variable types.
///
//----------------------------------------------------------------------

typedef enum SettableVariableType
{
    eSetVarTypeInt,
    eSetVarTypeBoolean,
    eSetVarTypeString,
    eSetVarTypeArray,
    eSetVarTypeDictionary,
    eSetVarTypeEnum,
    eSetVarTypeNone
} SettableVariableType;

typedef enum VarSetOperationType
{
    eVarSetOperationReplace,
    eVarSetOperationInsertBefore,
    eVarSetOperationInsertAfter,
    eVarSetOperationRemove,
    eVarSetOperationAppend,
    eVarSetOperationClear,
    eVarSetOperationAssign,
    eVarSetOperationInvalid
} VarSetOperationType;

//----------------------------------------------------------------------
/// Command argument types.
///
//----------------------------------------------------------------------

typedef enum CommandArgumentType
{
    eArgTypeAddress = 0,
    eArgTypeAliasName,
    eArgTypeAliasOptions,
    eArgTypeArchitecture,
    eArgTypeBoolean,
    eArgTypeBreakpointID,
    eArgTypeBreakpointIDRange,
    eArgTypeByteSize,
    eArgTypeCommandName,
    eArgTypeCount,
    eArgTypeEndAddress,
    eArgTypeExpression,
    eArgTypeExprFormat,
    eArgTypeFilename,
    eArgTypeFormat,
    eArgTypeFrameIndex,
    eArgTypeFullName,
    eArgTypeFunctionName,
    eArgTypeIndex,
    eArgTypeLineNum,
    eArgTypeLogCategory,
    eArgTypeLogChannel,
    eArgTypeMethod,
    eArgTypeName,
    eArgTypeNewPathPrefix,
    eArgTypeNumLines,
    eArgTypeNumberPerLine,
    eArgTypeOffset,
    eArgTypeOldPathPrefix,
    eArgTypeOneLiner,
    eArgTypePath, 
    eArgTypePid,
    eArgTypePlugin,
    eArgTypeProcessName,
    eArgTypeQueueName,
    eArgTypeRegisterName,
    eArgTypeRegularExpression,
    eArgTypeRunArgs,
    eArgTypeRunMode,
    eArgTypeScriptLang,
    eArgTypeSearchWord,
    eArgTypeSelector,
    eArgTypeSettingIndex,
    eArgTypeSettingKey,
    eArgTypeSettingPrefix,
    eArgTypeSettingVariableName,
    eArgTypeShlibName,
    eArgTypeSourceFile,
    eArgTypeSortOrder,
    eArgTypeStartAddress,
    eArgTypeSymbol,
    eArgTypeThreadID,
    eArgTypeThreadIndex,
    eArgTypeThreadName,
    eArgTypeUnixSignal,
    eArgTypeVarName,
    eArgTypeValue,
    eArgTypeWidth,
    eArgTypeNone,
    eArgTypeLastArg  // Always keep this entry as the last entry in this enumeration!!
} CommandArgumentType;

typedef enum ArgumentRepetitionType
{
    eArgRepeatPlain,            // Exactly one occurrence
    eArgRepeatOptional,         // At most one occurrence, but it's optional
    eArgRepeatPlus,             // One or more occurrences
    eArgRepeatStar,             // Zero or more occurrences
    eArgRepeatRange,            // Repetition of same argument, from 1 to n
    eArgRepeatPairPlain,        // A pair of arguments that must always go together ([arg-type arg-value]), occurs exactly once
    eArgRepeatPairOptional,     // A pair that occurs at most once (optional)
    eArgRepeatPairPlus,         // One or more occurrences of a pair
    eArgRepeatPairStar,         // Zero or more occurrences of a pair
    eArgRepeatPairRange,        // A pair that repeats from 1 to n
    eArgRepeatPairRangeOptional // A pair that repeats from 1 to n, but is optional
} ArgumentRepetitionType;

typedef enum SortOrder
{
    eSortOrderNone,
    eSortOrderByAddress,
    eSortOrderByName,
} SortOrder;


//----------------------------------------------------------------------
// Used in conjunction with Host::GetLLDBResource () to find files that
// are related to 
//----------------------------------------------------------------------
typedef enum PathType
{
    ePathTypeLLDBShlibDir,          // The directory where the lldb.so (unix) or LLDB mach-o file in LLDB.framework (MacOSX) exists
    ePathTypeSupportExecutableDir,  // Find LLDB support executable directory (debugserver, etc)
    ePathTypeHeaderDir,             // Find LLDB header file directory
    ePathTypePythonDir              // Find Python modules (PYTHONPATH) directory
} PathType;

} // namespace lldb


#endif  // LLDB_enumerations_h_
