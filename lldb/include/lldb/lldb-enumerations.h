//===-- lldb-enumerations.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_enumerations_h_
#define LLDB_lldb_enumerations_h_

namespace lldb {

    //----------------------------------------------------------------------
    // Process and Thread States
    //----------------------------------------------------------------------
    typedef enum StateType
    {
        eStateInvalid = 0,
        eStateUnloaded,     ///< Process is object is valid, but not currently loaded
        eStateConnected,    ///< Process is connected to remote debug services, but not launched or attached to anything yet
        eStateAttaching,    ///< Process is currently trying to attach
        eStateLaunching,    ///< Process is in the process of launching
        eStateStopped,      ///< Process or thread is stopped and can be examined.
        eStateRunning,      ///< Process or thread is running and can't be examined.
        eStateStepping,     ///< Process or thread is in the process of stepping and can not be examined.
        eStateCrashed,      ///< Process or thread has crashed and can be examined.
        eStateDetached,     ///< Process has been detached and can't be examined.
        eStateExited,       ///< Process has exited and can't be examined.
        eStateSuspended     ///< Process or thread is in a suspended state as far
                            ///< as the debugger is concerned while other processes
                            ///< or threads get the chance to run.
    } StateType;

    //----------------------------------------------------------------------
    // Launch Flags
    //----------------------------------------------------------------------
    typedef enum LaunchFlags
    {
        eLaunchFlagNone         = 0u,
        eLaunchFlagExec         = (1u << 0),  ///< Exec when launching and turn the calling process into a new process
        eLaunchFlagDebug        = (1u << 1),  ///< Stop as soon as the process launches to allow the process to be debugged
        eLaunchFlagStopAtEntry  = (1u << 2),  ///< Stop at the program entry point instead of auto-continuing when launching or attaching at entry point
        eLaunchFlagDisableASLR  = (1u << 3),  ///< Disable Address Space Layout Randomization
        eLaunchFlagDisableSTDIO = (1u << 4),  ///< Disable stdio for inferior process (e.g. for a GUI app)
        eLaunchFlagLaunchInTTY  = (1u << 5)   ///< Launch the process in a new TTY if supported by the host 
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
    // Byte ordering definitions
    //----------------------------------------------------------------------
    typedef enum ByteOrder
    {
        eByteOrderInvalid   = 0,
        eByteOrderBig       = 1,
        eByteOrderPDP       = 2,
        eByteOrderLittle    = 4
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
        eFormatComplex,         // Floating point complex type
        eFormatComplexFloat = eFormatComplex,
        eFormatCString,         // NULL terminated C strings
        eFormatDecimal,
        eFormatEnum,
        eFormatHex,
        eFormatFloat,
        eFormatOctal,
        eFormatOSType,          // OS character codes encoded into an integer 'PICT' 'text' etc...
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
        eFormatVectorOfUInt128,
        eFormatComplexInteger   // Integer complex type

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
        eValueTypeConstResult       = 7     // constant result variables
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

    typedef enum InputReaderAction
    {
        eInputReaderActivate,   // reader is newly pushed onto the reader stack 
        eInputReaderAsynchronousOutputWritten, // an async output event occurred; the reader may want to do something
        eInputReaderReactivate, // reader is on top of the stack again after another reader was popped off 
        eInputReaderDeactivate, // another reader was pushed on the stack 
        eInputReaderGotToken,   // reader got one of its tokens (granularity)
        eInputReaderInterrupt,  // reader received an interrupt signal (probably from a control-c)
        eInputReaderEndOfFile,  // reader received an EOF char (probably from a control-d)
        eInputReaderDone        // reader was just popped off the stack and is done
    } InputReaderAction;

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

    typedef enum DynamicValueType
    {
        eNoDynamicValues = 0,
        eDynamicCanRunTarget    = 1,
        eDynamicDontRunTarget   = 2
    } DynamicValueType;
    
    typedef enum AccessType
    {
        eAccessNone,
        eAccessPublic,
        eAccessPrivate,
        eAccessProtected,
        eAccessPackage
    } AccessType;

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
        eArgTypeClassName,
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
        eArgTypePlatform,
        eArgTypeLastArg  // Always keep this entry as the last entry in this enumeration!!
    } CommandArgumentType;

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

    typedef enum EmulateInstructionOptions
    {
        eEmulateInstructionOptionNone               = (0u),
        eEmulateInstructionOptionAutoAdvancePC      = (1u << 0),
        eEmulateInstructionOptionIgnoreConditions   = (1u << 1)
    } EmulateInstructionOptions;

} // namespace lldb


#endif  // LLDB_lldb_enumerations_h_
