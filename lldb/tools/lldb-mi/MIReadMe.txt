========================================================================
    LLDB Machine Interface V2 (MI)  Project Overview  
	28/01/2014
========================================================================

All the files in this directory are required to build the MI executable.
The executable is intended to compile and work on the following platforms:

	Windows (Vista or newer)
	LINUX
	OSX

The MI driver (CMIDriver) is a driver in its own right to work alongside
the LLDB driver (driver .h/.cpp). Only one is operated at a time depending
on the options entered on the command line. The MI driver inputs and
outputs MI (GDB instruction) to be interpreted by a client i.e. Eclipse. 
Should MI not understand an instruction it can be passed to the LLDB driver for 
interpretation (the MI stub is on top of LLDB driver not GDB)(build
configuration dependant). Should the LLDB driver be chosen then it the 
MI driver will behave as a normal LLDB driver code MI.

Type --help for instruction on using the MI driver. MI produces a MILog.txt file
which records the actions of the MI driver (only) found in the directory
of the lldbMI executable.

All the files prefix with MI are specifically for the MI driver code only.
Non prefixed code is the original LLDB driver which has been left untouched
as much as possible. This allows the LLDB driver code to develop 
independently and make future integration more straight forward. 

File MIDriverMain.cpp contains the executables main() function and some 
common global functions common to the two drivers.

=========================================================================
Versions:	
1.0.0.1 	First version from scratch 28/1/2014 to 28/3/2014. 
		MI working alpha. MI framework not complete.
1.0.0.2 	7/3/2014. 
		MI working beta. MI framework not complete. 
1.0.0.3	Code refactor tidy. Release to community for evaluation 
		7/5/2014. 
		MI working beta - code refactored and tidied. MI framework
		complete. Just missing commands (which may still require
		changes).

=========================================================================
MI commands implemented are:
CMICmdCmdBreakDelete
CMICmdCmdBreakInsert
CMICmdCmdDataEvaluateExpression
CMICmdCmdEnablePrettyPrinting
CMICmdCmdEnvironmentCd
CMICmdCmdExecContinue
CMICmdCmdExecFinish
CMICmdCmdExecNext
CMICmdCmdExecNextInstruction
CMICmdCmdExecRun
CMICmdCmdExecStep
CMICmdCmdExecStepInstruction
CMICmdCmdFileExecAndSymbols
CMICmdCmdGdbExit
CMICmdCmdGdbSet
CMICmdCmdListThreadGroups
CMICmdCmdSource
CMICmdCmdStackInfoDepth
CMICmdCmdStackListArguments
CMICmdCmdStackListFrames
CMICmdCmdStackListLocals
CMICmdCmdSupportInfoMiCmdQuery
CMICmdCmdSupportListFeatures
CMICmdCmdTargetSelect
CMICmdCmdThread
CMICmdCmdThreadInfo
CMICmdCmdTraceStatus
CMICmdCmdVarAssign
CMICmdCmdVarCreate
CMICmdCmdVarDelete
CMICmdCmdVarEvaluateExpression
CMICmdCmdVarInfoPathExpression
CMICmdCmdVarListChildren
CMICmdCmdVarSetFormat
CMICmdCmdVarUpdate

=========================================================================
MI build configuration:
MICmnConfig.h defines various preprocessor build options i.e. enable
LLDB fall through should MI interpretor not recognise a command (option
not fully implemented - may be removed in the future).

=========================================================================
MI uses the following libraries:
Standard Template library
	Thread
	Containers
	String
	File
	Time
LLDB public API
OS specific
	OS error reporting windows
	OS error handling OSX (not implemented)
	OS error handling LINUX (not implemented)


