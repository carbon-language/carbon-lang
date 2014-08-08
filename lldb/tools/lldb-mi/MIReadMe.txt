========================================================================
    The MI Driver - LLDB Machine Interface V2 (MI)  Project Overview  
	24/07/2014
========================================================================

The MI Driver is a stand alone executable that either be used via a 
client i.e. Eclipse or directly from the command line. 

All the files in this directory are required to build the MI executable.
The executable is intended to compile and work on the following platforms:

	Windows (Vista or newer) (Compiler: Visual C++ 12)
	LINUX (Compiler: gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1)
	OSX (Not tested)

THe MI Driver has two modes of operation; LLDB and MI. The MI Driver (CMIDriver)
which operates the MI mode is a driver in its own right to work alongside
the LLDB driver (driver .h/.cpp). Only one is operatational at a time depending
on the options entered on the command line. The MI Driver reads MI inputs and
outputs MI responses to be interpreted by a client i.e. Eclipse. 
Should the MI Driver not understand an instruction it could be passed to the 
LLDB driver for interpretation (MI Driver build configuration dependant). Should 
the LLDB driver mode be chosen then it the MI Driver will behave as the normal 
LLDB driver.

For help information on using the MI driver type at the command line:

	lldb-mi --interpreter --help

A blog about the MI Driver is available on CodePlay's website 
http://www.codeplay.com/portal/. 

The MI Driver produces a MILog.txt file which records the actions of the MI 
Driver when in the MI mode only.

Note any command or text sent to the MI Driver in MI mode that is not a command 
registered in the MI Driver's Command Factory will be rejected given an error.

The MILogfile.txt keeps a history of the MI Driver's activity for one session
only. It is used to aid the debugging of the MI Driver in MI mode only. As well 
as recorded commands that are recognised by the MI Driver it also gives warnings
about command's which do not support certain argument or options.  

All the files prefix with MI are specifically for the MI driver code only.
Non prefixed code is the original LLDB driver which has been left untouched
as much as possible. This allows the LLDB driver code to develop 
independently and make future integration more straight forward. 

File MIDriverMain.cpp contains the executables main() function and some 
common global functions common to the two drivers.

=========================================================================
Current limitations:
1. Commands implemented likely not to have all their arguments supported
2. The MI Driver has only been tested with Eclipse Juno with an in-house
   plugin
3. Local target has been implemented but not tested
4. The MI Driver has been designed primarily to work in a 'remote-target'
   mode only. The MI Driver does not currently except arguments beyond
   those described above.
5. The MI Driver does not accept as arguments an executable to create a
   target instance.
6. Not all MI commands have been implemented. See section MI Driver
   commands for those that have been fully or partially implemented (not
   indicated - see command class).   
7. Not necessarily a limitation but the MI Driver is used with Codeplay's
   own Eclipse plugin (not supplied) which has allowed more control over
   the interaction with the MI Driver between Eclipse.

=========================================================================
Versions:	
1.0.0.1 First version from scratch 28/1/2014 to 28/3/2014. 
		MI working alpha. MI framework not complete.
1.0.0.2 First deliverable to client 7/3/2014. 
		MI working beta. MI framework not complete. 
1.0.0.3	Code refactor tidy. Release to community for evaluation 
		7/5/2014. 
		MI working beta - code refactored and tidied. MI framework
		complete. Just missing commands (which may still require
		changes).
1.0.0.4	Post release to community for evaluation 7/5/2014.
	1. MI command token now optional
	2. MI command token is now fixed length
	3. New commands added see section "MI commands implemented are:"
	4. Able to debug a local target as well as remote target
	5. MI Driver now sends to the client "(gdb)" + '\n' on 
	   initialising
	6. Improve coverage of parsing and fix command argument parsing
	7. Fix bug with stdin thinking there was no input when there was which
	   caused communication between the client and the MI Driver to halt
	   due to internal buffering, we now keep track of it ourself.
	8. Code comment fixes and additions. Code style fixes. 
	9. MI Driver now on receiving Ctrl-C (SIGINT) when the client pauses
	   an inferior program does not quit but continues operating.
	10.Fix commands "var-update", "var-evaluate-expression" to which did
	   not send back information to correctly update arrays and structures.
	11.Commands "Not implemented" are now not registered to the command
       factory except for GDB command "thread". Commands not registered
	   with the command factory produce MI error message "...not in 
	   Command Factory". (Removed from command section in this readme.txt)
1.0.0.5 Second deliverable to client 16/6/2014.
1.0.0.6 Post release of second deliverable to client 16/6/2014.
		Released to the community 24/6/2014.
	1. The MI Driver has a new option --noLog. If present the MI Driver 
	   does not output progress or status messages to it's log file.
	2. Moved OS specific handling of the stdin stream to their own class
	   implementations so any changes to one handler will not affect
	   another OS's handler.
	3. The session data/information map for sharing data between commands
	   now uses a variant object which enables objects of different types
	   to be stored instead of previously just text information.
	4. Debug session var object create, update and retrieve efficiency
	   improved by using a map type container.
	5. Re-enable the MI Driver's command line option --interpreter (see
	   --help). Up until now it was implementented but not enforced, it 
	   was always the MI Driver interpreter.
	6. Re-enable the compilation of the original LLDB driver code into
	   the MI Driver's code. See MICmnConfig.h for build configuration.
1.0.0.7	Post release to community. Delivered to client 30/6/2014.
	1. Fix MI Driver's output of "(gdb)" appearing when running in LLDB
	   mode (no --interpreter argument)'
	2. Fix command "interpret-exec" to allow commands to be entered
	   directly in the IDE console.
1.0.0.8 Post release to client. Delivered to client 29/07/2014	
	1. Fix command "break-insert" argument -f not accepting file paths 
	   as a string. Looked like the MI Driver was not accepting LINUX
	   style file paths in the Windows version and vice versa.
	2. Fix command "stack-list-arguments" handling only the current 
	   stack frame. Eclipse now shows variables for all frames.
	3. Fix and improve MI response for sending back information on
	   stack local variables, stack arguments and stack frame selection.
	4. Fix recursive crash when asking to gather information on link
	   lists.
	5. Fix MI Driver's Log date and time field.
	6. Fix MI response return from event 'StopReason' and 'Breakpoint-
	   hit'.
	7. Fix command "environment-cd" to handle paths with spaces in the
	   path.
	8. Fix not displaying backtrace (stack) variable information when 
	   choosing frames other than the current frame.
	9. Fix command "data-evaluate-expression" to be able to handle 
	   valid SBValue objects but have no value object name. Fix same 
	   command to handle expressions surround by string format inserted 
	   quotes.
	10.Fix command "break-insert" to handle file location that is
	   surrounded by quotes.
	11.For commands "var-create" and "data-evaluate-expression" improve 
	   variable type handling for quoted expressions.
	12.Implement command "inferior-tty-set". It just responds with 
	   "^Done".
	13.Improve the MI Driver's help description.
	14.Fix file name paths that contained '.', '-' and '_' in the path 
	   as being treated as invalid.
	15.Fix trying to interpret escapse character text as an errorous
	   command.
1.0.0.9 Post release to client.	

=========================================================================
MI Driver Commands
MI commands below are written to work for Eclipse Juno 7.4. If may be 
one are more commands required by other IDEs are missing or do not 
support all arguments or options. Additionally some commands may handle
additional arguments or options not documented here 
https://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Data-Manipulation.html#GDB_002fMI-Data-Manipulation.
The implemented commands are:
CMICmdCmdBreakAfter
CMICmdCmdBreakCondition
CMICmdCmdBreakDelete
CMICmdCmdBreakDisable
CMICmdCmdBreakEnable
CMICmdCmdBreakInsert
CMICmdCmdDataEvaluateExpression
CMICmdCmdDataDisassemble
CMICmdCmdDataListRegisterChanged
CMICmdCmdDataListRegisterNames
CMICmdCmdDataListRegisterValues
CMICmdCmdDataReadMemory
CMICmdCmdDataReadMemoryBytes
CMICmdCmdDataWriteMemory
CMICmdCmdEnablePrettyPrinting
CMICmdCmdEnvironmentCd
CMICmdCmdExecContinue
CMICmdCmdExecFinish
CMICmdCmdExecInterrupt
CMICmdCmdExecNext
CMICmdCmdExecNextInstruction
CMICmdCmdExecRun
CMICmdCmdExecStep
CMICmdCmdExecStepInstruction
CMICmdCmdFileExecAndSymbols
CMICmdCmdGdbExit
CMICmdCmdGdbInfo
CMICmdCmdGdbSet	
CMICmdCmdGdbSet - solib-search-path option
CMICmdCmdInferiorTtySet (not functionally implemented)
CMICmdCmdInterpreterExec
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
CMICmdCmdTraceStatus (not functionally implemented)
CMICmdCmdVarAssign
CMICmdCmdVarCreate
CMICmdCmdVarDelete
CMICmdCmdVarEvaluateExpression
CMICmdCmdVarInfoPathExpression
CMICmdCmdVarListChildren
CMICmdCmdVarSetFormat
CMICmdCmdVarShowAttributes
CMICmdCmdVarUpdate

=========================================================================
The MI Driver build configuration:
MICmnConfig.h defines various preprocessor build options i.e. enable
LLDB driver fall through (Driver.h/.cpp) should MI Driver not recognise a 
command (option not fully implemented - may be removed in the future).

=========================================================================
Code standard, documentation and code style scope:
The coding style and coding documentation scope covers all MI prefixed
files and where MI code is implemented in the LLDB driver files. Should 
you wish to make improvements or fixes to the MI code (which is encouraged)
please DO comment your code in the style already applied. The same applies
to the coding style. Class names should also follow this lead and ideally
should be one class per file (.h/.cpp). Class interface files (.h) should 
not contain any implementation code unless there is a performance issue or
templated functions. You get the idea, look around the existing code and
follow by example :)

Where code comment or documentation is wrong or can be improved to help
others then it is strongly encouraged you DO improve the documentation.

=========================================================================
MI Driver license:
The MI Driver code is under the University of Illinois Open Source License
agreement. Submitted by Codeplay Ltd UK.

Source code found at: llvm/tools/lldb/tools/lldb-mi.

=========================================================================
The MI Driver uses the following libraries:
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


