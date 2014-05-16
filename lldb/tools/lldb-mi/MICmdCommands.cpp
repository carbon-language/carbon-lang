//===-- MICmdCommands.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCommands.cpp
//
// Overview:	MI command are registered with the MI command factory.
//
//				To implement new MI commands derive a new command class from the command base 
//				class. To enable the new command for interpretation add the new command class
//				to the command factory. The files of relevance are:
//					MICmdCommands.cpp
//					MICmdBase.h / .cpp
//					MICmdCmd.h / .cpp
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmdCommands.h"
#include "MICmdFactory.h"
#include "MICmdCmd.h"
#include "MICmdCmdBreak.h"
#include "MICmdCmdData.h"
#include "MICmdCmdEnviro.h"
#include "MICmdCmdExec.h"
#include "MICmdCmdFile.h"
#include "MICmdCmdMiscellanous.h"
#include "MICmdCmdStack.h"
#include "MICmdCmdSupportInfo.h"
#include "MICmdCmdSupportList.h"
#include "MICmdCmdTarget.h"
#include "MICmdCmdThread.h"
#include "MICmdCmdTrace.h"
#include "MICmdCmdVar.h"

namespace MICmnCommands
{
	template< typename T >
	static bool Register( void );
}

//++ ------------------------------------------------------------------------------------
// Details:	Command to command factory registration function. 
// Type:	Template function.
// Args:	None.
// Return:	bool  - True = yes command is registered, false = command failed to register.
// Throws:	None.
//--
template< typename T >											
static bool MICmnCommands::Register( void  )								
{																	
	static CMICmdFactory & rCmdFactory = CMICmdFactory::Instance();
	const CMIUtilString strMiCmd = T().GetMiCmd();				
	CMICmdFactory::CmdCreatorFnPtr fn = T().GetCmdCreatorFn();	
	return rCmdFactory.CmdRegister( strMiCmd, fn );				
}																	

//++ ------------------------------------------------------------------------------------
// Details:	Register commands with MI command factory
// Type:	Function.
// Args:	None.
// Return:	bool  - True = yes all commands are registered, 
//					false = one or more commands failed to register.
// Throws:	None.
//--
bool MICmnCommands::RegisterAll( void )
{
	bool bOk = MIstatus::success;

	bOk &= Register< CMICmdCmdSupportInfoMiCmdQuery >();
	bOk &= Register< CMICmdCmdSupportListFeatures >();
	bOk &= Register< CMICmdCmdEnvironmentCd >();
	bOk &= Register< CMICmdCmdGdbSet >();
	bOk &= Register< CMICmdCmdEnablePrettyPrinting >();
	bOk &= Register< CMICmdCmdGdbExit >();
	bOk &= Register< CMICmdCmdSource >();
	bOk &= Register< CMICmdCmdFileExecAndSymbols >();
	bOk &= Register< CMICmdCmdTargetSelect >();
	bOk &= Register< CMICmdCmdListThreadGroups >();
	bOk &= Register< CMICmdCmdExecRun >();
	bOk &= Register< CMICmdCmdExecContinue >();
	bOk &= Register< CMICmdCmdTraceStatus >();
	bOk &= Register< CMICmdCmdThreadInfo >();
	bOk &= Register< CMICmdCmdBreakInsert >();
	bOk &= Register< CMICmdCmdBreakDelete >();
	bOk &= Register< CMICmdCmdThread >();
	bOk &= Register< CMICmdCmdStackInfoDepth >();
	bOk &= Register< CMICmdCmdStackListFrames >();
	bOk &= Register< CMICmdCmdStackListArguments >();
	bOk &= Register< CMICmdCmdStackListLocals >();
	bOk &= Register< CMICmdCmdVarCreate >();
	bOk &= Register< CMICmdCmdExecNext >();
	bOk &= Register< CMICmdCmdExecStep >();
	bOk &= Register< CMICmdCmdExecNextInstruction >();
	bOk &= Register< CMICmdCmdExecStepInstruction >();
	bOk &= Register< CMICmdCmdExecFinish >();
	bOk &= Register< CMICmdCmdVarUpdate >();
	bOk &= Register< CMICmdCmdVarDelete >();
	bOk &= Register< CMICmdCmdVarAssign >();
	bOk &= Register< CMICmdCmdVarSetFormat >();
	bOk &= Register< CMICmdCmdVarListChildren >();
	bOk &= Register< CMICmdCmdVarEvaluateExpression >();
	bOk &= Register< CMICmdCmdVarInfoPathExpression >();
	bOk &= Register< CMICmdCmdDataEvaluateExpression >();

	return bOk;
}