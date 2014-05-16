//===-- MICmdCmd.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmd.cpp
//
// Overview:	CMICmdCmdEnablePrettyPrinting	implementation.
//				CMICmdCmdSource					implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third Party Headers:
#include <lldb/API/SBBreakpointLocation.h>
#include <lldb/API/SBCommandInterpreter.h>
#include <lldb/API/SBStream.h>
#include <lldb/API/SBThread.h>
#include <lldb/API/SBTypeFormat.h>
#include <limits.h>

// In-house headers:
#include "MICmnConfig.h"
#include "MICmdCmd.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnMIOutOfBandRecord.h"
#include "MICmnResources.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MIDriverBase.h"
#include "MIUtilDebug.h"
#include "MIDriver.h"
#include "MIUtilFileStd.h"
#include "MICmnLLDBProxySBValue.h"
#include "MICmdArgContext.h"
#include "MICmdArgValFile.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdEnablePrettyPrinting constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdEnablePrettyPrinting::CMICmdCmdEnablePrettyPrinting( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "enable-pretty-printing";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdEnablePrettyPrinting::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdEnablePrettyPrinting destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdEnablePrettyPrinting::~CMICmdCmdEnablePrettyPrinting( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdEnablePrettyPrinting::Execute( void )
{
	// Do nothing
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdEnablePrettyPrinting::Acknowledge( void )
{
	const CMICmnMIValueConst miValueConst( "0" );
	const CMICmnMIValueResult miValueResult( "supported", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
	m_miResultRecord = miRecordResult;
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this commmand. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdEnablePrettyPrinting::CreateSelf( void )
{
	return new CMICmdCmdEnablePrettyPrinting();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdSource constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdSource::CMICmdCmdSource( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "source";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdSource::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdSource destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdSource::~CMICmdCmdSource( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdSource::Execute( void )
{
	// Do nothing
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdSource::Acknowledge( void )
{
	const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done );
	m_miResultRecord = miRecordResult;
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this commmand. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdSource::CreateSelf( void )
{
	return new CMICmdCmdSource();
}