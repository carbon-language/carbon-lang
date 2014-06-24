//===-- MICmdBase.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdBase.cpp
//
// Overview:	CMICmdBase implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueConst.h"
#include "MICmnLLDBDebugSessionInfo.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdBase constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdBase::CMICmdBase( void )
:	m_pSelfCreatorFn( nullptr )
,	m_rLLDBDebugSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() )
,	m_bHasResultRecordExtra( false )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdBase destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdBase::~CMICmdBase( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function.
// Type:	Overridden.
// Args:	None.
// Return:	SMICmdData & -	*this command's present status/data/information.
// Throws:	None.
//--
const SMICmdData & CMICmdBase::GetCmdData( void ) const 
{
	return m_cmdData; 
}
	
//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString & -	*this command's current error description.
//								Empty string indicates command status ok.
// Throws:	None.
//--
const CMIUtilString & CMICmdBase::GetErrorDescription( void ) const 
{
	return m_strCurrentErrDescription; 
}

//++ ------------------------------------------------------------------------------------
// Details:	The CMICmdFactory requires this function. Retrieve the command and argument
//			options description string.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString & -	Command decription.
// Throws:	None.
//--
const CMIUtilString & CMICmdBase::GetMiCmd( void ) const
{ 
	return m_strMiCmd; 
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. A command must be given working data and
//			provide data about its status or provide information to other objects.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdBase::SetCmdData( const SMICmdData & vCmdData ) 
{
	m_cmdData = vCmdData;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The command factory requires this function. The factory calls this function 
//			so it can obtain *this command's creation function.
// Type:	Overridden.
// Args:	None.
// Return:	CMICmdFactory::CmdCreatorFnPtr - Function pointer.
// Throws:	None.
//--
CMICmdFactory::CmdCreatorFnPtr CMICmdBase::GetCmdCreatorFn( void ) const
{
	return m_pSelfCreatorFn;
}

//++ ------------------------------------------------------------------------------------
// Details:	If a command is an event type (has callbacks registered with SBListener) it
//			needs to inform the Invoker that it has finished its work so that the
//			Invoker can tidy up and call the commands Acknowledge function (yes the
//			command itself could call the Acknowledge itself but not doing that way).
// Type:	Overridden.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
void CMICmdBase::CmdFinishedTellInvoker( void ) const
{
	CMICmdInvoker::Instance().CmdExecuteFinished( const_cast< CMICmdBase & >( *this ) );
}

//++ ------------------------------------------------------------------------------------
// Details:	Returns the final version of the MI result record built up in the command's
//			Acknowledge function. The one line text of MI result.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString &	- MI text version of the MI result record.
// Throws:	None.
//--
const CMIUtilString & CMICmdBase::GetMIResultRecord( void ) const
{
	return m_miResultRecord.GetString();
}
	
//++ ------------------------------------------------------------------------------------
// Details:	Retrieve from the command additional MI result to its 1 line response.
//			Because of using LLDB addtional 'fake'/hack output is sometimes required to
//			help the driver client operate i.e. Eclipse.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString &	- MI text version of the MI result record.
// Throws:	None.
//--
const CMIUtilString & CMICmdBase::GetMIResultRecordExtra( void ) const
{
	return m_miResultRecordExtra;
}

//++ ------------------------------------------------------------------------------------
// Details:	Hss *this command got additional MI result to its 1 line response.
//			Because of using LLDB addtional 'fake'/hack output is sometimes required to
//			help the driver client operate i.e. Eclipse.
// Type:	Overridden.
// Args:	None.
// Return:	bool	- True = Yes have additional MI output, false = no nothing extra.
// Throws:	None.
//--
bool CMICmdBase::HasMIResultRecordExtra( void ) const
{
	return m_bHasResultRecordExtra;
}

//++ ------------------------------------------------------------------------------------
// Details:	Short cut function to enter error information into the command's metadata
//			object and set the command's error status.
// Type:	Method.
// Args:	rErrMsg	- (R) Error description.
// Return:	None.
// Throws:	None.
//--
void CMICmdBase::SetError( const CMIUtilString & rErrMsg )
{
	m_cmdData.bCmdValid = false;
	m_cmdData.strErrorDescription = rErrMsg;
	m_cmdData.bCmdExecutedSuccessfully = false;
	
	const CMICmnMIValueResult valueResult( "msg", CMICmnMIValueConst( rErrMsg ) );
	const CMICmnMIResultRecord miResultRecord( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, valueResult );
	m_miResultRecord = miResultRecord;
	m_cmdData.strMiCmdResultRecord = miResultRecord.GetString();
}

//++ ------------------------------------------------------------------------------------
// Details:	Ask a command to provide its unique identifier.
// Type:	Method.
// Args:	A unique identifier for this command class.
// Return:	None.
// Throws:	None.
//--
MIuint CMICmdBase::GetGUID( void )
{
	MIuint64 vptr = reinterpret_cast< MIuint64 >( this );
	MIuint id  = (vptr      ) & 0xFFFFFFFF;
			id ^= (vptr >> 32) & 0xFFFFFFFF;

	return id;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdBase::ParseArgs( void ) 
{ 
	// Do nothing - override to implement

	return MIstatus::success; 
}
	
