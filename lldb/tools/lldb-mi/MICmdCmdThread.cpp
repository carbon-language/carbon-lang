//===-- MICmdCmdThread.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdThread.cpp
//
// Overview:	CMICmdCmdThread					implementation.
//				CMICmdCmdThreadInfo				implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third Party Headers:
#include <lldb/API/SBBreakpointLocation.h>
#include <lldb/API/SBThread.h>

// In-house headers:
#include "MICmnConfig.h"
#include "MICmdCmdThread.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmdArgContext.h"
#include "MICmdArgValFile.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdThread constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdThread::CMICmdCmdThread( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "thread";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdThread::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdThread destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdThread::~CMICmdCmdThread( void )
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
bool CMICmdCmdThread::Execute( void )
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
bool CMICmdCmdThread::Acknowledge( void )
{
	const CMICmnMIValueConst miValueConst( MIRSRC( IDS_WORD_NOT_IMPLEMENTED ) );
	const CMICmnMIValueResult miValueResult( "msg", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
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
CMICmdBase * CMICmdCmdThread::CreateSelf( void )
{
	return new CMICmdCmdThread();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdThreadInfo constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdThreadInfo::CMICmdCmdThreadInfo( void )
:	m_bSingleThread( false )
,	m_bThreadInvalid( true )
,	m_constStrArgNamedThreadId( "thread-id" )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "thread-info";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdThreadInfo::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdThreadInfo destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdThreadInfo::~CMICmdCmdThreadInfo( void )
{
	m_vecMIValueTuple.clear();
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
bool CMICmdCmdThreadInfo::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgNamedThreadId, false, true )) );
	CMICmdArgContext argCntxt( m_cmdData.strMiCmdOption );
	if( bOk && !m_setCmdArgs.Validate( m_cmdData.strMiCmd, argCntxt ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_ARGS ), m_cmdData.strMiCmd.c_str(), m_setCmdArgs.GetErrorDescription().c_str() ) );
		return MIstatus::failure;
	}
	
	return bOk;
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
bool CMICmdCmdThreadInfo::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgThreadId, Number, m_constStrArgNamedThreadId );
	MIuint nThreadId = 0;
	if( pArgThreadId->GetFound() && pArgThreadId->GetValid() )
	{
		m_bSingleThread  = true;
		nThreadId = static_cast< MIuint >( pArgThreadId->GetValue() );
	}

	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	lldb::SBThread thread = rProcess.GetSelectedThread();
	
	if( m_bSingleThread )
	{
		thread = rProcess.GetThreadByIndexID( nThreadId );
		m_bThreadInvalid = thread.IsValid();
		if( !m_bThreadInvalid )
			return MIstatus::success;

		CMICmnMIValueTuple miTuple;
		if( !rSessionInfo.MIResponseFormThreadInfo( m_cmdData, thread, miTuple ) )
			return MIstatus::failure;

		m_miValueTupleThread = miTuple;

		return MIstatus::success;
	}

	// Multiple threads
	m_vecMIValueTuple.clear();
	const MIuint nThreads = rProcess.GetNumThreads();
	for( MIuint i = 0; i < nThreads; i++ )
	{
		lldb::SBThread thread = rProcess.GetThreadAtIndex( i );
		if( thread.IsValid() )
		{
			CMICmnMIValueTuple miTuple;
			if( !rSessionInfo.MIResponseFormThreadInfo( m_cmdData, thread, miTuple ) )
				return MIstatus::failure;

			m_vecMIValueTuple.push_back( miTuple );
		}
	}
	
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
bool CMICmdCmdThreadInfo::Acknowledge( void )
{
	if( m_bSingleThread )
	{
		if( !m_bThreadInvalid )
		{
			const CMICmnMIValueConst miValueConst( "invalid thread id" );
			const CMICmnMIValueResult miValueResult( "msg", miValueConst );
			const CMICmnMIResultRecord miRecordResult(  m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
			m_miResultRecord = miRecordResult;
			return MIstatus::success;
		}

		// MI print "%s^done,threads=[{id=\"%d\",target-id=\"%s\",frame={},state=\"%s\"}]
		const CMICmnMIValueList miValueList( m_miValueTupleThread );
		const CMICmnMIValueResult miValueResult( "threads", miValueList );
		const CMICmnMIResultRecord miRecordResult(  m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
		m_miResultRecord = miRecordResult;
		return MIstatus::success;	
	}

	// Build up a list of thread information from tuples
	VecMIValueTuple_t::const_iterator it = m_vecMIValueTuple.begin();
	if( it == m_vecMIValueTuple.end() )
	{
		const CMICmnMIValueConst miValueConst( "[]" );
		const CMICmnMIValueResult miValueResult( "threads", miValueConst );
		const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
		m_miResultRecord = miRecordResult;
		return MIstatus::success;
	}
	CMICmnMIValueList miValueList( *it );
	++it;
	while( it != m_vecMIValueTuple.end() )
	{
		const CMICmnMIValueTuple & rTuple( *it );
		miValueList.Add( rTuple );

		// Next
		++it;
	}

	const CMICmnMIValueResult miValueResult( "threads", miValueList );
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
CMICmdBase * CMICmdCmdThreadInfo::CreateSelf( void )
{
	return new CMICmdCmdThreadInfo();
}
