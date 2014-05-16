//===-- MICmdCmdMiscellanous.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdMiscellanous.cpp
//
// Overview:	CMICmdCmdGdbSet					implementation.
//				CMICmdCmdGdbExit				implementation.
//				CMICmdCmdListThreadGroups		implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third Party Headers:
#include <lldb/API/SBThread.h>

// In-house headers:
#include "MICmnConfig.h"
#include "MICmdCmdMiscellanous.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnMIOutOfBandRecord.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MIDriverBase.h"
#include "MICmdArgContext.h"
#include "MICmdArgValFile.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdGdbSet constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdGdbSet::CMICmdCmdGdbSet( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "gdb-set";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdGdbSet::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdGdbSet destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdGdbSet::~CMICmdCmdGdbSet( void )
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
bool CMICmdCmdGdbSet::Execute( void )
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
bool CMICmdCmdGdbSet::Acknowledge( void )
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
CMICmdBase * CMICmdCmdGdbSet::CreateSelf( void )
{
	return new CMICmdCmdGdbSet();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdGdbExit constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdGdbExit::CMICmdCmdGdbExit( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "gdb-exit";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdGdbExit::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdGdbExit destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdGdbExit::~CMICmdCmdGdbExit( void )
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
bool CMICmdCmdGdbExit::Execute( void )
{
	CMICmnLLDBDebugger::Instance().GetDriver().SetExitApplicationFlag();
	const lldb::SBError sbErr = m_rLLDBDebugSessionInfo.m_lldbProcess.Detach();
	// Do not check for sbErr.Fail() here, m_lldbProcess is likely !IsValid()
		
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
bool CMICmdCmdGdbExit::Acknowledge( void )
{
	const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Exit );
	m_miResultRecord = miRecordResult;

	// Prod the client i.e. Eclipse with out-of-band results to help it 'continue' because it is using LLDB debugger
	// Give the client '=thread-group-exited,id="i1"'
	m_bHasResultRecordExtra = true;
	const CMICmnMIValueConst miValueConst2( "i1" );
	const CMICmnMIValueResult miValueResult2( "id", miValueConst2 );
	const CMICmnMIOutOfBandRecord miOutOfBand( CMICmnMIOutOfBandRecord::eOutOfBand_ThreadGroupExited, miValueResult2 );
	m_miResultRecordExtra = miOutOfBand.GetString();

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
CMICmdBase * CMICmdCmdGdbExit::CreateSelf( void )
{
	return new CMICmdCmdGdbExit();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdListThreadGroups constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdListThreadGroups::CMICmdCmdListThreadGroups( void )
:	m_bIsI1( false )
,	m_bHaveArgOption( false )
,	m_bHaveArgRecurse( false )
,	m_constStrArgNamedAvailable( "available" )
,	m_constStrArgNamedRecurse( "recurse" )
,	m_constStrArgNamedGroup( "group" )
,	m_constStrArgNamedThreadGroup( "i1" )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "list-thread-groups";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdListThreadGroups::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdListThreadGroups destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdListThreadGroups::~CMICmdCmdListThreadGroups( void )
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
bool CMICmdCmdListThreadGroups::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgNamedAvailable, false, true )) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgNamedRecurse, false, true, CMICmdArgValListBase::eArgValType_Number, 1 )) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValListOfN( m_constStrArgNamedGroup, false, true, CMICmdArgValListBase::eArgValType_Number )) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValThreadGrp( m_constStrArgNamedThreadGroup, false, true )) );
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
//			Synopis: -list-thread-groups [ --available ] [ --recurse 1 ] [ group ... ]
//			This command does not follow the MI documentation exactly. Has an extra
//			argument "i1" to handle. 
//			Ref: http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Miscellaneous-Commands.html#GDB_002fMI-Miscellaneous-Commands
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdListThreadGroups::Execute( void )
{
	if( m_setCmdArgs.IsArgContextEmpty() )
		// No options so "top level thread groups"
		return MIstatus::success;

	CMICMDBASE_GETOPTION( pArgAvailable, OptionLong, m_constStrArgNamedAvailable );
	CMICMDBASE_GETOPTION( pArgRecurse, OptionLong, m_constStrArgNamedRecurse );
	CMICMDBASE_GETOPTION( pArgGroup, ListOfN, m_constStrArgNamedGroup );
	CMICMDBASE_GETOPTION( pArgThreadGroup, ThreadGrp, m_constStrArgNamedThreadGroup );
	
	// Demo of how to get the value of long argument --recurse's option of 1 "--recurse 1"
	const CMICmdArgValOptionLong::VecArgObjPtr_t & rVecOptions( pArgRecurse->GetExpectedOptions() );
	const CMICmdArgValNumber * pRecurseDepthOption = (rVecOptions.size() > 0) ? static_cast< CMICmdArgValNumber * >( rVecOptions[ 1 ] ) : nullptr;
	const MIuint nRecurseDepth = (pRecurseDepthOption != nullptr) ? pRecurseDepthOption->GetValue() : 0;
	
	// Demo of how to get List of N numbers (the Group argument not implement for this command (yet))
	const CMICmdArgValListOfN::VecArgObjPtr_t & rVecGroupId( pArgGroup->GetValue() );
	CMICmdArgValListOfN::VecArgObjPtr_t::const_iterator it = rVecGroupId.begin();
	while( it != rVecGroupId.end() )
	{
		const CMICmdArgValNumber * pOption = static_cast< CMICmdArgValNumber * >( *it );
		const MIuint nGrpId = pOption->GetValue();

		// Next
		++it;
	}

	// Got some options so "threads"
	if( pArgAvailable->GetFound() )
	{
		if( pArgRecurse->GetFound() )
		{
			m_bHaveArgRecurse = true;
			return MIstatus::success;
		}

		m_bHaveArgOption = true;
		return MIstatus::success;
	}
	// "i1" as first argument (pos 0 of possible arg)
	if( !pArgThreadGroup->GetFound() )
		return MIstatus::success;
	m_bIsI1 = true;

	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	
	// Note do not check for rProcess is IsValid(), continue
		
	m_vecMIValueTuple.clear();
	const MIuint nThreads = rProcess.GetNumThreads();
	for( MIuint i = 0; i < nThreads; i++ )
	{
		//	GetThreadAtIndex() uses a base 0 index
		//	GetThreadByIndexID() uses a base 1 index
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
bool CMICmdCmdListThreadGroups::Acknowledge( void )
{
	if( m_bHaveArgOption )
	{
		if( m_bHaveArgRecurse )
		{
			const CMICmnMIValueConst miValueConst( MIRSRC( IDS_WORD_NOT_IMPLEMENTED_BRKTS ) );
			const CMICmnMIValueResult miValueResult( "msg", miValueConst );
			const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
			m_miResultRecord = miRecordResult;
			
			return MIstatus::success;
		}

		const CMICmnMIValueConst miValueConst1( "i1" );
		const CMICmnMIValueResult miValueResult1( "id", miValueConst1 );
		CMICmnMIValueTuple miTuple( miValueResult1 );

		const CMICmnMIValueConst miValueConst2( "process" );
		const CMICmnMIValueResult miValueResult2( "type", miValueConst2 );
		miTuple.Add( miValueResult2 );

		CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
		const lldb::pid_t pid = rSessionInfo.m_lldbProcess.GetProcessID();
		const CMIUtilString strPid( CMIUtilString::Format( "%lld", pid ) );
		const CMICmnMIValueConst miValueConst3( strPid );
		const CMICmnMIValueResult miValueResult3( "pid", miValueConst3 );
		miTuple.Add( miValueResult3 );

		const CMICmnMIValueConst miValueConst4( MIRSRC( IDS_WORD_NOT_IMPLEMENTED_BRKTS ) );
		const CMICmnMIValueResult miValueResult4( "num_children", miValueConst4 );
		miTuple.Add( miValueResult4 );

		const CMICmnMIValueConst miValueConst5( MIRSRC( IDS_WORD_NOT_IMPLEMENTED_BRKTS ) );
		const CMICmnMIValueResult miValueResult5( "cores", miValueConst5 );
		miTuple.Add( miValueResult5 );

		const CMICmnMIValueList miValueList( miTuple );
		const CMICmnMIValueResult miValueResult6( "groups", miValueList );
		const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult6 );
		m_miResultRecord = miRecordResult;
		
		return MIstatus::success;
	}
	
	if( !m_bIsI1 )
	{
		const CMICmnMIValueConst miValueConst1( "i1" );
		const CMICmnMIValueResult miValueResult1( "id", miValueConst1 );
		CMICmnMIValueTuple miTuple( miValueResult1 );

		const CMICmnMIValueConst miValueConst2( "process" );
		const CMICmnMIValueResult miValueResult2( "type", miValueConst2 );
		miTuple.Add( miValueResult2 );

		CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
		const lldb::pid_t pid = rSessionInfo.m_lldbProcess.GetProcessID();
		const CMIUtilString strPid( CMIUtilString::Format( "%lld", pid ) );
		const CMICmnMIValueConst miValueConst3( strPid );
		const CMICmnMIValueResult miValueResult3( "pid", miValueConst3 );
		miTuple.Add( miValueResult3 );

		lldb::SBTarget & rTrgt = rSessionInfo.m_lldbTarget;
		const char * pDir = rTrgt.GetExecutable().GetDirectory();
		const char * pFileName = rTrgt.GetExecutable().GetFilename();
		const CMIUtilString strFile( CMIUtilString::Format( "%s/%s", pDir, pFileName ) );
		const CMICmnMIValueConst miValueConst4( strFile );
		const CMICmnMIValueResult miValueResult4( "executable", miValueConst4 );
		miTuple.Add( miValueResult4 );
		
		const CMICmnMIValueList miValueList( miTuple );
		const CMICmnMIValueResult miValueResult5( "groups", miValueList );
		const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult5 );
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
CMICmdBase * CMICmdCmdListThreadGroups::CreateSelf( void )
{
	return new CMICmdCmdListThreadGroups();
}