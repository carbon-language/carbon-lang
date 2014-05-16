//===-- MICmdCmdBreak.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdBreak.cpp
//
// Overview:	CMICmdCmdBreakInsert			implementation.
//				CMICmdCmdBreakDelete			implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third Party Headers:
#include <lldb/API/SBBreakpointLocation.h>

// In-house headers:
#include "MICmnConfig.h"
#include "MICmdCmdBreak.h"
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
// Details:	CMICmdCmdBreakInsert constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdBreakInsert::CMICmdCmdBreakInsert( void )
:	m_bBrkPtIsTemp( false )
,	m_brkName()
,	m_constStrArgNamedTempBrkPt( "t" )
,	m_constStrArgNamedHWBrkPt( "h" )
,	m_constStrArgNamedPendinfBrkPt( "f" )
,	m_constStrArgNamedDisableBrkPt( "d" )
,	m_constStrArgNamedTracePt( "a" )
,	m_constStrArgNamedConditionalBrkPt( "c" )
,	m_constStrArgNamedInoreCnt( "i" )
,	m_constStrArgNamedRestrictBrkPtToThreadId( "p" )
,	m_constStrArgNamedLocation( "location" )
,	m_constStrArgNamedThreadGroup( "thread-group" )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "break-insert";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdBreakInsert::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdBreakInsert destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdBreakInsert::~CMICmdCmdBreakInsert( void )
{
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
bool CMICmdCmdBreakInsert::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedTempBrkPt, false, true )) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedHWBrkPt, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedPendinfBrkPt, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedDisableBrkPt, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedTracePt, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedConditionalBrkPt, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedInoreCnt, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgNamedRestrictBrkPtToThreadId, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgNamedLocation, false, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgNamedThreadGroup, false, true, CMICmdArgValListBase::eArgValType_ThreadGrp, 1 ) ) ); 
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
bool CMICmdCmdBreakInsert::Execute( void )
{
	// Note not all arguments required ATM based on C's version off the code
	CMICMDBASE_GETOPTION( pArgTempBrkPt, OptionShort, m_constStrArgNamedTempBrkPt );
	CMICMDBASE_GETOPTION( pArgThreadGroup, OptionLong, m_constStrArgNamedThreadGroup );
	CMICMDBASE_GETOPTION( pArgLocation, String, m_constStrArgNamedLocation );
	
	m_bBrkPtIsTemp = pArgTempBrkPt->GetFound();
	m_bHaveArgOptionThreadGrp = pArgThreadGroup->GetFound();
	if( m_bHaveArgOptionThreadGrp )
	{
		const CMICmdArgValOptionLong::VecArgObjPtr_t & rVecOptions( pArgThreadGroup->GetExpectedOptions() );
		const CMICmdArgValThreadGrp * pThreadGrp = (rVecOptions.size() > 0) ? static_cast< CMICmdArgValThreadGrp * >( rVecOptions[ 0 ] ) : nullptr;
		const MIuint nThreadGrp = (pThreadGrp != nullptr) ? pThreadGrp->GetValue() : 0;
		m_strArgOptionThreadGrp = CMIUtilString::Format( "%d", nThreadGrp );
	}
	m_brkName = pArgLocation->GetFound() ? pArgLocation->GetValue() : "";

	// Determine if break on a file line or at a function
	BreakPoint_e eBrkPtType = eBreakPoint_NotDefineYet;
	const CMIUtilString cColon = ":";
	bool bIsFileLine = false;
	bool bIsFileFn = false;
	CMIUtilString fileName;
	MIuint nFileLine = 0;
	CMIUtilString strFileFn;
	const MIint nPosColon = m_brkName.find( cColon );
	if( nPosColon != (MIint) std::string::npos )
	{
		CMIUtilString::VecString_t vecFileAndLocation;
		const MIuint nSplits = m_brkName.Split( cColon, vecFileAndLocation );
		if( vecFileAndLocation.size() != 2 )
		{
			SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_BRKPT_LOCATION_FORMAT ), m_cmdData.strMiCmd.c_str(), m_brkName.c_str() ) );
			return MIstatus::failure;
		}
		fileName = vecFileAndLocation.at( 0 );
		const CMIUtilString & rStrLineOrFn( vecFileAndLocation.at( 1 ) );
		if( rStrLineOrFn.empty() )
			eBrkPtType = eBreakPoint_ByName;
		else
		{
			MIint64 nValue = 0;
			if( rStrLineOrFn.ExtractNumber( nValue ) )
			{
				nFileLine = static_cast< MIuint >( nValue );
				eBrkPtType = eBreakPoint_ByFileLine;
			}
			else
			{
				strFileFn = rStrLineOrFn;
				eBrkPtType = eBreakPoint_ByFileFn;
			}
		}
	}

	// Determine if break defined as an address
	lldb::addr_t nAddress = 0;
	if( eBrkPtType == eBreakPoint_NotDefineYet ) 
	{
		MIint64 nValue = 0;
		if( m_brkName.ExtractNumber( nValue ) )
		{
			nAddress = static_cast< lldb::addr_t >( nValue );
			eBrkPtType = eBreakPoint_ByAddress;
		}
	}
	
	// Break defined as an function
	if( eBrkPtType == eBreakPoint_NotDefineYet ) 
	{
		eBrkPtType = eBreakPoint_ByName;
	}

	// Ask LLDB to create a breakpoint
	bool bOk = MIstatus::success;
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBTarget & rTarget = rSessionInfo.m_lldbTarget;
	switch( eBrkPtType ) 
	{
	case eBreakPoint_ByAddress:
		m_brkPt = rTarget.BreakpointCreateByAddress( nAddress );
		break;
	case eBreakPoint_ByFileFn:
		m_brkPt = rTarget.BreakpointCreateByName( strFileFn.c_str(), fileName.c_str() );
		break;
	case eBreakPoint_ByFileLine:
		m_brkPt = rTarget.BreakpointCreateByLocation( fileName.c_str(), nFileLine );
		break;
	case eBreakPoint_ByName:
		m_brkPt = rTarget.BreakpointCreateByName( m_brkName.c_str(), rTarget.GetExecutable().GetFilename() );
		break;
	case eBreakPoint_count:
	case eBreakPoint_NotDefineYet:
	case eBreakPoint_Invalid:
		bOk = MIstatus::failure;
		break;
	}
	if( !m_brkPt.IsValid() || !bOk )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_BRKPT_INVALID ), m_cmdData.strMiCmd.c_str(), m_brkName.c_str() ) );
		return MIstatus::failure;
	}
	if( m_brkPt.GetID() > (lldb::break_id_t) rSessionInfo.m_nBrkPointCntMax )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_BRKPT_CNT_EXCEEDED ), m_cmdData.strMiCmd.c_str(), rSessionInfo.m_nBrkPointCntMax, m_brkName.c_str() ) );
		return MIstatus::failure;
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
bool CMICmdCmdBreakInsert::Acknowledge( void )
{
	// Get breakpoint information
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBTarget & rTarget = rSessionInfo.m_lldbTarget;
	lldb::SBBreakpointLocation brkPtLoc = m_brkPt.GetLocationAtIndex( 0 );
	lldb::SBAddress brkPtAddr = brkPtLoc.GetAddress();
	lldb::SBSymbolContext symbolCntxt = brkPtAddr.GetSymbolContext( lldb::eSymbolContextEverything );
	const char * pUnkwn = "??";
	lldb::SBModule rModule = symbolCntxt.GetModule();
	const char * pModule = rModule.IsValid() ? rModule.GetFileSpec().GetFilename() : pUnkwn;
	const char * pFile = pUnkwn;
	const char * pFn = pUnkwn;
	const char * pFilePath = pUnkwn;
	size_t nLine = 0;
	const size_t nAddr = brkPtAddr.GetLoadAddress( rTarget );

	lldb::SBCompileUnit rCmplUnit = symbolCntxt.GetCompileUnit();
	if( rCmplUnit.IsValid() )
	{
		lldb::SBFileSpec rFileSpec = rCmplUnit.GetFileSpec();
		pFile = rFileSpec.GetFilename();
		pFilePath = rFileSpec.GetDirectory();
		lldb::SBFunction rFn = symbolCntxt.GetFunction();
		if( rFn.IsValid() )
			pFn = rFn.GetName();
		lldb::SBLineEntry rLnEntry = symbolCntxt.GetLineEntry();
		if( rLnEntry.GetLine() > 0 )
			nLine = rLnEntry.GetLine();
	}
	
	// Form MI result record example
	// MI print "=breakpoint-modified,bkpt={number=\"%d\",type=\"breakpoint\",disp=\"%s\",enabled=\"%c\",addr=\"0x%08x\", func=\"%s\",file=\"%s\",fullname=\"%s/%s\",line=\"%d\",times=\"%d\",original-location=\"%s\"}"
	CMICmnMIValueTuple miValueTuple;
	if( !CMICmnLLDBDebugSessionInfo::Instance().MIResponseFormBrkPtInfo(	m_brkPt.GetID(),			// "number="
																			"breakpoint",				// "type="
																			m_bBrkPtIsTemp,				// "disp="	
																			m_brkPt.IsEnabled(),		// "enabled="
																			nAddr,						// "addr="
																			pFn,						// "func="
																			pFile,						// "file="
																			pFilePath,					// "fullname="
																			nLine,						// "line="
																			m_bHaveArgOptionThreadGrp,	// 
																			m_strArgOptionThreadGrp,	// "thread-groups="
																			m_brkPt.GetNumLocations(),	// "times="
																			m_brkName,					// "original-location="
																			miValueTuple ))				
	{
		return MIstatus::failure;
	}

	const CMICmnMIValueResult miValueResultD( "bkpt", miValueTuple );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResultD );
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
CMICmdBase * CMICmdCmdBreakInsert::CreateSelf( void )
{
	return new CMICmdCmdBreakInsert();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdBreakDelete constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdBreakDelete::CMICmdCmdBreakDelete( void )
:	m_constStrArgNamedBrkPt( "breakpoint" )
,	m_constStrArgNamedThreadGrp( "thread-group")
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "break-delete";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdBreakDelete::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdBreakDelete destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdBreakDelete::~CMICmdCmdBreakDelete( void )
{
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
bool CMICmdCmdBreakDelete::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgNamedThreadGrp, false, false, CMICmdArgValListBase::eArgValType_ThreadGrp, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValListOfN( m_constStrArgNamedBrkPt, true, true, CMICmdArgValListBase::eArgValType_Number ) ) );
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
bool CMICmdCmdBreakDelete::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgBrkPt, ListOfN, m_constStrArgNamedBrkPt );

	// ATM we only handle one break point ID
	MIuint64 nBrk = UINT64_MAX;
	if( !pArgBrkPt->GetExpectedOption< CMICmdArgValNumber, MIuint64 >( nBrk ) )
	{
		const CMIUtilString errMsg( CMIUtilString::Format( MIRSRC( IDS_CMD_ARGS_ERR_N_OPTIONS_REQUIRED ), 1 ) );
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_BRKPT_INVALID ), m_cmdData.strMiCmd.c_str(), errMsg.c_str() ) );
		return MIstatus::failure;
	}

	const CMIUtilString strBrkNum( CMIUtilString::Format( "%d", nBrk ) );
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	const bool bBrkPt = rSessionInfo.m_lldbTarget.BreakpointDelete( static_cast< lldb::break_id_t >( nBrk ) );
	if( !bBrkPt )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_BRKPT_INVALID ), m_cmdData.strMiCmd.c_str(), strBrkNum.c_str() ) );
		return MIstatus::failure;
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
bool CMICmdCmdBreakDelete::Acknowledge( void )
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
CMICmdBase * CMICmdCmdBreakDelete::CreateSelf( void )
{
	return new CMICmdCmdBreakDelete();
}
