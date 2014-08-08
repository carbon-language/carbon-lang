//===-- MICmdCmdGdbSet.cpp -------      -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdGdbSet.cpp
//
// Overview:	CMICmdCmdGdbSet	implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmdCmdGdbSet.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmdArgValString.h"
#include "MICmdArgValListOfN.h"
#include "MICmnLLDBDebugSessionInfo.h"

// Instantiations:
const CMICmdCmdGdbSet::MapGdbOptionNameToFnGdbOptionPtr_t CMICmdCmdGdbSet::ms_mapGdbOptionNameToFnGdbOptionPtr =
{
	// { "target-async", &CMICmdCmdGdbSet::OptionFnTargetAsync },		// Example code if need to implement GDB set other options
	// { "auto-solib-add", &CMICmdCmdGdbSet::OptionFnAutoSolibAdd },	// Example code if need to implement GDB set other options
	{ "solib-search-path", &CMICmdCmdGdbSet::OptionFnSolibSearchPath },
	{ "fallback", &CMICmdCmdGdbSet::OptionFnFallback }
};

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdGdbSet constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdGdbSet::CMICmdCmdGdbSet( void )
:	m_constStrArgNamedGdbOption( "option" )
,	m_bGdbOptionRecognised( true ) 
,	m_bGdbOptionFnSuccessful( false )
,	m_bGbbOptionFnHasError( false )
,	m_strGdbOptionFnError( MIRSRC( IDS_WORD_ERR_MSG_NOT_IMPLEMENTED_BRKTS ) )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "gdb-set";
	
	// Required by the CMICmdFactory when registering *this command
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
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdGdbSet::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValListOfN( m_constStrArgNamedGdbOption, true, true, CMICmdArgValListBase::eArgValType_StringQuotedNumberPath ) ) );
	return (bOk && ParseValidateCmdOptions() );
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
	CMICMDBASE_GETOPTION( pArgGdbOption, ListOfN, m_constStrArgNamedGdbOption );
	const CMICmdArgValListBase::VecArgObjPtr_t & rVecWords( pArgGdbOption->GetExpectedOptions() );
	
	// Get the gdb-set option to carry out
	CMICmdArgValListBase::VecArgObjPtr_t::const_iterator it = rVecWords.begin();
	const CMICmdArgValString * pOption = static_cast< const CMICmdArgValString * >( *it );
	const CMIUtilString strOption( pOption->GetValue() );
	++it;

	// Retrieve the parameter(s) for the option
	CMIUtilString::VecString_t vecWords;
	while( it != rVecWords.end() )
	{
		const CMICmdArgValString * pWord = static_cast< const CMICmdArgValString * >( *it );
		vecWords.push_back( pWord->GetValue() );

		// Next
		++it;
	}

	FnGdbOptionPtr pPrintRequestFn = nullptr;
	if( !GetOptionFn( strOption, pPrintRequestFn ) )
	{
		// For unimplemented option handlers, fallback on a generic handler
		// ToDo: Remove this when ALL options have been implemented
		if( !GetOptionFn( "fallback", pPrintRequestFn ) )
		{
			m_bGdbOptionRecognised = false;
			m_strGdbOptionName = "fallback"; // This would be the strOption name
			return MIstatus::success;
		}
	}

	m_bGdbOptionFnSuccessful = (this->*(pPrintRequestFn))( vecWords );
	if( !m_bGdbOptionFnSuccessful && !m_bGbbOptionFnHasError )
		return MIstatus::failure;

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
	if( !m_bGdbOptionRecognised )
	{
		const CMICmnMIValueConst miValueConst( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_INFO_PRINTFN_NOT_FOUND ), m_strGdbOptionName.c_str() ) );
		const CMICmnMIValueResult miValueResult( "msg", miValueConst );
		const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
		m_miResultRecord = miRecordResult;
		return MIstatus::success;
	}
	
	if( m_bGdbOptionFnSuccessful )
	{
		const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done );
		m_miResultRecord = miRecordResult;
		return MIstatus::success;
	}

	const CMICmnMIValueConst miValueConst( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_INFO_PRINTFN_FAILED ), m_strGdbOptionFnError.c_str() ) );
	const CMICmnMIValueResult miValueResult( "msg", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
	m_miResultRecord = miRecordResult;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
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

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the print function's pointer for the matching print request.
// Type:	Method.
// Args:	vrPrintFnName	- (R) The info requested.
//			vrwpFn			- (W) The print function's pointer of the function to carry out 
// Return:	bool	- True = Print request is implemented, false = not found.
// Throws:	None.
//--
bool CMICmdCmdGdbSet::GetOptionFn( const CMIUtilString & vrPrintFnName, FnGdbOptionPtr & vrwpFn ) const
{
	vrwpFn = nullptr;

	const MapGdbOptionNameToFnGdbOptionPtr_t::const_iterator it = ms_mapGdbOptionNameToFnGdbOptionPtr.find( vrPrintFnName );
	if( it != ms_mapGdbOptionNameToFnGdbOptionPtr.end() )
	{
		vrwpFn = (*it).second;
		return true;
	}

	return false;
}

//++ ------------------------------------------------------------------------------------
// Details:	Carry out work to complete the GDB set option 'solib-search-path' to prepare 
//			and send back information asked for.
// Type:	Method.
// Args:	vrWords	- (R) List of additional parameters used by this option.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdGdbSet::OptionFnSolibSearchPath( const CMIUtilString::VecString_t & vrWords )
{
	// Check we have at least one argument
	if( vrWords.size() < 1 )
	{
		m_bGbbOptionFnHasError = true;
		m_strGdbOptionFnError = MIRSRC( IDS_CMD_ERR_GDBSET_OPT_SOLIBSEARCHPATH );
		return MIstatus::failure;
	}
	const CMIUtilString & rStrValSolibPath( vrWords[ 0 ] );

	// Add 'solib-search-path' to the shared data list
	const CMIUtilString & rStrKeySolibPath( m_rLLDBDebugSessionInfo.m_constStrSharedDataSolibPath );
	if( !m_rLLDBDebugSessionInfo.SharedDataAdd< CMIUtilString >( rStrKeySolibPath, rStrValSolibPath ) )
	{
		m_bGbbOptionFnHasError = false;
		SetError( CMIUtilString::Format( MIRSRC( IDS_DBGSESSION_ERR_SHARED_DATA_ADD ), m_cmdData.strMiCmd.c_str(), rStrKeySolibPath.c_str() ) );
		return MIstatus::failure;
	}	

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Carry out work to complete the GDB set option to prepare and send back information
//			asked for.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdGdbSet::OptionFnFallback( const CMIUtilString::VecString_t & vrWords )
{
	MIunused( vrWords );

	// Do nothing - intentional. This is a fallback temporary action function to do nothing.
	// This allows the search for gdb-set options to always suceed when the option is not 
	// found (implemented).

	return MIstatus::success;
}
