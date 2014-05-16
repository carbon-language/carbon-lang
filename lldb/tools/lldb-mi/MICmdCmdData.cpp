//===-- MICmdCmdData.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdData.cpp
//
// Overview:	CMICmdCmdDataEvaluateExpression	implementation.
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
#include "MICmdCmdData.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnMIOutOfBandRecord.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmnLLDBProxySBValue.h"
#include "MICmdArgContext.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataEvaluateExpression constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataEvaluateExpression::CMICmdCmdDataEvaluateExpression( void )
:	m_bExpressionValid( true )
,	m_bEvaluatedExpression( true )
,	m_strValue( "??" )
,	m_bCompositeVarType( false )
,	m_constStrArgThread( "thread" )
,	m_constStrArgFrame( "frame" )
,	m_constStrArgExpr( "expr" )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-evaluate-expression";
	
	// Required by the CMICmdFactory when registering *this commmand
	m_pSelfCreatorFn = &CMICmdCmdDataEvaluateExpression::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataEvaluateExpression destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataEvaluateExpression::~CMICmdCmdDataEvaluateExpression( void )
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
bool CMICmdCmdDataEvaluateExpression::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgFrame, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgExpr, true, true, true ) ) );
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
bool CMICmdCmdDataEvaluateExpression::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgExpr, String, m_constStrArgExpr );

	const CMIUtilString & rExpression( pArgExpr->GetValue() );
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	lldb::SBThread thread = rProcess.GetSelectedThread();
	lldb::SBFrame frame = thread.GetSelectedFrame();
	m_bExpressionValid = (thread.GetNumFrames() > 0);
	if( !m_bExpressionValid )
		return MIstatus::success;
	
	lldb::SBValue value = frame.EvaluateExpression( rExpression.c_str() );
	if( !value.IsValid() )
		value = frame.FindVariable( rExpression.c_str() );
	if( !value.IsValid() )
	{
		m_bEvaluatedExpression = false;
		return MIstatus::success;
	}

	MIuint64 nNumber = 0;
	if( CMICmnLLDBProxySBValue::GetValueAsUnsigned( value, nNumber ) == MIstatus::success )
	{
		const lldb::ValueType eValueType = value.GetValueType();
		m_strValue = (value.GetValue() != nullptr) ? value.GetValue() : "??";

		CMIUtilString strCString;
		if( CMICmnLLDBProxySBValue::GetCString( value, strCString ) )
		{
			m_strValue += CMIUtilString::Format( " '%s'", strCString.c_str() );
		}
		return MIstatus::success;
	}

	// Composite type i.e. struct
	m_bCompositeVarType = true;
	MIuint nChild = value.GetNumChildren();
	for( MIuint i = 0; i < nChild; i++ )
	{
		lldb::SBValue member = value.GetChildAtIndex( i );
		const bool bValid = member.IsValid();
		CMIUtilString strType( MIRSRC( IDS_WORD_UNKNOWNTYPE_BRKTS ) );
		if( bValid )
		{
			const CMIUtilString strValue( CMICmnLLDBDebugSessionInfoVarObj::GetValueStringFormatted( member, CMICmnLLDBDebugSessionInfoVarObj::eVarFormat_Natural ) );
			const char * pTypeName = member.GetName();
			if( pTypeName != nullptr )
				strType = pTypeName;

			// MI print "{varaiable = 1, variable2 = 3, variable3 = 5}"
			const bool bNoQuotes = true;
			const CMICmnMIValueConst miValueConst( strValue, bNoQuotes );
			const bool bUseSpaces = true;
			const CMICmnMIValueResult miValueResult( strType, miValueConst, bUseSpaces );
			m_miValueTuple.Add( miValueResult, bUseSpaces );
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
bool CMICmdCmdDataEvaluateExpression::Acknowledge( void )
{
	if( m_bExpressionValid )
	{
		if( m_bEvaluatedExpression )
		{
			if( m_bCompositeVarType )
			{
				const CMICmnMIValueConst miValueConst( m_miValueTuple.GetString() );
				const CMICmnMIValueResult miValueResult( "value", miValueConst );
				const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
				m_miResultRecord = miRecordResult;
				return MIstatus::success;
			}
			
			const CMICmnMIValueConst miValueConst( m_strValue );
			const CMICmnMIValueResult miValueResult( "value", miValueConst );
			const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
			m_miResultRecord = miRecordResult;
			return MIstatus::success;
		}

		const CMICmnMIValueConst miValueConst( "Could not evaluate expression" );
		const CMICmnMIValueResult miValueResult( "msg", miValueConst );
		const CMICmnMIResultRecord miRecordResult( m_cmdData.nMiCmdNumber, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
		m_miResultRecord = miRecordResult;
		return MIstatus::success;
	}

	const CMICmnMIValueConst miValueConst( "invalid expression" );
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
CMICmdBase * CMICmdCmdDataEvaluateExpression::CreateSelf( void )
{
	return new CMICmdCmdDataEvaluateExpression();
}
